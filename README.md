<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/brand/cuteafd-logo-color-dark.svg">
    <img src="assets/brand/cuteafd-logo-color.svg" alt="cuteafd" width="480">
  </picture>
</p>

CuteAFD brings frontier-scale open-weights models into the home lab at
data-center speed. It disaggregates attention from the routed experts: one or
two consumer-Blackwell RTX PRO 6000 cards run attention, the dense
backbone, routing and sampling, while a pool of DGX Sparks (GB10, SM121) holds
the routed experts and answers over RoCE. The same engine loads a model's
standard Hugging Face checkpoint directly — no side files, no repacking — and
runs it on the formats it actually ships in: official FP8 and MXFP4,
NVIDIA ModelOpt NVFP4, and EXL3, with native kernels that honor each
checkpoint's own numerics instead of converting everything to one internal
format.

- Attention/FFN disaggregation (AFD): RTX cards own the backbone, Sparks own
  the experts, exchanging activations over RoCE with GPU-direct landing.
- Robust quant support: official FP8/MXFP4, NVIDIA ModelOpt NVFP4, and EXL3,
  loaded from the checkpoint's own `config.json` and tensor headers.
- Sliced HF snapshots need no annotation: MiMo, GLM Flash and Qwen coordinators
  need only their role's shards; official V4.1 Spark workers need only expert shards.
  `cuteafd plan MODEL --files --role coordinator --spark-ranks 4` emits an rsync
  file list; add `--fetch --destination /path/to/snapshot --dry-run` to preview a
  role-only copy (omit `--dry-run` to copy). Transfers materialize HF blob symlinks
  as plain snapshot files; matching sizes are skipped and nothing is deleted.
  For host unions, `--file-layout layout.json --host worker` accepts a JSON host-to-role
  map such as `{"worker":["spark0","vision"]}`; omit `--host` with `--json` to list all
  hosts. `--host worker --fetch` copies via SSH; add `--source peer:/snapshot` to
  pull from a peer (MODEL supplies the local index/config). Remote size checks are
  batched into one SSH session before and after copying. `--forward-agent` opts into
  `ssh -A` for peer-to-target copies (default off). `--source auto` reports whether
  sparknest serves a sealed local copy or streams; without it, the local snapshot
  is used. Omit `--host` with `--file-layout --fetch` to copy the whole layout,
  capped by `--fetch-parallel` (default 2). Add `--drafter-snapshot`, `--vision-snapshot`
  or `--audio-snapshot` with `--json` to include separate repos for the selected
  host's enabled roles; fetch those snapshot roots individually.
- Exact prefix caching for agentic work: the deepest cached snapshot that
  prefixes a request is restored byte-identical, not approximated.
- Own your intelligence: your weights, your hardware, your rate limits (none),
  agentic coding at full speed on a machine you control, not a shared tenant.

## v1.0.0 changes since v0.1.0

- MiMo Pro uses the MOPD checkpoint. Single-copy FP8 defaults cover the V4.1
  vocabulary head, MiMo head/o_proj/drafter, GLM drafters, GLM Flash per-layout
  projections and Qwen head.
- Qwen gains local experts and MTP3, including mixed NVFP4 backbone / FP8 MTP.
  GLM Flash gains a two-RTX head split and DFlash2 on every layout.
- MiMo Pro prefill gains lanes under head split, A8 down projection and exact
  Spark slices; GLM 5.3 gains FP8 MLA prefill and faster Spark kernels.
- Tool-call grammar fixes, automatic KV pools and planner core, a live console
  for every family, and the benchmark dashboard. Device-driven exchange ships
  opt-in. MXFP4 32-row tails and V4.1 exact slices move to v1.x.

## WIP Hardware Cards

Use `scripts/bench/wip-cards.py` instead of copying an RC kit driver. It reads
`cfg/*.config` and `matrix-*.json` from the kit (default: release-v2-rc2), writes
private one-card matrices and a 0600 API key, and runs `cuteafd bench smoke`
under `setsid` with numeric exit files. Smoke alone owns serving locks. Use a
current CLI containing the opt-in observer/precheck hooks for probes.

```sh
python3 scripts/bench/wip-cards.py --interleave --repeats 3 \
  --cards v41-flash-min --arm baseline=my-base:slot --arm candidate=my-new:slot \
  --task v41-ab --probe v41-flash-min:image,memory,console \
  --expect-pool v41-flash-min=2097152 --dry-run
python3 scripts/bench/wip-cards.py --matrix --parallel 2 \
  --cards v41-flash-sim5090 glm53f-exl3-sim5090 \
  --arm left=my-left:slot --arm right=my-right:slot \
  --card-arm v41-flash-sim5090=left --card-arm glm53f-exl3-sim5090=right --dry-run
```

Remove `--dry-run` to execute. `--set CARD:KEY=VALUE` overrides serving keys.
Parallel mode accepts only simulated or explicitly correctness-only cards
(`kind`: `sim-5090`, `fidelity`, `cache`, `image`, `no-fit`, `packed-check`),
refuses shared GPUs/Sparks/ports/instances and published/compared cards, and
withholds incidental performance numbers. Shared-arm parallel cards use one
smoke scheduler; separately bound arms use independent smoke processes with
disjoint lock sets. Simulated timings never enter
medians or paired deltas. Results include JSON, a markdown table, hook logs,
raw/corrected CUDA 50ms peaks (NVML sampler-context subtraction), worker ring
ledger peaks and authenticated console stage events. Explicit probes fail
rather than silently claim missing evidence. Pool admission and red-square
image checks run after readiness, before the benchmark. Explicit `panels:`
selections are validated before launch and checked against the finished report.
Serving instance names use a short hash suffix to stay within run.sh's 41-byte
limit; WIP arm identities remain unchanged. `--interleave --matched-prompts`
replays the same `CUTEAFD_BENCH_NONCE_SEED` sequence across fresh arm servers,
with a new recorded seed per pair. Basic/decode-content requests remain greedy
with their fixed token limits. Authenticated console rounds report per-request
emitted tok/s (first-token-to-retirement decode time), tokens/step and ms/step
(round service time); round-service tok/s is reported separately. Missing or different first-request token hashes fail the pair. This
requires arm images with opt-in prompt-hash telemetry; release cards without
the flag keep their existing prompts.

Optional `--build baseline=REV --build candidate=REV` freezes separate source
worktrees, uses identical family scopes, stages sealed slots to the union of
selected Spark hosts, and checks every card's local/remote seals before launch.
Builds use `build.lock`, nice 19 and 16 CPU jobs. Sources with opt-in export
locking take the specifically mounted raptor GPU lock only during native/AOT
work. Legacy frozen sources hold that lock for the full build instead; the
mode is logged and recorded per arm. No scripts are overlaid into frozen arms.
A busy Spark seed is
refused using a raptor-side lock check (not an atomic reservation; operators
must still keep the seed idle throughout export). `--cleanup` verifies and
removes only task-owned build roots/containers, overlays and registrations on
raptor and all six Sparks; existing arms without this task's ownership marker
are deliberately refused, never adopted. Keep the same `--task`, `--state`
and arms for cleanup. Source worktrees are retained for inspection.

## Models

Basic benchmark profile per family on its natural-minimum (1× RTX + fewest
Sparks) and maximum (2× RTX + 4 or 6 Sparks) hardware. Other reports:
[`benchmarks/`](benchmarks/README.md).

All 28 required release-prep smoke cards pass, including Qwen NVFP4 with
resident local MTP3 on one RTX. The grid contains four refreshed RC2 cards
and 24 retained RC1 cards. See [qualification and known limits](PLAN.md#v1-regression-follow-up-rc2-2026-10-05)
and the [release scope](PLAN.md#release-v1-scope-decided-2026-10-04).

MiMo V2.6 Flash MOPD replaces the legacy Flash current rows after text-only
qualification on a task overlay, not a new release cut. Its cards cover 16K
context / 32K pool; [conditions and limits](docs/models/mimo_v2.md#flash-mopd-qualification-2026-10-05)
include the separate three-run C1/C4 comparison and reasoning-on completion gate.

<!-- results:begin -->
<table>
<tr><th>Model · quant</th><th>Minimum hardware</th><th>Maximum hardware</th></tr>
<tr>
<td width="20%" valign="top"><a href="docs/models/deepseek_v41.md"><b>DeepSeek V4.1</b></a><br><sub>deepseek-ai/DeepSeek-V4.1-Flash</sub><br><sub>mxfp4-g32</sub></td>
<td width="40%" valign="top"><a href="benchmarks/deepseek_v41/2026-10-04-smoke-deepseek-v4-1-flash-1rtx-4spark-v41-flash-min/card.svg"><img src="benchmarks/deepseek_v41/2026-10-04-smoke-deepseek-v4-1-flash-1rtx-4spark-v41-flash-min/card.svg" alt="deepseek-ai/DeepSeek-V4.1-Flash on 1× RTX PRO 6000 @ 325 W + 4× DGX Spark"></a><br><sub>1× RTX + 4× Spark</sub></td>
<td width="40%" valign="top"><a href="benchmarks/deepseek_v41/2026-10-05-smoke-deepseek-v4-1-flash-2rtx-4spark-rc2/card.svg"><img src="benchmarks/deepseek_v41/2026-10-05-smoke-deepseek-v4-1-flash-2rtx-4spark-rc2/card.svg" alt="deepseek-ai/DeepSeek-V4.1-Flash on 2× RTX PRO 6000 @ 325 W + 4× DGX Spark"></a><br><sub>2× RTX + 4× Spark</sub></td>
</tr>
<tr>
<td width="20%" valign="top"><a href="docs/models/deepseek_v41.md"><b>DeepSeek V4.1</b></a><br><sub>nvidia/DeepSeek-V4.1-Flash-NVFP4</sub><br><sub>nvfp4-g16</sub></td>
<td width="40%" valign="top"><a href="benchmarks/deepseek_v41/2026-10-04-smoke-deepseek-v4-1-flash-nvfp4-1rtx-4spark-v41-flash-nvfp4-min/card.svg"><img src="benchmarks/deepseek_v41/2026-10-04-smoke-deepseek-v4-1-flash-nvfp4-1rtx-4spark-v41-flash-nvfp4-min/card.svg" alt="nvidia/DeepSeek-V4.1-Flash-NVFP4 on 1× RTX PRO 6000 @ 325 W + 4× DGX Spark"></a><br><sub>1× RTX + 4× Spark</sub></td>
<td width="40%" valign="top"><a href="benchmarks/deepseek_v41/2026-10-04-smoke-deepseek-v4-1-flash-nvfp4-2rtx-4spark-v41-flash-nvfp4-max/card.svg"><img src="benchmarks/deepseek_v41/2026-10-04-smoke-deepseek-v4-1-flash-nvfp4-2rtx-4spark-v41-flash-nvfp4-max/card.svg" alt="nvidia/DeepSeek-V4.1-Flash-NVFP4 on 2× RTX PRO 6000 @ 325 W + 4× DGX Spark"></a><br><sub>2× RTX + 4× Spark</sub></td>
</tr>
<tr>
<td width="20%" valign="top"><a href="docs/models/deepseek_v4.md"><b>DeepSeek V4</b></a><br><sub>deepseek-ai/DeepSeek-V4-Flash-0731</sub><br><sub>mxfp4-g32</sub></td>
<td width="40%" valign="top"><a href="benchmarks/deepseek_v4/2026-10-04-smoke-deepseek-v4-flash-0731-1rtx-2spark-v4-flash-min/card.svg"><img src="benchmarks/deepseek_v4/2026-10-04-smoke-deepseek-v4-flash-0731-1rtx-2spark-v4-flash-min/card.svg" alt="deepseek-ai/DeepSeek-V4-Flash-0731 on 1× RTX PRO 6000 @ 325 W + 2× DGX Spark"></a><br><sub>1× RTX + 2× Spark</sub></td>
<td width="40%" valign="top"><a href="benchmarks/deepseek_v4/2026-10-04-smoke-deepseek-v4-flash-0731-2rtx-4spark-v4-flash-max/card.svg"><img src="benchmarks/deepseek_v4/2026-10-04-smoke-deepseek-v4-flash-0731-2rtx-4spark-v4-flash-max/card.svg" alt="deepseek-ai/DeepSeek-V4-Flash-0731 on 2× RTX PRO 6000 @ 325 W + 4× DGX Spark"></a><br><sub>2× RTX + 4× Spark</sub></td>
</tr>
<tr>
<td width="20%" valign="top"><a href="docs/models/deepseek_v4.md"><b>DeepSeek V4</b></a><br><sub>wrldsuksgo2mars/DeepSeek-V4-Pro-0813-EXL3-K2-calibrated-v1</sub><br><sub>exl3-k2</sub></td>
<td width="40%" valign="top"><a href="benchmarks/deepseek_v4/2026-10-04-smoke-deepseek-v4-pro-0813-exl3-k2-calibrated-v1-1rtx-4spark-v4-pro-exl3-min/card.svg"><img src="benchmarks/deepseek_v4/2026-10-04-smoke-deepseek-v4-pro-0813-exl3-k2-calibrated-v1-1rtx-4spark-v4-pro-exl3-min/card.svg" alt="wrldsuksgo2mars/DeepSeek-V4-Pro-0813-EXL3-K2-calibrated-v1 on 1× RTX PRO 6000 @ 325 W + 4× DGX Spark"></a><br><sub>1× RTX + 4× Spark</sub></td>
<td width="40%" valign="top"><a href="benchmarks/deepseek_v4/2026-10-04-smoke-deepseek-v4-pro-0813-exl3-k2-calibrated-v1-2rtx-6spark-v4-pro-exl3-max/card.svg"><img src="benchmarks/deepseek_v4/2026-10-04-smoke-deepseek-v4-pro-0813-exl3-k2-calibrated-v1-2rtx-6spark-v4-pro-exl3-max/card.svg" alt="wrldsuksgo2mars/DeepSeek-V4-Pro-0813-EXL3-K2-calibrated-v1 on 2× RTX PRO 6000 @ 325 W + 6× DGX Spark"></a><br><sub>2× RTX + 6× Spark</sub></td>
</tr>
<tr>
<td width="20%" valign="top"><a href="docs/models/glm5.md"><b>GLM 5.3</b></a><br><sub>wrldsuksgo2mars/GLM-5.3-EXL3-K4-v1</sub><br><sub>exl3-k4</sub></td>
<td width="40%" valign="top"><a href="benchmarks/glm5/2026-10-04-smoke-glm-5-3-exl3-k4-v1-1rtx-4spark-glm53-exl3-min/card.svg"><img src="benchmarks/glm5/2026-10-04-smoke-glm-5-3-exl3-k4-v1-1rtx-4spark-glm53-exl3-min/card.svg" alt="wrldsuksgo2mars/GLM-5.3-EXL3-K4-v1 on 1× RTX PRO 6000 @ 325 W + 4× DGX Spark"></a><br><sub>1× RTX + 4× Spark</sub></td>
<td width="40%" valign="top"><a href="benchmarks/glm5/2026-10-04-smoke-glm-5-3-exl3-k4-v1-2rtx-6spark-glm53-exl3-max/card.svg"><img src="benchmarks/glm5/2026-10-04-smoke-glm-5-3-exl3-k4-v1-2rtx-6spark-glm53-exl3-max/card.svg" alt="wrldsuksgo2mars/GLM-5.3-EXL3-K4-v1 on 2× RTX PRO 6000 @ 325 W + 6× DGX Spark"></a><br><sub>2× RTX + 6× Spark</sub></td>
</tr>
<tr>
<td width="20%" valign="top"><a href="docs/models/glm5.md"><b>GLM 5.3</b></a><br><sub>nvidia/GLM-5.3-NVFP4</sub><br><sub>nvfp4-g16</sub></td>
<td width="40%" valign="top"><a href="benchmarks/glm5/2026-10-04-smoke-glm-5-3-nvfp4-1rtx-6spark-glm53-nvfp4-min/card.svg"><img src="benchmarks/glm5/2026-10-04-smoke-glm-5-3-nvfp4-1rtx-6spark-glm53-nvfp4-min/card.svg" alt="nvidia/GLM-5.3-NVFP4 on 1× RTX PRO 6000 @ 325 W + 6× DGX Spark"></a><br><sub>1× RTX + 6× Spark</sub></td>
<td width="40%" valign="top"><a href="benchmarks/glm5/2026-10-04-smoke-glm-5-3-nvfp4-2rtx-6spark-glm53-nvfp4-max/card.svg"><img src="benchmarks/glm5/2026-10-04-smoke-glm-5-3-nvfp4-2rtx-6spark-glm53-nvfp4-max/card.svg" alt="nvidia/GLM-5.3-NVFP4 on 2× RTX PRO 6000 @ 325 W + 6× DGX Spark"></a><br><sub>2× RTX + 6× Spark</sub></td>
</tr>
<tr>
<td width="20%" valign="top"><a href="docs/models/glm5_flash.md"><b>GLM 5.3 Flash</b></a><br><sub>zai-org/GLM-5.3-Flash</sub><br><sub>fp8-block128x128/f32</sub></td>
<td width="40%" valign="top"><a href="benchmarks/glm5_flash/2026-10-04-smoke-glm-5-3-flash-1rtx-4spark-glm53f-fp8-min/card.svg"><img src="benchmarks/glm5_flash/2026-10-04-smoke-glm-5-3-flash-1rtx-4spark-glm53f-fp8-min/card.svg" alt="zai-org/GLM-5.3-Flash on 1× RTX PRO 6000 @ 325 W + 4× DGX Spark"></a><br><sub>1× RTX + 4× Spark</sub></td>
<td width="40%" valign="top"><a href="benchmarks/glm5_flash/2026-10-04-smoke-glm-5-3-flash-2rtx-4spark-glm53f-fp8-max/card.svg"><img src="benchmarks/glm5_flash/2026-10-04-smoke-glm-5-3-flash-2rtx-4spark-glm53f-fp8-max/card.svg" alt="zai-org/GLM-5.3-Flash on 2× RTX PRO 6000 @ 325 W + 4× DGX Spark"></a><br><sub>2× RTX + 4× Spark</sub></td>
</tr>
<tr>
<td width="20%" valign="top"><a href="docs/models/glm5_flash.md"><b>GLM 5.3 Flash</b></a><br><sub>wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3.25-v1</sub><br><sub>exl3-k3+exl3-k4</sub></td>
<td width="40%" valign="top"><a href="benchmarks/glm5_flash/2026-10-04-smoke-glm-5-3-flash-exl3-k3-25-v1-1rtx-2spark-glm53f-exl3-min/card.svg"><img src="benchmarks/glm5_flash/2026-10-04-smoke-glm-5-3-flash-exl3-k3-25-v1-1rtx-2spark-glm53f-exl3-min/card.svg" alt="wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3.25-v1 on 1× RTX PRO 6000 @ 325 W + 2× DGX Spark"></a><br><sub>1× RTX + 2× Spark</sub></td>
<td width="40%" valign="top"><a href="benchmarks/glm5_flash/2026-10-04-smoke-glm-5-3-flash-exl3-k3-25-v1-2rtx-4spark-glm53f-exl3-max/card.svg"><img src="benchmarks/glm5_flash/2026-10-04-smoke-glm-5-3-flash-exl3-k3-25-v1-2rtx-4spark-glm53f-exl3-max/card.svg" alt="wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3.25-v1 on 2× RTX PRO 6000 @ 325 W + 4× DGX Spark"></a><br><sub>2× RTX + 4× Spark</sub></td>
</tr>
<tr>
<td width="20%" valign="top"><a href="docs/models/glm5_flash.md"><b>GLM 5.3 Flash</b></a><br><sub>nvidia/GLM-5.3-Flash-NVFP4</sub><br><sub>nvfp4-g16</sub></td>
<td width="40%" valign="top"><a href="benchmarks/glm5_flash/2026-10-04-smoke-glm-5-3-flash-nvfp4-1rtx-2spark-glm53f-nvfp4-min/card.svg"><img src="benchmarks/glm5_flash/2026-10-04-smoke-glm-5-3-flash-nvfp4-1rtx-2spark-glm53f-nvfp4-min/card.svg" alt="nvidia/GLM-5.3-Flash-NVFP4 on 1× RTX PRO 6000 @ 325 W + 2× DGX Spark"></a><br><sub>1× RTX + 2× Spark</sub></td>
<td width="40%" valign="top"><a href="benchmarks/glm5_flash/2026-10-05-smoke-glm-5-3-flash-nvfp4-2rtx-4spark-rc2/card.svg"><img src="benchmarks/glm5_flash/2026-10-05-smoke-glm-5-3-flash-nvfp4-2rtx-4spark-rc2/card.svg" alt="nvidia/GLM-5.3-Flash-NVFP4 on 2× RTX PRO 6000 @ 325 W + 4× DGX Spark"></a><br><sub>2× RTX + 4× Spark</sub></td>
</tr>
<tr>
<td width="20%" valign="top"><a href="docs/models/glm5_flash.md"><b>GLM 5.3 Flash</b></a><br><sub>brandonmusic/GLM-5.3-Flash-tr3-4bpw</sub><br><sub>exl3-k4</sub></td>
<td width="40%" valign="top"><a href="benchmarks/glm5_flash/2026-10-04-smoke-glm-5-3-flash-tr3-4bpw-1rtx-2spark-glm53f-tr3-min/card.svg"><img src="benchmarks/glm5_flash/2026-10-04-smoke-glm-5-3-flash-tr3-4bpw-1rtx-2spark-glm53f-tr3-min/card.svg" alt="brandonmusic/GLM-5.3-Flash-tr3-4bpw on 1× RTX PRO 6000 @ 325 W + 2× DGX Spark"></a><br><sub>1× RTX + 2× Spark</sub></td>
<td width="40%" valign="top"><a href="benchmarks/glm5_flash/2026-10-04-smoke-glm-5-3-flash-tr3-4bpw-2rtx-4spark-glm53f-tr3-max/card.svg"><img src="benchmarks/glm5_flash/2026-10-04-smoke-glm-5-3-flash-tr3-4bpw-2rtx-4spark-glm53f-tr3-max/card.svg" alt="brandonmusic/GLM-5.3-Flash-tr3-4bpw on 2× RTX PRO 6000 @ 325 W + 4× DGX Spark"></a><br><sub>2× RTX + 4× Spark</sub></td>
</tr>
<tr>
<td width="20%" valign="top"><a href="docs/models/mimo_v2.md"><b>MiMo V2</b></a><br><sub>XiaomiMiMo/MiMo-V2.6-Flash-MOPD</sub><br><sub>mxfp4-g32</sub></td>
<td width="40%" valign="top"><a href="benchmarks/mimo_v2/2026-10-05-smoke-mimo-v2-6-flash-mopd-1rtx-2spark/card.svg"><img src="benchmarks/mimo_v2/2026-10-05-smoke-mimo-v2-6-flash-mopd-1rtx-2spark/card.svg" alt="XiaomiMiMo/MiMo-V2.6-Flash-MOPD on 1× RTX PRO 6000 @ 325 W + 2× DGX Spark"></a><br><sub>1× RTX + 2× Spark</sub></td>
<td width="40%" valign="top"><a href="benchmarks/mimo_v2/2026-10-05-smoke-mimo-v2-6-flash-mopd-2rtx-4spark/card.svg"><img src="benchmarks/mimo_v2/2026-10-05-smoke-mimo-v2-6-flash-mopd-2rtx-4spark/card.svg" alt="XiaomiMiMo/MiMo-V2.6-Flash-MOPD on 2× RTX PRO 6000 @ 325 W + 4× DGX Spark"></a><br><sub>2× RTX + 4× Spark</sub></td>
</tr>
<tr>
<td width="20%" valign="top"><a href="docs/models/mimo_v2.md"><b>MiMo V2</b></a><br><sub>XiaomiMiMo/MiMo-V2.6-Pro-MOPD</sub><br><sub>mxfp4-g32</sub></td>
<td width="40%" valign="top"><a href="benchmarks/mimo_v2/2026-10-04-smoke-mimo-v2-6-pro-mopd-1rtx-6spark-mimo-pro-min/card.svg"><img src="benchmarks/mimo_v2/2026-10-04-smoke-mimo-v2-6-pro-mopd-1rtx-6spark-mimo-pro-min/card.svg" alt="XiaomiMiMo/MiMo-V2.6-Pro-MOPD on 1× RTX PRO 6000 @ 325 W + 6× DGX Spark"></a><br><sub>1× RTX + 6× Spark</sub></td>
<td width="40%" valign="top"><a href="benchmarks/mimo_v2/2026-10-04-smoke-mimo-v2-6-pro-mopd-2rtx-6spark-mimo-pro-max/card.svg"><img src="benchmarks/mimo_v2/2026-10-04-smoke-mimo-v2-6-pro-mopd-2rtx-6spark-mimo-pro-max/card.svg" alt="XiaomiMiMo/MiMo-V2.6-Pro-MOPD on 2× RTX PRO 6000 @ 325 W + 6× DGX Spark"></a><br><sub>2× RTX + 6× Spark</sub></td>
</tr>
<tr>
<td width="20%" valign="top"><a href="docs/models/qwen4.md"><b>Qwen 3.8</b></a><br><sub>wrldsuksgo2mars/Qwen3.8-Flash-Next-EXL3-K4.25-PLE-FP8-v1</sub><br><sub>exl3-k4+exl3-k5</sub></td>
<td width="40%" valign="top"><a href="benchmarks/qwen4/2026-10-05-smoke-qwen38-exl3-32gib-1rtx-1spark-mtp3/card.svg"><img src="benchmarks/qwen4/2026-10-05-smoke-qwen38-exl3-32gib-1rtx-1spark-mtp3/card.svg" alt="wrldsuksgo2mars/Qwen3.8-Flash-Next-EXL3-K4.25-PLE-FP8-v1 on 1× RTX PRO 6000 @ 325 W + 1× DGX Spark"></a><br><sub>1× RTX (32 GiB budget) + 1× Spark</sub></td>
<td width="40%" valign="top"><a href="benchmarks/qwen4/2026-10-05-smoke-qwen38-exl3-1rtx-local-mtp3/card.svg"><img src="benchmarks/qwen4/2026-10-05-smoke-qwen38-exl3-1rtx-local-mtp3/card.svg" alt="wrldsuksgo2mars/Qwen3.8-Flash-Next-EXL3-K4.25-PLE-FP8-v1 on 1× RTX PRO 6000 @ 325 W"></a><br><sub>1× RTX</sub></td>
</tr>
<tr>
<td width="20%" valign="top"><a href="docs/models/qwen4.md"><b>Qwen 3.8</b></a><br><sub>nvidia/Qwen3.8-Flash-Next-NVFP4</sub><br><sub>nvfp4-g16</sub></td>
<td width="40%" valign="top"><a href="benchmarks/qwen4/2026-10-05-smoke-qwen3-8-flash-next-nvfp4-1rtx-rc2/card.svg"><img src="benchmarks/qwen4/2026-10-05-smoke-qwen3-8-flash-next-nvfp4-1rtx-rc2/card.svg" alt="nvidia/Qwen3.8-Flash-Next-NVFP4 on 1× RTX PRO 6000 @ 325 W"></a><br><sub>1× RTX</sub></td>
<td width="40%" valign="top"><a href="benchmarks/qwen4/2026-10-04-smoke-qwen3-8-flash-next-nvfp4-1rtx-4spark-qwen38-nvfp4-max/card.svg"><img src="benchmarks/qwen4/2026-10-04-smoke-qwen3-8-flash-next-nvfp4-1rtx-4spark-qwen38-nvfp4-max/card.svg" alt="nvidia/Qwen3.8-Flash-Next-NVFP4 on 1× RTX PRO 6000 @ 325 W + 4× DGX Spark"></a><br><sub>1× RTX + 4× Spark</sub></td>
</tr>
</table>

| Family | Checkpoint | Hardware | KV / req | C1 code | prose | JSON | 8K prefill | TTFT | Quality | Report |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| [DeepSeek V4.1](docs/models/deepseek_v41.md) | DeepSeek-V4.1-Flash (mxfp4-g32) | 1× RTX + 4× Spark (min) | 18.8M tok / 16 req | 129 | 77.7 | 147 | 4,809 | 1.69 s | ✓ KL 0.017 · top-1 90.8% ✓ exact cache ✓ lossless spec | [2026-10-04 · v1.0.0-rc1](benchmarks/deepseek_v41/2026-10-04-smoke-deepseek-v4-1-flash-1rtx-4spark-v41-flash-min/report.svg) |
| [DeepSeek V4.1](docs/models/deepseek_v41.md) | DeepSeek-V4.1-Flash (mxfp4-g32) | 2× RTX + 4× Spark (max) | 14.7M tok / 16 req | 169 | 89.0 | 184 | 7,411 | 1.09 s | ✓ KL 0.018 · top-1 90.8% ✓ exact cache | [2026-10-04 · v1.0.0-rc2](benchmarks/deepseek_v41/2026-10-05-smoke-deepseek-v4-1-flash-2rtx-4spark-rc2/report.svg) |
| [DeepSeek V4.1](docs/models/deepseek_v41.md) | DeepSeek-V4.1-Flash-NVFP4 (nvfp4-g16) | 1× RTX + 4× Spark (min) | 18.8M tok / 16 req | 113 | 69.2 | 128 | 4,308 | 1.89 s | ✓ KL 0.034 · top-1 88.3% ✓ exact cache | [2026-10-04 · v1.0.0-rc1](benchmarks/deepseek_v41/2026-10-04-smoke-deepseek-v4-1-flash-nvfp4-1rtx-4spark-v41-flash-nvfp4-min/report.svg) |
| [DeepSeek V4.1](docs/models/deepseek_v41.md) | DeepSeek-V4.1-Flash-NVFP4 (nvfp4-g16) | 2× RTX + 4× Spark (max) | 14.7M tok / 16 req | 117 | 72.4 | 147 | 5,368 | 1.51 s | ✓ KL 0.037 · top-1 87.5% ✓ exact cache | [2026-10-04 · v1.0.0-rc1](benchmarks/deepseek_v41/2026-10-04-smoke-deepseek-v4-1-flash-nvfp4-2rtx-4spark-v41-flash-nvfp4-max/report.svg) |
| [DeepSeek V4](docs/models/deepseek_v4.md) | DeepSeek-V4-Flash-0731 (mxfp4-g32) | 1× RTX + 2× Spark (min) | 258K tok / 8 req | 151 | 69.9 | 157 | 4,275 | 1.90 s | ✓ KL 0.012 · top-1 92.0% ✓ exact cache ✓ lossless spec | [2026-10-04 · v1.0.0-rc1](benchmarks/deepseek_v4/2026-10-04-smoke-deepseek-v4-flash-0731-1rtx-2spark-v4-flash-min/report.svg) |
| [DeepSeek V4](docs/models/deepseek_v4.md) | DeepSeek-V4-Flash-0731 (mxfp4-g32) | 2× RTX + 4× Spark (max) | 258K tok / 8 req | 222 | 93.5 | 213 | 5,319 | 1.52 s | ✓ KL 0.014 · top-1 92.2% ✓ exact cache ✓ lossless spec | [2026-10-04 · v1.0.0-rc1](benchmarks/deepseek_v4/2026-10-04-smoke-deepseek-v4-flash-0731-2rtx-4spark-v4-flash-max/report.svg) |
| [DeepSeek V4](docs/models/deepseek_v4.md) | DeepSeek-V4-Pro-0813-EXL3-K2-calibrated-v1 (exl3-k2) | 1× RTX + 4× Spark (min) | 258K tok / 8 req | 69.9 | 33.3 | 80.7 | 2,134 | 3.79 s | ✓ ✓ exact cache | [2026-10-04 · v1.0.0-rc1](benchmarks/deepseek_v4/2026-10-04-smoke-deepseek-v4-pro-0813-exl3-k2-calibrated-v1-1rtx-4spark-v4-pro-exl3-min/report.svg) |
| [DeepSeek V4](docs/models/deepseek_v4.md) | DeepSeek-V4-Pro-0813-EXL3-K2-calibrated-v1 (exl3-k2) | 2× RTX + 6× Spark (max) | 258K tok / 8 req | 94.1 | 43.0 | 101 | 2,472 | 3.28 s | ✓ ✓ exact cache | [2026-10-04 · v1.0.0-rc1](benchmarks/deepseek_v4/2026-10-04-smoke-deepseek-v4-pro-0813-exl3-k2-calibrated-v1-2rtx-6spark-v4-pro-exl3-max/report.svg) |
| [GLM 5.3](docs/models/glm5.md) | GLM-5.3-EXL3-K4-v1 (exl3-k4) | 1× RTX + 4× Spark (min) | 1.1M tok / 8 req | 44.2 | 33.9 | 55.0 | 2,707 | 2.98 s | ✓ KL 0.021 · top-1 90.8% ✓ exact cache | [2026-10-04 · v1.0.0-rc1](benchmarks/glm5/2026-10-04-smoke-glm-5-3-exl3-k4-v1-1rtx-4spark-glm53-exl3-min/report.svg) |
| [GLM 5.3](docs/models/glm5.md) | GLM-5.3-EXL3-K4-v1 (exl3-k4) | 2× RTX + 6× Spark (max) | 1.3M tok / 8 req | 55.4 | 39.3 | 67.8 | 3,122 | 2.59 s | ✓ KL 0.022 · top-1 90.2% ✓ exact cache | [2026-10-04 · v1.0.0-rc1](benchmarks/glm5/2026-10-04-smoke-glm-5-3-exl3-k4-v1-2rtx-6spark-glm53-exl3-max/report.svg) |
| [GLM 5.3](docs/models/glm5.md) | GLM-5.3-NVFP4 (nvfp4-g16) | 1× RTX + 6× Spark (min) | 1.1M tok / 8 req | 55.0 | 42.9 | 71.6 | 2,958 | 2.73 s | ✓ KL 0.035 · top-1 88.1% ✓ exact cache | [2026-10-04 · v1.0.0-rc1](benchmarks/glm5/2026-10-04-smoke-glm-5-3-nvfp4-1rtx-6spark-glm53-nvfp4-min/report.svg) |
| [GLM 5.3](docs/models/glm5.md) | GLM-5.3-NVFP4 (nvfp4-g16) | 2× RTX + 6× Spark (max) | 1.3M tok / 8 req | 61.9 | 39.9 | 73.8 | 3,014 | 2.67 s | ✓ KL 0.032 · top-1 87.9% ✓ exact cache | [2026-10-04 · v1.0.0-rc1](benchmarks/glm5/2026-10-04-smoke-glm-5-3-nvfp4-2rtx-6spark-glm53-nvfp4-max/report.svg) |
| [GLM 5.3 Flash](docs/models/glm5_flash.md) | GLM-5.3-Flash (fp8-block128x128/f32) | 1× RTX + 4× Spark (min) | 2M tok / 8 req | 75.4 | 61.4 | 97.1 | 2,509 | 3.21 s | ✓ KL 0.018 · top-1 92.0% ✓ exact cache | [2026-10-04 · v1.0.0-rc1](benchmarks/glm5_flash/2026-10-04-smoke-glm-5-3-flash-1rtx-4spark-glm53f-fp8-min/report.svg) |
| [GLM 5.3 Flash](docs/models/glm5_flash.md) | GLM-5.3-Flash (fp8-block128x128/f32) | 2× RTX + 4× Spark (max) | 2M tok / 8 req | 76.7 | 61.3 | 85.7 | 2,661 | 3.03 s | ✓ KL 0.018 · top-1 90.6% ✓ exact cache ✓ lossless spec | [2026-10-04 · v1.0.0-rc1](benchmarks/glm5_flash/2026-10-04-smoke-glm-5-3-flash-2rtx-4spark-glm53f-fp8-max/report.svg) |
| [GLM 5.3 Flash](docs/models/glm5_flash.md) | GLM-5.3-Flash-EXL3-K3.25-v1 (exl3-k3+exl3-k4) | 1× RTX + 2× Spark (min) | 2M tok / 8 req | 83.1 | 65.5 | 110 | 4,898 | 1.65 s | ✓ KL 0.042 · top-1 86.5% ✓ exact cache | [2026-10-04 · v1.0.0-rc1](benchmarks/glm5_flash/2026-10-04-smoke-glm-5-3-flash-exl3-k3-25-v1-1rtx-2spark-glm53f-exl3-min/report.svg) |
| [GLM 5.3 Flash](docs/models/glm5_flash.md) | GLM-5.3-Flash-EXL3-K3.25-v1 (exl3-k3+exl3-k4) | 2× RTX + 4× Spark (max) | 2M tok / 8 req | 120 | 92.3 | 159 | 2,946 | 2.74 s | ✓ KL 0.045 · top-1 88.5% ✓ exact cache | [2026-10-04 · v1.0.0-rc1](benchmarks/glm5_flash/2026-10-04-smoke-glm-5-3-flash-exl3-k3-25-v1-2rtx-4spark-glm53f-exl3-max/report.svg) |
| [GLM 5.3 Flash](docs/models/glm5_flash.md) | GLM-5.3-Flash-NVFP4 (nvfp4-g16) | 1× RTX + 2× Spark (min) | 2M tok / 8 req | 67.4 | 57.4 | 76.9 | 5,060 | 1.59 s | ✓ KL 0.040 · top-1 87.3% ✓ exact cache | [2026-10-04 · v1.0.0-rc1](benchmarks/glm5_flash/2026-10-04-smoke-glm-5-3-flash-nvfp4-1rtx-2spark-glm53f-nvfp4-min/report.svg) |
| [GLM 5.3 Flash](docs/models/glm5_flash.md) | GLM-5.3-Flash-NVFP4 (nvfp4-g16) | 2× RTX + 4× Spark (max) | 2M tok / 8 req | 115 | 85.1 | 129 | 7,239 | 1.11 s | ✓ KL 0.039 · top-1 87.3% ✓ exact cache ✓ lossless spec | [2026-10-04 · v1.0.0-rc2](benchmarks/glm5_flash/2026-10-05-smoke-glm-5-3-flash-nvfp4-2rtx-4spark-rc2/report.svg) |
| [GLM 5.3 Flash](docs/models/glm5_flash.md) | GLM-5.3-Flash-tr3-4bpw (exl3-k4) | 1× RTX + 2× Spark (min) | 2M tok / 8 req | 71.8 | 57.4 | 83.8 | 4,615 | 1.74 s | ✓ KL 0.037 · top-1 87.5% ✓ exact cache | [2026-10-04 · v1.0.0-rc1](benchmarks/glm5_flash/2026-10-04-smoke-glm-5-3-flash-tr3-4bpw-1rtx-2spark-glm53f-tr3-min/report.svg) |
| [GLM 5.3 Flash](docs/models/glm5_flash.md) | GLM-5.3-Flash-tr3-4bpw (exl3-k4) | 2× RTX + 4× Spark (max) | 2M tok / 8 req | 111 | 90.9 | 126 | 5,159 | 1.56 s | ✓ KL 0.036 · top-1 87.3% ✓ exact cache ✓ lossless spec | [2026-10-04 · v1.0.0-rc1](benchmarks/glm5_flash/2026-10-04-smoke-glm-5-3-flash-tr3-4bpw-2rtx-4spark-glm53f-tr3-max/report.svg) |
| [MiMo V2](docs/models/mimo_v2.md) | MiMo-V2.6-Flash-MOPD (mxfp4-g32) | 1× RTX + 2× Spark (min) | 32K tok / 4 req | 89.7 | 49.1 | 82.2 | 5,325 | 1.52 s | ✓ KL 0.022 · top-1 89.8% ✓ exact cache ✓ lossless spec | [2026-10-05 · 03f3ddb17ce0](benchmarks/mimo_v2/2026-10-05-smoke-mimo-v2-6-flash-mopd-1rtx-2spark/report.svg) |
| [MiMo V2](docs/models/mimo_v2.md) | MiMo-V2.6-Flash-MOPD (mxfp4-g32) | 2× RTX + 4× Spark (max) | 32K tok / 4 req | 122 | 74.1 | 130 | 7,395 | 1.09 s | ✓ KL 0.026 · top-1 88.7% ✓ exact cache ✓ lossless spec | [2026-10-05 · 03f3ddb17ce0](benchmarks/mimo_v2/2026-10-05-smoke-mimo-v2-6-flash-mopd-2rtx-4spark/report.svg) |
| [MiMo V2](docs/models/mimo_v2.md) | MiMo-V2.6-Pro-MOPD (mxfp4-g32) | 1× RTX + 6× Spark (min) | 2M tok / 8 req | 50.0 | 37.1 | 76.7 | 2,631 | 3.06 s | ✓ KL 0.020 · top-1 90.2% ✓ exact cache ✓ lossless spec | [2026-10-04 · v1.0.0-rc1](benchmarks/mimo_v2/2026-10-04-smoke-mimo-v2-6-pro-mopd-1rtx-6spark-mimo-pro-min/report.svg) |
| [MiMo V2](docs/models/mimo_v2.md) | MiMo-V2.6-Pro-MOPD (mxfp4-g32) | 2× RTX + 6× Spark (max) | 2M tok / 8 req | 71.7 | 41.0 | 86.9 | 3,470 | 2.32 s | ✓ KL 0.025 · top-1 88.3% ✓ exact cache ✓ lossless spec | [2026-10-04 · v1.0.0-rc1](benchmarks/mimo_v2/2026-10-04-smoke-mimo-v2-6-pro-mopd-2rtx-6spark-mimo-pro-max/report.svg) |
| [Qwen 3.8](docs/models/qwen4.md) | Qwen3.8-Flash-Next-EXL3-K4.25-PLE-FP8-v1 (exl3-k4+exl3-k5) | 1× RTX (32 GiB budget) + 1× Spark (min) | 32K tok / 4 req | 121 | 82.2 | 132 | 3,698 | 2.16 s | ✓ KL 0.036 · top-1 88.5% ✓ exact cache ✓ lossless spec | [2026-10-05 · cdf793385d86](benchmarks/qwen4/2026-10-05-smoke-qwen38-exl3-32gib-1rtx-1spark-mtp3/report.svg) |
| [Qwen 3.8](docs/models/qwen4.md) | Qwen3.8-Flash-Next-EXL3-K4.25-PLE-FP8-v1 (exl3-k4+exl3-k5) | 1× RTX (max) | 32K tok / 4 req | 222 | 153 | 263 | 6,704 | 1.19 s | ✓ KL 0.036 · top-1 88.5% ✓ exact cache ✓ lossless spec | [2026-10-05 · cdf793385d86](benchmarks/qwen4/2026-10-05-smoke-qwen38-exl3-1rtx-local-mtp3/report.svg) |
| [Qwen 3.8](docs/models/qwen4.md) | Qwen3.8-Flash-Next-NVFP4 (nvfp4-g16) | 1× RTX (min) | 32K tok / 8 req | 231 | 159 | 267 | 9,108 | 876 ms | ✓ KL 0.048 · top-1 85.9% ✓ exact cache ✓ lossless spec | [2026-10-04 · v1.0.0-rc2](benchmarks/qwen4/2026-10-05-smoke-qwen3-8-flash-next-nvfp4-1rtx-rc2/report.svg) |
| [Qwen 3.8](docs/models/qwen4.md) | Qwen3.8-Flash-Next-NVFP4 (nvfp4-g16) | 1× RTX + 4× Spark (max) | 32K tok / 8 req | 78.3 | 76.6 | 85.2 | 3,655 | 2.18 s | ✓ KL 0.053 · top-1 84.6% ✓ exact cache ✓ lossless spec | [2026-10-04 · v1.0.0-rc1](benchmarks/qwen4/2026-10-04-smoke-qwen3-8-flash-next-nvfp4-1rtx-4spark-qwen38-nvfp4-max/report.svg) |

tok/s; C1 decode with thinking off, 8K prefill cold. Quality: logit fidelity against the family golden reference, prefix-cache restore exactness, lossless speculation.

<!-- results:end -->

Each family's page ([`docs/models/`](docs/models/)) has its supported
checkpoints and quants, engineering summary and known limits:
[DeepSeek V4.1 Flash](docs/models/deepseek_v41.md),
[DeepSeek V4 Flash/Pro](docs/models/deepseek_v4.md),
[GLM 5.3](docs/models/glm5.md), [GLM 5.3 Flash](docs/models/glm5_flash.md),
[MiMo V2 Flash / V2.6 Pro](docs/models/mimo_v2.md),
[Qwen 3.8 Flash Next](docs/models/qwen4.md).

## Storage

The checkpoints and quants behind these numbers were served from
[SparkNest](https://github.com/tpurtell/sparknest), a distributed model store
across the cluster: hosts that already hold a sealed local copy of a shard
read it at NVMe speed, hosts without one stream it over RoCE at roughly
5 GB/s. SparkNest is what made the quoted load times possible; it is a
separate project, not required to run CuteAFD, and any standard Hugging Face
cache layout works.

## Quick start

1. `cuteafd plan MODEL` (any Hugging Face model id or local snapshot
   directory) reports what the checkpoint needs — tensors, formats, shapes,
   and which kernels are missing — before you touch a GPU. Add `--layout
   --rtx 1|2 --pool-tokens 0` for per-device weights, cache admission,
   workspaces and Spark ranks. `--coordinator-gpu-budget-gib 31.8` sets the
   logical per-GPU ceiling for plan, serve and golden (plan defaults to 95.5
   GiB). Plan's separate `--coordinator-weight-budget-gib` caps only weights
   (default 80 GiB). The old `--rtx-budget-gib`, `--rtx-gib` and
   `--coordinator-budget-gib` spellings remain deprecated aliases.
   V4 Flash/Pro workspace formulas use the
   matching image manifest (`--workspace-manifest PROGRAMS.json`). The
   image also supplies EXL3 allocation manifests; when exporting metadata,
   keep their `exl3/` tree alongside `PROGRAMS.json`.
   V4 Flash native TP2, V4.1 native TP4 and Qwen local EXL3 K4.25 layouts are
   calibrated on one RTX PRO 6000 after 8K prefill and C4; unmeasured layouts
   stay estimates. Offline auto can differ slightly from runtime admission,
   which samples concrete CUDA owners before allocating the pool.
2. Pick or adapt a config under [`examples/configs/`](examples/configs/) or
   edit `cuteafd.config` for your own topology (coordinator GPUs, Spark
   ranks, TP/EP layout). `POOL_TOKENS=auto` selects planner admission for
   V4, V4.1 and Qwen; the engine spelling is `--pool-tokens 0`. V4.1 keeps
   its existing pool policy when this option is omitted.
3. `./run.sh` launches the release images named in the config,
   `ghcr.io/tpurtell/cuteafd-coordinator:v1.0.0` on the RTX host and
   `ghcr.io/tpurtell/cuteafd-spark-expert:v1.0.0` on each Spark; `docker pull`
   them on those hosts first (`./run.sh` does not pull). `./wip.sh --slot S --role both`
   plus `./run.sh --wip S --restart` is the faster loop while iterating.

V4.1 defaults to `CUTEAFD_V41_FP8_HEAD=all`: one E4M3 vocabulary head
(or one shard per GPU) shared by target and dSpark, with BF16 released after
packing. `off` selects BF16; experimental `draft` retains BF16 for the target
and adds FP8 for dSpark. The accepted target-head quality gate and matched RC2
controls support the release default.

## Client APIs: Claude Code, Codex and Realtime

Besides OpenAI Chat Completions (`/v1/chat/completions`), cuteafd serves the
APIs that the Claude Code and Codex CLIs and Realtime voice clients speak, so
they can run against your own model with no proxy in between.

- **Anthropic Messages:**
  - `POST /v1/messages`, streaming and non-streaming, with tools, thinking,
    images and stop reasons;
  - `POST /v1/messages/count_tokens`;
  - the Anthropic model listing.
- **OpenAI Responses:**
  - `POST /v1/responses`, over SSE or WebSocket;
  - `GET`/`DELETE /v1/responses/{id}` and `input_items`;
  - `previous_response_id`;
  - function, custom (freeform) and `local_shell` tools;
  - `/v1/responses/compact` and `input_tokens`;
  - a Codex model catalog at `/v1/codex/models.json`.
- **OpenAI Realtime:** a `GET /v1/realtime` WebSocket, with GA and beta event
  names.
  - Text and function calling work. Audio input works on audio-capable models.
  - Speech output and transcription are not available yet. Requests for them
    get an explicit error.

The Messages, Responses and Realtime routes are built and tested against an
upstream test backend today. Wiring them to the engine's own serve path is
the next step (PLAN.md, "v3 API gateway and sessions").

**Keys.** Start the server with `--api-key-file FILE`. Clients send that key
the way they would to the real service:
- `x-api-key` or `Authorization: Bearer` for Messages;
- Bearer for Responses;
- Bearer or the `openai-insecure-api-key.<key>` WebSocket subprotocol for
  Realtime.

**Model names.** `--official-model-names` accepts any requested model id and
runs the served model. It also advertises the ids the CLIs look for: Claude
Code's model discovery lists only `claude-*` ids, and Codex has a fixed set of
slugs. To refresh those lists without a rebuild, generate a file with
`scripts/gateway/official-model-names.py` and pass it as
`--official-model-names-file`.

**Web search.** Claude Code's WebSearch and Codex's web search run on the
server: `--search exa` (needs `EXA_API_KEY`) or `--search searxng=URL` (no
key; `scripts/gateway/searxng/` runs a local SearXNG).

**Claude Code:**

```sh
export ANTHROPIC_BASE_URL=http://HOST:PORT
export ANTHROPIC_API_KEY=$(cat FILE)
export CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1   # optional: /model picker
claude
```

All of its model slots (`ANTHROPIC_MODEL`,
`ANTHROPIC_DEFAULT_{OPUS,SONNET,HAIKU}_MODEL`, `CLAUDE_CODE_SUBAGENT_MODEL`)
map to the served model.

**Codex CLI,** in `~/.codex/config.toml`:

```toml
model = "gpt-6.1-sol"          # a slug Codex knows, so it keeps its full tool set
model_provider = "cuteafd"
web_search = "live"            # optional: server-side web search

[model_providers.cuteafd]
name = "cuteafd"
base_url = "http://HOST:PORT/v1"
wire_api = "responses"
env_key = "CUTEAFD_API_KEY"    # export CUTEAFD_API_KEY=$(cat FILE)
# Optional: the served model's real context window and output limit, so
# Codex compacts at the right point instead of using its built-in numbers.
model_catalog_url = "http://HOST:PORT/v1/codex/models.json"
```

**Realtime:** any client that accepts a custom URL can connect to
`ws://HOST:PORT/v1/realtime?model=...`. These run headless against it:
- the openai-python and openai-node SDKs (Node requires `wss://`);
- Agents JS/Python;
- Pipecat.

Runners are in `scripts/gateway/realtime-clients/`.

## Working on it

[`AGENTS.md`](AGENTS.md) is the standing guide for agents and collaborators
working on CuteAFD. [`PLAN.md`](PLAN.md) is the roadmap.

## Building from source

Initialize the pinned submodules with `git submodule update --init --recursive`.
The shared toolchain image is published as `ghcr.io/tpurtell/cuteafd-dev` for
amd64 and arm64: choose an immutable `tc-<hash>` tag (or digest) for repeatable
builds; `latest` follows dev toolchain updates. Set `COORDINATOR_DOCKER_DEV` and
`SPARK_EXPERT_DOCKER_DEV` in your config to that reference. WIP and release builds
pull registry references when absent; local image names remain the default.

The image contains no SparkInfer: consumers verify the checkout's tree lock
and import its mounted submodule. A kernel pin change does not require rebuilding
the toolchain. Rebuild with `scripts/build/build-dev-images.sh --dry-run` first,
then without `--dry-run`; `--publish --spark-hosts rhea` publishes only the dev
package and preserves existing local campaign tags. Serialize actual builds
with `~/.cache/cuteafd/build.lock` and the Spark build host's lock.

Compiler caches are off by default. `CUTEAFD_KACHE=1` uses the image's kache for
C/C++ and Rust; `CUTEAFD_KACHE_SPARK=1` enables it on Spark builds.
`CUTEAFD_SCCACHE_CUDA=1` independently selects sccache for CUDA. Cache directories
are host NVMe bind mounts under `~/.cache/cuteafd/builds/compiler-cache`, not
image layers. Read-only source/JIT smoke: run
`scripts/build/smoke-b12x-readonly.py --source /source --jit` inside a matching
architecture's dev container with `/source:ro`, writable `HOME` and cache roots,
and NVIDIA driver libraries (no GPU device is needed).
