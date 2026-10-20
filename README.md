<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/brand/cuteafd-logo-color-dark.svg">
    <img src="assets/brand/cuteafd-logo-color.svg" alt="cuteafd" width="480">
  </picture>
</p>

CuteAFD brings frontier-scale open-weights models into the home lab at
data-center speed: one or two RTX PRO 6000 cards run attention while a pool of
DGX Sparks serves the routed experts over RoCE.

- **Attention/FFN disaggregation:** RTX cards run attention and the backbone,
  DGX Sparks hold the routed experts, and activations cross RoCE straight into
  GPU memory.
- **Checkpoints as shipped:** official FP8 and MXFP4, NVIDIA NVFP4 and EXL3
  load from the Hugging Face snapshot itself, with no repacking or side files.
- **Exact prefix cache:** agentic sessions restore the deepest cached prefix
  byte-for-byte instead of recomputing it.
- **Built-in dashboard:** a live console of lanes, pipeline stages and draft
  acceptance, plus a benchmark page with speed, quality and agentic panels.
  [Dashboard](docs/dashboard.md)
- **Images and audio:** bundled vision and audio towers run on a Spark or an
  RTX wherever the planner finds room. [Design](docs/multimodal-design.md)
- **Plan before you serve:** `cuteafd plan` names what a checkpoint needs and
  how it fits your cards, and can copy each host only the shards it reads.
  [Sliced checkpoints](docs/sliced-checkpoints.md)
- **Claude Code and Codex on your model** (next release): Anthropic Messages,
  OpenAI Responses and Realtime APIs with server-side web search, so the CLIs
  connect directly. [Client APIs](docs/client-apis.md)
- **Own your intelligence:** your weights, your hardware, no rate limits.

## What's new in v3.0.0

- **Two-GPU placement for every family:** experts run TP2 across both RTX
  cards, and attention is placed without copying context-scaling KV to both
  GPUs (split by context or owned by layer, per family).
- **Planned memory:** one placement planner sizes every model's weights,
  KV pool and workspaces exactly before anything loads, on the coordinator
  and on each Spark.
- **One draft policy for every family:** a simple shared policy picks which
  draft tokens to verify, with dSpark/DFlash2/MTP drafters placed by the
  planner.
- **DeepSeek V4.1 Flash 1017** is the default V4.1 checkpoint, loaded and
  identified from its own files.
- **Dense Qwen 3.5 / 3.8** serve natively on RTX cards.
- **Dashboard:** read-only console open to everyone, with an admin link for
  live text, benchmarks and usage; GPU temperature, power and clock for every
  RTX and Spark; fabric health as retransmits per packet.
- **Fidelity datasets** score each quant against text its own model wrote,
  generated from the official API and checked against the original model.

Full notes: [v3.0.0 release](https://github.com/tpurtell/cuteafd/releases/tag/v3.0.0) ·
[all releases](https://github.com/tpurtell/cuteafd/releases)

## Models

One basic-profile card per model and quant on this release: the natural minimum
(1× RTX + the fewest Sparks) or maximum (2× RTX + 4 or 6 Sparks). In the 5090
column, cards labelled "simulated 5090" ran on an RTX PRO 6000 capped at
31.8 GiB (memory only, not SM count or clocks) until real RTX 5090 cards replace
them; the others ran on a GeForce RTX 5090. Earlier releases and every other
report: [benchmarks](benchmarks/README.md).

<!-- results:begin -->
<table>
<tr><th>Model · quant</th><th>5090</th><th>1× RTX</th><th>2× RTX</th></tr>
<tr>
<td width="19%" valign="top"><a href="docs/models/deepseek_v41.md"><b>DeepSeek V4.1</b></a><br><sub>deepseek-ai/DeepSeek-V4.1-Flash-1017</sub><br><sub>mxfp4-g32</sub></td>
<td width="27%" valign="top"><a href="benchmarks/deepseek_v41/2026-10-18-smoke-deepseek-v4-1-flash-1017-1rtx-3spark-rtx5090-v41-flash-5090/card.svg"><img src="benchmarks/deepseek_v41/2026-10-18-smoke-deepseek-v4-1-flash-1017-1rtx-3spark-rtx5090-v41-flash-5090/card.svg" alt="deepseek-ai/DeepSeek-V4.1-Flash-1017 on 1× GeForce RTX 5090 @ 600 W + 3× DGX Spark · 31.8 GiB budget"></a><br><sub>RTX 5090 (31.8 GiB budget) + 3× Spark</sub></td>
<td width="27%" valign="top"><a href="benchmarks/deepseek_v41/2026-10-18-smoke-deepseek-v4-1-flash-1017-1rtx-3spark-v41-flash-min/card.svg"><img src="benchmarks/deepseek_v41/2026-10-18-smoke-deepseek-v4-1-flash-1017-1rtx-3spark-v41-flash-min/card.svg" alt="deepseek-ai/DeepSeek-V4.1-Flash-1017 on 1× RTX PRO 6000 @ gpu-powerd 550 W (500–600) + 3× DGX Spark"></a><br><sub>1× RTX + 3× Spark</sub></td>
<td width="27%" valign="top"><a href="benchmarks/deepseek_v41/2026-10-18-smoke-deepseek-v4-1-flash-1017-2rtx-4spark-v41-flash-max/card.svg"><img src="benchmarks/deepseek_v41/2026-10-18-smoke-deepseek-v4-1-flash-1017-2rtx-4spark-v41-flash-max/card.svg" alt="deepseek-ai/DeepSeek-V4.1-Flash-1017 on 2× RTX PRO 6000 @ gpu-powerd 450/450 W (300–550) + 4× DGX Spark"></a><br><sub>2× RTX + 4× Spark</sub></td>
</tr>
<tr>
<td width="19%" valign="top"><a href="docs/models/deepseek_v4.md"><b>DeepSeek V4</b></a><br><sub>deepseek-ai/DeepSeek-V4-Flash-0731</sub><br><sub>mxfp4-g32</sub></td>
<td width="27%" valign="top"><a href="benchmarks/deepseek_v4/2026-10-18-smoke-deepseek-v4-flash-0731-1rtx-2spark-v4-flash-5090/card.svg"><img src="benchmarks/deepseek_v4/2026-10-18-smoke-deepseek-v4-flash-0731-1rtx-2spark-v4-flash-5090/card.svg" alt="deepseek-ai/DeepSeek-V4-Flash-0731 on simulated 5090: RTX PRO 6000 (188 SMs) capped at 31.8 GiB + 2 Spark"></a><br><sub>simulated 5090: RTX PRO 6000 (188 SMs) capped at 31.8 GiB + 2 Spark</sub></td>
<td width="27%" valign="top"><a href="benchmarks/deepseek_v4/2026-10-18-smoke-deepseek-v4-flash-0731-1rtx-2spark-v4-flash-min/card.svg"><img src="benchmarks/deepseek_v4/2026-10-18-smoke-deepseek-v4-flash-0731-1rtx-2spark-v4-flash-min/card.svg" alt="deepseek-ai/DeepSeek-V4-Flash-0731 on 1× RTX PRO 6000 @ gpu-powerd 460/435 W (395–500) + 2× DGX Spark"></a><br><sub>1× RTX + 2× Spark</sub></td>
<td width="27%" valign="top">n/a<br><sub>Not measured for v3 (one card per checkpoint).</sub></td>
</tr>
<tr>
<td width="19%" valign="top"><a href="docs/models/deepseek_v4.md"><b>DeepSeek V4</b></a><br><sub>wrldsuksgo2mars/DeepSeek-V4-Pro-0813-EXL3-K2-calibrated-v1</sub><br><sub>exl3-k2</sub></td>
<td width="27%" valign="top">n/a<br><sub>Doesn&#39;t fit one 32 GB card with this checkpoint.</sub></td>
<td width="27%" valign="top">n/a<br><sub>Not measured for v3 (one card per checkpoint).</sub></td>
<td width="27%" valign="top"><a href="benchmarks/deepseek_v4/2026-10-18-smoke-deepseek-v4-pro-0813-exl3-k2-calibrated-v1-2rtx-6spark-v4-pro-exl3-max/card.svg"><img src="benchmarks/deepseek_v4/2026-10-18-smoke-deepseek-v4-pro-0813-exl3-k2-calibrated-v1-2rtx-6spark-v4-pro-exl3-max/card.svg" alt="wrldsuksgo2mars/DeepSeek-V4-Pro-0813-EXL3-K2-calibrated-v1 on 2× RTX PRO 6000 @ gpu-powerd 450/445 W (300–550) + 6× DGX Spark"></a><br><sub>2× RTX + 6× Spark</sub></td>
</tr>
<tr>
<td width="19%" valign="top"><a href="docs/models/glm5.md"><b>GLM 5.3</b></a><br><sub>wrldsuksgo2mars/GLM-5.3-EXL3-K4-v1</sub><br><sub>exl3-k4</sub></td>
<td width="27%" valign="top">n/a<br><sub>Doesn&#39;t fit one 32 GB card with this checkpoint.</sub></td>
<td width="27%" valign="top"><a href="benchmarks/glm5/2026-10-18-smoke-glm-5-3-exl3-k4-v1-1rtx-4spark-glm53-exl3-min/card.svg"><img src="benchmarks/glm5/2026-10-18-smoke-glm-5-3-exl3-k4-v1-1rtx-4spark-glm53-exl3-min/card.svg" alt="wrldsuksgo2mars/GLM-5.3-EXL3-K4-v1 on 1× RTX PRO 6000 @ gpu-powerd 335/510 W (265–600) + 4× DGX Spark"></a><br><sub>1× RTX + 4× Spark</sub></td>
<td width="27%" valign="top">n/a<br><sub>Not measured for v3 (one card per checkpoint).</sub></td>
</tr>
<tr>
<td width="19%" valign="top"><a href="docs/models/glm5.md"><b>GLM 5.3</b></a><br><sub>nvidia/GLM-5.3-NVFP4</sub><br><sub>nvfp4-g16</sub></td>
<td width="27%" valign="top">n/a<br><sub>Doesn&#39;t fit one 32 GB card with this checkpoint.</sub></td>
<td width="27%" valign="top"><a href="benchmarks/glm5/2026-10-18-smoke-glm-5-3-nvfp4-1rtx-4spark-glm53-nvfp4-min/card.svg"><img src="benchmarks/glm5/2026-10-18-smoke-glm-5-3-nvfp4-1rtx-4spark-glm53-nvfp4-min/card.svg" alt="nvidia/GLM-5.3-NVFP4 on 1× RTX PRO 6000 @ gpu-powerd 480/405 W (200–600) + 4× DGX Spark"></a><br><sub>1× RTX + 4× Spark</sub></td>
<td width="27%" valign="top">n/a<br><sub>Not measured for v3 (one card per checkpoint).</sub></td>
</tr>
<tr>
<td width="19%" valign="top"><a href="docs/models/glm5_flash.md"><b>GLM 5.3 Flash</b></a><br><sub>wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3.25-v1</sub><br><sub>exl3-k3+exl3-k4</sub></td>
<td width="27%" valign="top"><a href="benchmarks/glm5_flash/2026-10-18-smoke-glm-5-3-flash-exl3-k3-25-v1-1rtx-2spark-glm53f-exl3-5090/card.svg"><img src="benchmarks/glm5_flash/2026-10-18-smoke-glm-5-3-flash-exl3-k3-25-v1-1rtx-2spark-glm53f-exl3-5090/card.svg" alt="wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3.25-v1 on simulated 5090: RTX PRO 6000 (188 SMs) capped at 31.8 GiB + 2 Spark"></a><br><sub>simulated 5090: RTX PRO 6000 (188 SMs) capped at 31.8 GiB + 2 Spark</sub></td>
<td width="27%" valign="top"><a href="benchmarks/glm5_flash/2026-10-18-smoke-glm-5-3-flash-exl3-k3-25-v1-1rtx-2spark-glm53f-exl3-min/card.svg"><img src="benchmarks/glm5_flash/2026-10-18-smoke-glm-5-3-flash-exl3-k3-25-v1-1rtx-2spark-glm53f-exl3-min/card.svg" alt="wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3.25-v1 on 1× RTX PRO 6000 @ gpu-powerd 415/465 W (300–600) + 2× DGX Spark"></a><br><sub>1× RTX + 2× Spark</sub></td>
<td width="27%" valign="top"><a href="benchmarks/glm5_flash/2026-10-18-smoke-glm-5-3-flash-exl3-k3-25-v1-2rtx-glm53f-exl3-max/card.svg"><img src="benchmarks/glm5_flash/2026-10-18-smoke-glm-5-3-flash-exl3-k3-25-v1-2rtx-glm53f-exl3-max/card.svg" alt="wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3.25-v1 on 2× RTX PRO 6000 @ gpu-powerd 455/440 W (300–570)"></a><br><sub>2× RTX</sub></td>
</tr>
<tr>
<td width="19%" valign="top"><a href="docs/models/glm5_flash.md"><b>GLM 5.3 Flash</b></a><br><sub>nvidia/GLM-5.3-Flash-NVFP4</sub><br><sub>nvfp4-g16</sub></td>
<td width="27%" valign="top"><a href="benchmarks/glm5_flash/2026-10-18-smoke-glm-5-3-flash-nvfp4-1rtx-2spark-glm53f-nvfp4-5090/card.svg"><img src="benchmarks/glm5_flash/2026-10-18-smoke-glm-5-3-flash-nvfp4-1rtx-2spark-glm53f-nvfp4-5090/card.svg" alt="nvidia/GLM-5.3-Flash-NVFP4 on simulated 5090: RTX PRO 6000 (188 SMs) capped at 31.8 GiB + 2 Spark"></a><br><sub>simulated 5090: RTX PRO 6000 (188 SMs) capped at 31.8 GiB + 2 Spark</sub></td>
<td width="27%" valign="top"><a href="benchmarks/glm5_flash/2026-10-18-smoke-glm-5-3-flash-nvfp4-1rtx-2spark-glm53f-nvfp4-min/card.svg"><img src="benchmarks/glm5_flash/2026-10-18-smoke-glm-5-3-flash-nvfp4-1rtx-2spark-glm53f-nvfp4-min/card.svg" alt="nvidia/GLM-5.3-Flash-NVFP4 on 1× RTX PRO 6000 @ gpu-powerd 355/535 W (255–600) + 2× DGX Spark"></a><br><sub>1× RTX + 2× Spark</sub></td>
<td width="27%" valign="top">n/a<br><sub>Not measured for v3 (one card per checkpoint).</sub></td>
</tr>
<tr>
<td width="19%" valign="top"><a href="docs/models/glm5_flash.md"><b>GLM 5.3 Flash</b></a><br><sub>brandonmusic/GLM-5.3-Flash-tr3-4bpw</sub><br><sub>exl3-k4</sub></td>
<td width="27%" valign="top"><a href="benchmarks/glm5_flash/2026-10-18-smoke-glm-5-3-flash-tr3-4bpw-1rtx-2spark-glm53f-tr3-5090/card.svg"><img src="benchmarks/glm5_flash/2026-10-18-smoke-glm-5-3-flash-tr3-4bpw-1rtx-2spark-glm53f-tr3-5090/card.svg" alt="brandonmusic/GLM-5.3-Flash-tr3-4bpw on simulated 5090: RTX PRO 6000 (188 SMs) capped at 31.8 GiB + 2 Spark"></a><br><sub>simulated 5090: RTX PRO 6000 (188 SMs) capped at 31.8 GiB + 2 Spark</sub></td>
<td width="27%" valign="top"><a href="benchmarks/glm5_flash/2026-10-18-smoke-glm-5-3-flash-tr3-4bpw-1rtx-2spark-glm53f-tr3-min/card.svg"><img src="benchmarks/glm5_flash/2026-10-18-smoke-glm-5-3-flash-tr3-4bpw-1rtx-2spark-glm53f-tr3-min/card.svg" alt="brandonmusic/GLM-5.3-Flash-tr3-4bpw on 1× RTX PRO 6000 @ gpu-powerd 300/585 W (240–600) + 2× DGX Spark"></a><br><sub>1× RTX + 2× Spark</sub></td>
<td width="27%" valign="top">n/a<br><sub>Not measured for v3 (one card per checkpoint).</sub></td>
</tr>
<tr>
<td width="19%" valign="top"><a href="docs/models/mimo_v2.md"><b>MiMo V2</b></a><br><sub>XiaomiMiMo/MiMo-V2.6-Flash-MOPD</sub><br><sub>mxfp4-g32</sub></td>
<td width="27%" valign="top"><a href="benchmarks/mimo_v2/2026-10-18-smoke-mimo-v2-6-flash-mopd-1rtx-2spark-mimo26-flash-5090/card.svg"><img src="benchmarks/mimo_v2/2026-10-18-smoke-mimo-v2-6-flash-mopd-1rtx-2spark-mimo26-flash-5090/card.svg" alt="XiaomiMiMo/MiMo-V2.6-Flash-MOPD on simulated 5090: RTX PRO 6000 (188 SMs) capped at 31.8 GiB + 2 Spark"></a><br><sub>simulated 5090: RTX PRO 6000 (188 SMs) capped at 31.8 GiB + 2 Spark</sub></td>
<td width="27%" valign="top"><a href="benchmarks/mimo_v2/2026-10-18-smoke-mimo-v2-6-flash-mopd-1rtx-2spark-mimo26-flash-min/card.svg"><img src="benchmarks/mimo_v2/2026-10-18-smoke-mimo-v2-6-flash-mopd-1rtx-2spark-mimo26-flash-min/card.svg" alt="XiaomiMiMo/MiMo-V2.6-Flash-MOPD on 1× RTX PRO 6000 @ gpu-powerd 310/570 W (240–600) + 2× DGX Spark"></a><br><sub>1× RTX + 2× Spark</sub></td>
<td width="27%" valign="top">n/a<br><sub>Not measured for v3 (one card per checkpoint).</sub></td>
</tr>
<tr>
<td width="19%" valign="top"><a href="docs/models/mimo_v2.md"><b>MiMo V2</b></a><br><sub>XiaomiMiMo/MiMo-V2.6-Pro-MOPD</sub><br><sub>mxfp4-g32</sub></td>
<td width="27%" valign="top">n/a<br><sub>Doesn&#39;t fit one 32 GB card with this checkpoint.</sub></td>
<td width="27%" valign="top">n/a<br><sub>Not measured for v3 (one card per checkpoint).</sub></td>
<td width="27%" valign="top"><a href="benchmarks/mimo_v2/2026-10-18-smoke-mimo-v2-6-pro-mopd-2rtx-6spark-mimo-pro-max/card.svg"><img src="benchmarks/mimo_v2/2026-10-18-smoke-mimo-v2-6-pro-mopd-2rtx-6spark-mimo-pro-max/card.svg" alt="XiaomiMiMo/MiMo-V2.6-Pro-MOPD on 2× RTX PRO 6000 @ gpu-powerd 435/380 W (200–535) + 6× DGX Spark"></a><br><sub>2× RTX + 6× Spark</sub></td>
</tr>
<tr>
<td width="19%" valign="top"><a href="docs/models/qwen4.md"><b>Qwen 3.8</b></a><br><sub>wrldsuksgo2mars/Qwen3.8-Flash-Next-EXL3-K4.25-PLE-FP8-v1</sub><br><sub>exl3-k4+exl3-k5</sub></td>
<td width="27%" valign="top"><a href="benchmarks/qwen4/2026-10-18-smoke-qwen3-8-flash-next-exl3-k4-25-ple-fp8-v1-1rtx-1spark-qwen38-exl3-5090/card.svg"><img src="benchmarks/qwen4/2026-10-18-smoke-qwen3-8-flash-next-exl3-k4-25-ple-fp8-v1-1rtx-1spark-qwen38-exl3-5090/card.svg" alt="wrldsuksgo2mars/Qwen3.8-Flash-Next-EXL3-K4.25-PLE-FP8-v1 on simulated 5090: RTX PRO 6000 (188 SMs) capped at 31.8 GiB + 1 Spark"></a><br><sub>simulated 5090: RTX PRO 6000 (188 SMs) capped at 31.8 GiB + 1 Spark</sub></td>
<td width="27%" valign="top"><a href="benchmarks/qwen4/2026-10-18-smoke-qwen3-8-flash-next-exl3-k4-25-ple-fp8-v1-1rtx-qwen38-exl3-min/card.svg"><img src="benchmarks/qwen4/2026-10-18-smoke-qwen3-8-flash-next-exl3-k4-25-ple-fp8-v1-1rtx-qwen38-exl3-min/card.svg" alt="wrldsuksgo2mars/Qwen3.8-Flash-Next-EXL3-K4.25-PLE-FP8-v1 on 1× RTX PRO 6000 @ gpu-powerd 325/565 W (260–600)"></a><br><sub>1× RTX</sub></td>
<td width="27%" valign="top">n/a<br><sub>Not measured for v3 (one card per checkpoint).</sub></td>
</tr>
<tr>
<td width="19%" valign="top"><a href="docs/models/qwen4.md"><b>Qwen 3.8</b></a><br><sub>nvidia/Qwen3.8-Flash-Next-NVFP4</sub><br><sub>nvfp4-g16</sub></td>
<td width="27%" valign="top"><a href="benchmarks/qwen4/2026-10-18-smoke-qwen3-8-flash-next-nvfp4-1rtx-2spark-qwen38-nvfp4-5090/card.svg"><img src="benchmarks/qwen4/2026-10-18-smoke-qwen3-8-flash-next-nvfp4-1rtx-2spark-qwen38-nvfp4-5090/card.svg" alt="nvidia/Qwen3.8-Flash-Next-NVFP4 on simulated 5090: RTX PRO 6000 (188 SMs) capped at 31.8 GiB + 2 Spark"></a><br><sub>simulated 5090: RTX PRO 6000 (188 SMs) capped at 31.8 GiB + 2 Spark</sub></td>
<td width="27%" valign="top"><a href="benchmarks/qwen4/2026-10-18-smoke-qwen3-8-flash-next-nvfp4-1rtx-2spark-qwen38-nvfp4-min/card.svg"><img src="benchmarks/qwen4/2026-10-18-smoke-qwen3-8-flash-next-nvfp4-1rtx-2spark-qwen38-nvfp4-min/card.svg" alt="nvidia/Qwen3.8-Flash-Next-NVFP4 on 1× RTX PRO 6000 @ gpu-powerd 450/450 W + 2× DGX Spark"></a><br><sub>1× RTX + 2× Spark</sub></td>
<td width="27%" valign="top">n/a<br><sub>Not measured for v3 (one card per checkpoint).</sub></td>
</tr>
</table>

Each card is one launch of the basic profile: tok/s for C1 decode and concurrent code with thinking off, cold 8K prefill, logit fidelity against the family golden reference, prefix-cache restore exactness and lossless speculation. Older releases, other profiles and every report: [benchmarks](benchmarks/README.md).

<!-- results:end -->

### Known limits

- **Calibrated NVFP4 quants** (GLM 5.3, GLM 5.3 Flash, Qwen 3.8) report
  fidelity as informational: W4A4 is the checkpoint's own calibrated numerics.
- **Qwen 1× RTX** default cells are text-only. A fixed 262,144-token pool turns
  vision on for EXL3. [Details](docs/models/qwen4.md#known-limits)
- **GLM Flash tr3** vision needs `CHAT_TEMPLATE_FROM=zai-org/GLM-5.3-Flash`;
  without it `VISION=auto` serves text only and says so.
- **GLM 5.3 EXLR8** (work/p0 only, not in a release): with multi-lane prefill,
  prompts beyond about 8K tokens can hang at the coordinator's peer wait
  ([#408](https://github.com/tpurtell/cuteafd/issues/408)). Use
  `CUTEAFD_GLM_PREFILL_LANES=1` in the launch environment for long prompts
  until the fix lands.

### Methodology

Cards measure text C1, warmed C8 aggregate code throughput, cold ~8K prefill,
quick fidelity against the family golden reference and exact prefix-cache
restore. They are single launches: compare releases only with matched-prompt
runs. Simulated 5090 cells cap memory only; SM count, L2, clocks and power are
not emulated, so their speed is indicative and likely optimistic.
[How releases are measured](docs/release-testing.md) ·
[all reports](benchmarks/README.md)

Family pages hold supported checkpoints, engineering notes and known limits:
[DeepSeek V4.1 Flash](docs/models/deepseek_v41.md),
[DeepSeek V4 Flash/Pro](docs/models/deepseek_v4.md),
[GLM 5.3](docs/models/glm5.md), [GLM 5.3 Flash](docs/models/glm5_flash.md),
[MiMo V2 Flash / V2.6 Pro](docs/models/mimo_v2.md),
[Qwen 3.8 Flash Next](docs/models/qwen4.md).

## Storage

The checkpoints behind these numbers were served from
[SparkNest](https://github.com/tpurtell/sparknest), a distributed model store
across the cluster: hosts with a sealed local copy of a shard read it at NVMe
speed, others stream it over RoCE at about 5 GB/s. It is a separate project and
not required; any standard Hugging Face cache layout works.

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
   `ghcr.io/tpurtell/cuteafd-coordinator:v3.0.0` on the RTX host and
   `ghcr.io/tpurtell/cuteafd-spark-expert:v3.0.0` on each Spark; `docker pull`
   them on those hosts first (`./run.sh` does not pull). `./wip.sh --slot S --role both`
   plus `./run.sh --wip S --restart` is the faster loop while iterating.

V4.1 defaults to `CUTEAFD_V41_FP8_HEAD=all`: one E4M3 vocabulary head
(or one shard per GPU) shared by target and dSpark, with BF16 released after
packing. `off` selects BF16; experimental `draft` retains BF16 for the target
and adds FP8 for dSpark. The accepted target-head quality gate and matched RC2
controls support the release default.

## Working on it

[`AGENTS.md`](AGENTS.md) is the standing guide for agents and collaborators
working on CuteAFD. [`PLAN.md`](PLAN.md) is the roadmap.
[WIP hardware cards](docs/wip-cards.md) runs candidate builds through the
release card matrix. [Usage history and console access](docs/usage-and-console-access.md)
covers request accounting, the `/usage` dashboard and the console unlock secret.
**The full request log is on by default: prompts, model outputs and media are
stored in plain text on the serving host for 24 hours** (turn it off on
`/usage` settings or with `USAGE=off`).

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
