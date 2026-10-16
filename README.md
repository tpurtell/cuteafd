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

## What's new in v2.0.0

- Image input for every family that ships a vision tower; MiMo audio is placed
  automatically.
- One SM120 image runs RTX PRO 6000 and RTX 5090, with 32 GB memory admission
  and a 2M-token (1M on 32 GB) automatic KV target.
- MiMo V2.6 Flash MOPD is the MiMo Flash default.
- GLM 5.3 Flash gains Hugh Madden's ports #14–#29, with the compact index
  cache, shared replay records and tensor-core draft head on by default.
- V4 Flash and V4.1 now fit a 32 GB coordinator.
- Coordinators and Sparks can hold only their own shards.
  [Sliced checkpoints](docs/sliced-checkpoints.md)
- Quick, standard and full fidelity tiers score every quant against its
  family's official reference. [Fidelity](docs/fidelity-design.md)
- Release cards add C8 throughput and a 32 GB (RTX 5090) column.
  [How releases are measured](docs/release-testing.md)

Known limits are listed under [Models](#models).
Full notes: [v2.0.0 release](https://github.com/tpurtell/cuteafd/releases/tag/v2.0.0) ·
[all releases](https://github.com/tpurtell/cuteafd/releases)

## Models

Basic profile per model and quant on the natural minimum (1× RTX + the fewest
Sparks) and maximum (2× RTX + 4 or 6 Sparks). The 5090 column publishes only
real RTX 5090 measurements; cells stay pending until those exports arrive.

<!-- results:begin -->
<table>
<tr><th>Model · quant</th><th>5090</th><th>1× RTX</th><th>2× RTX</th></tr>
<tr>
<td width="19%" valign="top"><a href="docs/models/deepseek_v41.md"><b>DeepSeek V4.1</b></a><br><sub>deepseek-ai/DeepSeek-V4.1-Flash</sub><br><sub>mxfp4-g32</sub></td>
<td width="27%" valign="top">pending<br><sub>Awaiting real RTX 5090 exports; simulations are archived only.</sub></td>
<td width="27%" valign="top"><a href="benchmarks/deepseek_v41/2026-10-14-smoke-deepseek-v4-1-flash-1rtx-3spark-v41-flash-min/card.svg"><img src="benchmarks/deepseek_v41/2026-10-14-smoke-deepseek-v4-1-flash-1rtx-3spark-v41-flash-min/card.svg" alt="deepseek-ai/DeepSeek-V4.1-Flash on 1× RTX PRO 6000 @ gpu-powerd 445 W (415–540) + 3× DGX Spark"></a><br><sub>1× RTX + 3× Spark</sub></td>
<td width="27%" valign="top"><a href="benchmarks/deepseek_v41/2026-10-14-smoke-deepseek-v4-1-flash-2rtx-4spark-v41-flash-max/card.svg"><img src="benchmarks/deepseek_v41/2026-10-14-smoke-deepseek-v4-1-flash-2rtx-4spark-v41-flash-max/card.svg" alt="deepseek-ai/DeepSeek-V4.1-Flash on 2× RTX PRO 6000 @ gpu-powerd 455/445 W (300–550) + 4× DGX Spark"></a><br><sub>2× RTX + 4× Spark</sub></td>
</tr>
<tr>
<td width="19%" valign="top"><a href="docs/models/deepseek_v41.md"><b>DeepSeek V4.1</b></a><br><sub>nvidia/DeepSeek-V4.1-Flash-NVFP4</sub><br><sub>nvfp4-g16</sub></td>
<td width="27%" valign="top">pending<br><sub>Awaiting real RTX 5090 exports; simulations are archived only.</sub></td>
<td width="27%" valign="top"><a href="benchmarks/deepseek_v41/2026-10-15-smoke-deepseek-v4-1-flash-nvfp4-1rtx-4spark-v41-flash-nvfp4-min/card.svg"><img src="benchmarks/deepseek_v41/2026-10-15-smoke-deepseek-v4-1-flash-nvfp4-1rtx-4spark-v41-flash-nvfp4-min/card.svg" alt="nvidia/DeepSeek-V4.1-Flash-NVFP4 on 1× RTX PRO 6000 @ gpu-powerd 450 W + 4× DGX Spark"></a><br><sub>1× RTX + 4× Spark</sub></td>
<td width="27%" valign="top"><a href="benchmarks/deepseek_v41/2026-10-15-smoke-deepseek-v4-1-flash-nvfp4-2rtx-4spark-v41-flash-nvfp4-max/card.svg"><img src="benchmarks/deepseek_v41/2026-10-15-smoke-deepseek-v4-1-flash-nvfp4-2rtx-4spark-v41-flash-nvfp4-max/card.svg" alt="nvidia/DeepSeek-V4.1-Flash-NVFP4 on 2× RTX PRO 6000 @ gpu-powerd 455/390 W (240–545) + 4× DGX Spark"></a><br><sub>2× RTX + 4× Spark</sub></td>
</tr>
<tr>
<td width="19%" valign="top"><a href="docs/models/deepseek_v4.md"><b>DeepSeek V4</b></a><br><sub>deepseek-ai/DeepSeek-V4-Flash-0731</sub><br><sub>mxfp4-g32</sub></td>
<td width="27%" valign="top">pending<br><sub>Awaiting real RTX 5090 exports; simulations are archived only.</sub></td>
<td width="27%" valign="top"><a href="benchmarks/deepseek_v4/2026-10-14-smoke-deepseek-v4-flash-0731-1rtx-2spark-v4-flash-min/card.svg"><img src="benchmarks/deepseek_v4/2026-10-14-smoke-deepseek-v4-flash-0731-1rtx-2spark-v4-flash-min/card.svg" alt="deepseek-ai/DeepSeek-V4-Flash-0731 on 1× RTX PRO 6000 @ gpu-powerd 455/445 W (300–600) + 2× DGX Spark"></a><br><sub>1× RTX + 2× Spark</sub></td>
<td width="27%" valign="top"><a href="benchmarks/deepseek_v4/2026-10-14-smoke-deepseek-v4-flash-0731-2rtx-4spark-v4-flash-max/card.svg"><img src="benchmarks/deepseek_v4/2026-10-14-smoke-deepseek-v4-flash-0731-2rtx-4spark-v4-flash-max/card.svg" alt="deepseek-ai/DeepSeek-V4-Flash-0731 on 2× RTX PRO 6000 @ gpu-powerd 445/450 W (300–550) + 4× DGX Spark"></a><br><sub>2× RTX + 4× Spark</sub></td>
</tr>
<tr>
<td width="19%" valign="top"><a href="docs/models/deepseek_v4.md"><b>DeepSeek V4</b></a><br><sub>wrldsuksgo2mars/DeepSeek-V4-Pro-0813-EXL3-K2-calibrated-v1</sub><br><sub>exl3-k2</sub></td>
<td width="27%" valign="top">pending<br><sub>Awaiting real RTX 5090 exports; simulations are archived only.</sub></td>
<td width="27%" valign="top"><a href="benchmarks/deepseek_v4/2026-10-14-smoke-deepseek-v4-pro-0813-exl3-k2-calibrated-v1-1rtx-4spark-v4-pro-exl3-min/card.svg"><img src="benchmarks/deepseek_v4/2026-10-14-smoke-deepseek-v4-pro-0813-exl3-k2-calibrated-v1-1rtx-4spark-v4-pro-exl3-min/card.svg" alt="wrldsuksgo2mars/DeepSeek-V4-Pro-0813-EXL3-K2-calibrated-v1 on 1× RTX PRO 6000 @ gpu-powerd 440/455 W (300–550) + 4× DGX Spark"></a><br><sub>1× RTX + 4× Spark</sub></td>
<td width="27%" valign="top"><a href="benchmarks/deepseek_v4/2026-10-14-smoke-deepseek-v4-pro-0813-exl3-k2-calibrated-v1-2rtx-6spark-v4-pro-exl3-max/card.svg"><img src="benchmarks/deepseek_v4/2026-10-14-smoke-deepseek-v4-pro-0813-exl3-k2-calibrated-v1-2rtx-6spark-v4-pro-exl3-max/card.svg" alt="wrldsuksgo2mars/DeepSeek-V4-Pro-0813-EXL3-K2-calibrated-v1 on 2× RTX PRO 6000 @ gpu-powerd 450/445 W (300–550) + 6× DGX Spark"></a><br><sub>2× RTX + 6× Spark</sub></td>
</tr>
<tr>
<td width="19%" valign="top"><a href="docs/models/glm5.md"><b>GLM 5.3</b></a><br><sub>wrldsuksgo2mars/GLM-5.3-EXL3-K4-v1</sub><br><sub>exl3-k4</sub></td>
<td width="27%" valign="top">pending<br><sub>Awaiting real RTX 5090 exports; simulations are archived only.</sub></td>
<td width="27%" valign="top"><a href="benchmarks/glm5/2026-10-14-smoke-glm-5-3-exl3-k4-v1-1rtx-4spark-glm53-exl3-min/card.svg"><img src="benchmarks/glm5/2026-10-14-smoke-glm-5-3-exl3-k4-v1-1rtx-4spark-glm53-exl3-min/card.svg" alt="wrldsuksgo2mars/GLM-5.3-EXL3-K4-v1 on 1× RTX PRO 6000 @ gpu-powerd 450/450 W (300–550) + 4× DGX Spark"></a><br><sub>1× RTX + 4× Spark</sub></td>
<td width="27%" valign="top"><a href="benchmarks/glm5/2026-10-14-smoke-glm-5-3-exl3-k4-v1-2rtx-6spark-glm53-exl3-max/card.svg"><img src="benchmarks/glm5/2026-10-14-smoke-glm-5-3-exl3-k4-v1-2rtx-6spark-glm53-exl3-max/card.svg" alt="wrldsuksgo2mars/GLM-5.3-EXL3-K4-v1 on 2× RTX PRO 6000 @ gpu-powerd 450/450 W (405–490) + 6× DGX Spark"></a><br><sub>2× RTX + 6× Spark</sub></td>
</tr>
<tr>
<td width="19%" valign="top"><a href="docs/models/glm5.md"><b>GLM 5.3</b></a><br><sub>nvidia/GLM-5.3-NVFP4</sub><br><sub>nvfp4-g16</sub></td>
<td width="27%" valign="top">pending<br><sub>Awaiting real RTX 5090 exports; simulations are archived only.</sub></td>
<td width="27%" valign="top"><a href="benchmarks/glm5/2026-10-15-smoke-glm-5-3-nvfp4-1rtx-4spark-glm53-nvfp4-min/card.svg"><img src="benchmarks/glm5/2026-10-15-smoke-glm-5-3-nvfp4-1rtx-4spark-glm53-nvfp4-min/card.svg" alt="nvidia/GLM-5.3-NVFP4 on 1× RTX PRO 6000 @ gpu-powerd 435/460 W (380–495) + 4× DGX Spark"></a><br><sub>1× RTX + 4× Spark</sub><br><sub>Fidelity versus the BF16 root is informational for calibrated W4A4 (#230). C1 is below rc3 because of the Spark NVFP4 W4A4 small-row expert kernel; investigation is parked on #233. Historical rc3 is not a controlled A/B (325 W and different fabric/defaults). Measured Spark rail A: 100 Gb/s per worker; coordinator: one 400 Gb/s port. gpu-powerd 900 W shared budget; serving physical GPU1 mean limit 460.4 W (380-495 W), no measured power-cap window.</sub></td>
<td width="27%" valign="top"><a href="benchmarks/glm5/2026-10-15-smoke-glm-5-3-nvfp4-2rtx-6spark-glm53-nvfp4-max/card.svg"><img src="benchmarks/glm5/2026-10-15-smoke-glm-5-3-nvfp4-2rtx-6spark-glm53-nvfp4-max/card.svg" alt="nvidia/GLM-5.3-NVFP4 on 2× RTX PRO 6000 @ gpu-powerd 450/445 W (395–500) + 6× DGX Spark"></a><br><sub>2× RTX + 6× Spark</sub><br><sub>Fidelity versus the BF16 root is informational for calibrated W4A4 (#230). C1 is below rc3 because of the Spark NVFP4 W4A4 small-row expert kernel; investigation is parked on #233. Historical rc3 is not a controlled A/B (325 W and different fabric/defaults). Measured Spark rail A: 100 Gb/s per worker; coordinator: one 400 Gb/s port. Prefill/8k has foreign-traffic attribution warnings: 5.54-7.96 Gb/s per Spark and 16.57 Gb/s coordinator; retain raw prefill, not a clean A/B. gpu-powerd 900 W shared budget; mean limits 451.0/446.8 W (435-500/395-450 W); power-cap windows 2.004/1.006 s.</sub></td>
</tr>
<tr>
<td width="19%" valign="top"><a href="docs/models/glm5_flash.md"><b>GLM 5.3 Flash</b></a><br><sub>zai-org/GLM-5.3-Flash</sub><br><sub>fp8-block128x128/f32</sub></td>
<td width="27%" valign="top">pending<br><sub>Awaiting real RTX 5090 exports; simulations are archived only.</sub></td>
<td width="27%" valign="top"><a href="benchmarks/glm5_flash/2026-10-14-smoke-glm-5-3-flash-1rtx-4spark-glm53f-fp8-min/card.svg"><img src="benchmarks/glm5_flash/2026-10-14-smoke-glm-5-3-flash-1rtx-4spark-glm53f-fp8-min/card.svg" alt="zai-org/GLM-5.3-Flash on 1× RTX PRO 6000 @ gpu-powerd 445/450 W (300–550) + 4× DGX Spark"></a><br><sub>1× RTX + 4× Spark</sub></td>
<td width="27%" valign="top"><a href="benchmarks/glm5_flash/2026-10-14-smoke-glm-5-3-flash-2rtx-4spark-glm53f-fp8-max/card.svg"><img src="benchmarks/glm5_flash/2026-10-14-smoke-glm-5-3-flash-2rtx-4spark-glm53f-fp8-max/card.svg" alt="zai-org/GLM-5.3-Flash on 2× RTX PRO 6000 @ gpu-powerd 450/450 W (395–500) + 4× DGX Spark"></a><br><sub>2× RTX + 4× Spark</sub></td>
</tr>
<tr>
<td width="19%" valign="top"><a href="docs/models/glm5_flash.md"><b>GLM 5.3 Flash</b></a><br><sub>wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3.25-v1</sub><br><sub>exl3-k3+exl3-k4</sub></td>
<td width="27%" valign="top">pending<br><sub>Awaiting real RTX 5090 exports; simulations are archived only.</sub></td>
<td width="27%" valign="top"><a href="benchmarks/glm5_flash/2026-10-14-smoke-glm-5-3-flash-exl3-k3-25-v1-1rtx-2spark-glm53f-exl3-min/card.svg"><img src="benchmarks/glm5_flash/2026-10-14-smoke-glm-5-3-flash-exl3-k3-25-v1-1rtx-2spark-glm53f-exl3-min/card.svg" alt="wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3.25-v1 on 1× RTX PRO 6000 @ gpu-powerd 450/445 W (300–570) + 2× DGX Spark"></a><br><sub>1× RTX + 2× Spark</sub></td>
<td width="27%" valign="top"><a href="benchmarks/glm5_flash/2026-10-14-smoke-glm-5-3-flash-exl3-k3-25-v1-2rtx-glm53f-exl3-max/card.svg"><img src="benchmarks/glm5_flash/2026-10-14-smoke-glm-5-3-flash-exl3-k3-25-v1-2rtx-glm53f-exl3-max/card.svg" alt="wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3.25-v1 on 2× RTX PRO 6000 @ gpu-powerd 445/445 W (300–550)"></a><br><sub>2× RTX</sub></td>
</tr>
<tr>
<td width="19%" valign="top"><a href="docs/models/glm5_flash.md"><b>GLM 5.3 Flash</b></a><br><sub>nvidia/GLM-5.3-Flash-NVFP4</sub><br><sub>nvfp4-g16</sub></td>
<td width="27%" valign="top">pending<br><sub>Awaiting real RTX 5090 exports; simulations are archived only.</sub></td>
<td width="27%" valign="top"><a href="benchmarks/glm5_flash/2026-10-14-smoke-glm-5-3-flash-nvfp4-1rtx-2spark-glm53f-nvfp4-min/card.svg"><img src="benchmarks/glm5_flash/2026-10-14-smoke-glm-5-3-flash-nvfp4-1rtx-2spark-glm53f-nvfp4-min/card.svg" alt="nvidia/GLM-5.3-Flash-NVFP4 on 1× RTX PRO 6000 @ gpu-powerd 430/455 W (335–530) + 2× DGX Spark"></a><br><sub>1× RTX + 2× Spark</sub></td>
<td width="27%" valign="top"><a href="benchmarks/glm5_flash/2026-10-14-smoke-glm-5-3-flash-nvfp4-2rtx-2spark-glm53f-nvfp4-max/card.svg"><img src="benchmarks/glm5_flash/2026-10-14-smoke-glm-5-3-flash-nvfp4-2rtx-2spark-glm53f-nvfp4-max/card.svg" alt="nvidia/GLM-5.3-Flash-NVFP4 on 2× RTX PRO 6000 @ gpu-powerd 450/450 W + 2× DGX Spark"></a><br><sub>2× RTX + 2× Spark</sub></td>
</tr>
<tr>
<td width="19%" valign="top"><a href="docs/models/glm5_flash.md"><b>GLM 5.3 Flash</b></a><br><sub>brandonmusic/GLM-5.3-Flash-tr3-4bpw</sub><br><sub>exl3-k4</sub></td>
<td width="27%" valign="top">pending<br><sub>Awaiting real RTX 5090 exports; simulations are archived only.</sub></td>
<td width="27%" valign="top"><a href="benchmarks/glm5_flash/2026-10-14-smoke-glm-5-3-flash-tr3-4bpw-1rtx-2spark-glm53f-tr3-min/card.svg"><img src="benchmarks/glm5_flash/2026-10-14-smoke-glm-5-3-flash-tr3-4bpw-1rtx-2spark-glm53f-tr3-min/card.svg" alt="brandonmusic/GLM-5.3-Flash-tr3-4bpw on 1× RTX PRO 6000 @ gpu-powerd 435/460 W (300–550) + 2× DGX Spark"></a><br><sub>1× RTX + 2× Spark</sub></td>
<td width="27%" valign="top"><a href="benchmarks/glm5_flash/2026-10-14-smoke-glm-5-3-flash-tr3-4bpw-2rtx-2spark-glm53f-tr3-max/card.svg"><img src="benchmarks/glm5_flash/2026-10-14-smoke-glm-5-3-flash-tr3-4bpw-2rtx-2spark-glm53f-tr3-max/card.svg" alt="brandonmusic/GLM-5.3-Flash-tr3-4bpw on 2× RTX PRO 6000 @ gpu-powerd 450/445 W (435–460) + 2× DGX Spark"></a><br><sub>2× RTX + 2× Spark</sub></td>
</tr>
<tr>
<td width="19%" valign="top"><a href="docs/models/mimo_v2.md"><b>MiMo V2</b></a><br><sub>XiaomiMiMo/MiMo-V2.6-Flash-MOPD</sub><br><sub>mxfp4-g32</sub></td>
<td width="27%" valign="top">pending<br><sub>Awaiting real RTX 5090 exports; simulations are archived only.</sub></td>
<td width="27%" valign="top"><a href="benchmarks/mimo_v2/2026-10-14-smoke-mimo-v2-6-flash-mopd-1rtx-2spark-mimo26-flash-min/card.svg"><img src="benchmarks/mimo_v2/2026-10-14-smoke-mimo-v2-6-flash-mopd-1rtx-2spark-mimo26-flash-min/card.svg" alt="XiaomiMiMo/MiMo-V2.6-Flash-MOPD on 1× RTX PRO 6000 @ gpu-powerd 440/455 W (300–550) + 2× DGX Spark"></a><br><sub>1× RTX + 2× Spark</sub></td>
<td width="27%" valign="top"><a href="benchmarks/mimo_v2/2026-10-14-smoke-mimo-v2-6-flash-mopd-2rtx-4spark-mimo26-flash-max/card.svg"><img src="benchmarks/mimo_v2/2026-10-14-smoke-mimo-v2-6-flash-mopd-2rtx-4spark-mimo26-flash-max/card.svg" alt="XiaomiMiMo/MiMo-V2.6-Flash-MOPD on 2× RTX PRO 6000 @ gpu-powerd 445/445 W (300–550) + 4× DGX Spark"></a><br><sub>2× RTX + 4× Spark</sub></td>
</tr>
<tr>
<td width="19%" valign="top"><a href="docs/models/mimo_v2.md"><b>MiMo V2</b></a><br><sub>XiaomiMiMo/MiMo-V2.6-Pro-MOPD</sub><br><sub>mxfp4-g32</sub></td>
<td width="27%" valign="top">pending<br><sub>Awaiting real RTX 5090 exports; simulations are archived only.</sub></td>
<td width="27%" valign="top"><a href="benchmarks/mimo_v2/2026-10-14-smoke-mimo-v2-6-pro-mopd-1rtx-6spark-mimo-pro-min/card.svg"><img src="benchmarks/mimo_v2/2026-10-14-smoke-mimo-v2-6-pro-mopd-1rtx-6spark-mimo-pro-min/card.svg" alt="XiaomiMiMo/MiMo-V2.6-Pro-MOPD on 1× RTX PRO 6000 @ gpu-powerd 450/450 W (345–545) + 6× DGX Spark"></a><br><sub>1× RTX + 6× Spark</sub></td>
<td width="27%" valign="top"><a href="benchmarks/mimo_v2/2026-10-14-smoke-mimo-v2-6-pro-mopd-2rtx-6spark-mimo-pro-max/card.svg"><img src="benchmarks/mimo_v2/2026-10-14-smoke-mimo-v2-6-pro-mopd-2rtx-6spark-mimo-pro-max/card.svg" alt="XiaomiMiMo/MiMo-V2.6-Pro-MOPD on 2× RTX PRO 6000 @ gpu-powerd 450/450 W (300–550) + 6× DGX Spark"></a><br><sub>2× RTX + 6× Spark</sub></td>
</tr>
<tr>
<td width="19%" valign="top"><a href="docs/models/qwen4.md"><b>Qwen 3.8</b></a><br><sub>wrldsuksgo2mars/Qwen3.8-Flash-Next-EXL3-K4.25-PLE-FP8-v1</sub><br><sub>exl3-k4+exl3-k5</sub></td>
<td width="27%" valign="top">pending<br><sub>Awaiting real RTX 5090 exports; simulations are archived only.</sub></td>
<td width="27%" valign="top"><a href="benchmarks/qwen4/2026-10-14-smoke-qwen3-8-flash-next-exl3-k4-25-ple-fp8-v1-1rtx-qwen38-exl3-min/card.svg"><img src="benchmarks/qwen4/2026-10-14-smoke-qwen3-8-flash-next-exl3-k4-25-ple-fp8-v1-1rtx-qwen38-exl3-min/card.svg" alt="wrldsuksgo2mars/Qwen3.8-Flash-Next-EXL3-K4.25-PLE-FP8-v1 on 1× RTX PRO 6000 @ gpu-powerd 410/485 W (300–600)"></a><br><sub>1× RTX</sub></td>
<td width="27%" valign="top"><a href="benchmarks/qwen4/2026-10-14-smoke-qwen3-8-flash-next-exl3-k4-25-ple-fp8-v1-2rtx-qwen38-exl3-max/card.svg"><img src="benchmarks/qwen4/2026-10-14-smoke-qwen3-8-flash-next-exl3-k4-25-ple-fp8-v1-2rtx-qwen38-exl3-max/card.svg" alt="wrldsuksgo2mars/Qwen3.8-Flash-Next-EXL3-K4.25-PLE-FP8-v1 on 2× RTX PRO 6000 @ gpu-powerd 440/455 W (300–550)"></a><br><sub>2× RTX</sub></td>
</tr>
<tr>
<td width="19%" valign="top"><a href="docs/models/qwen4.md"><b>Qwen 3.8</b></a><br><sub>nvidia/Qwen3.8-Flash-Next-NVFP4</sub><br><sub>nvfp4-g16</sub></td>
<td width="27%" valign="top">pending<br><sub>Awaiting real RTX 5090 exports; simulations are archived only.</sub></td>
<td width="27%" valign="top"><a href="benchmarks/qwen4/2026-10-15-smoke-qwen3-8-flash-next-nvfp4-1rtx-2spark-qwen38-nvfp4-min/card.svg"><img src="benchmarks/qwen4/2026-10-15-smoke-qwen3-8-flash-next-nvfp4-1rtx-2spark-qwen38-nvfp4-min/card.svg" alt="nvidia/Qwen3.8-Flash-Next-NVFP4 on 1× RTX PRO 6000 @ gpu-powerd 390/465 W (370–500) + 2× DGX Spark"></a><br><sub>1× RTX + 2× Spark</sub></td>
<td width="27%" valign="top"><a href="benchmarks/qwen4/2026-10-14-smoke-qwen3-8-flash-next-nvfp4-2rtx-qwen38-nvfp4-max/card.svg"><img src="benchmarks/qwen4/2026-10-14-smoke-qwen3-8-flash-next-nvfp4-2rtx-qwen38-nvfp4-max/card.svg" alt="nvidia/Qwen3.8-Flash-Next-NVFP4 on 2× RTX PRO 6000 @ gpu-powerd 450/450 W (355–540)"></a><br><sub>2× RTX</sub></td>
</tr>
</table>

| Family | Checkpoint | Hardware | KV / req | C1 code | Concurrent code (aggregate) | prose | JSON | 8K prefill | TTFT | Quality | Report |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| [DeepSeek V4.1](docs/models/deepseek_v41.md) | DeepSeek-V4.1-Flash (mxfp4-g32) | 1× RTX + 3× Spark (1× RTX) | 2.1M tok / 16 req | 118 | C1: 118 · C4: 211 · C8: 248 · C16: 272 | 72.1 | 129 | 6,248 | 1.30 s | ✓ KL 0.015 · top-1 97.0% ✓ exact cache ✓ lossless spec | [2026-10-14 · 92ad5b03d029](benchmarks/deepseek_v41/2026-10-14-smoke-deepseek-v4-1-flash-1rtx-3spark-v41-flash-min/report.svg) |
| [DeepSeek V4.1](docs/models/deepseek_v41.md) | DeepSeek-V4.1-Flash (mxfp4-g32) | 2× RTX + 4× Spark (2× RTX) | 2.1M tok / 16 req | 152 | C1: 152 · C4: 288 · C8: 398 · C16: 453 | 79.2 | 164 | 8,290 | 979 ms | ✓ KL 0.016 · top-1 97.0% ✓ exact cache ✓ lossless spec | [2026-10-14 · 92ad5b03d029](benchmarks/deepseek_v41/2026-10-14-smoke-deepseek-v4-1-flash-2rtx-4spark-v41-flash-max/report.svg) |
| [DeepSeek V4.1](docs/models/deepseek_v41.md) | DeepSeek-V4.1-Flash-NVFP4 (nvfp4-g16) | 1× RTX + 4× Spark (1× RTX) | 2.1M tok / 16 req | 99.5 | C1: 99.5 · C4: 189 · C8: 223 · C16: 227 | 69.8 | 126 | 3,766 | 2.15 s | ✓ KL 0.062 · top-1 93.8% (info: calibrated W4A4) ✓ exact cache ✓ lossless spec | [2026-10-15 · bac064ac2cf5](benchmarks/deepseek_v41/2026-10-15-smoke-deepseek-v4-1-flash-nvfp4-1rtx-4spark-v41-flash-nvfp4-min/report.svg) |
| [DeepSeek V4.1](docs/models/deepseek_v41.md) | DeepSeek-V4.1-Flash-NVFP4 (nvfp4-g16) | 2× RTX + 4× Spark (2× RTX) | 2.1M tok / 16 req | 122 | C1: 122 · C4: 246 · C8: 315 · C16: 398 | 68.6 | 138 | 6,868 | 1.18 s | ✓ KL 0.066 · top-1 93.6% (info: calibrated W4A4) ✓ exact cache ✓ lossless spec | [2026-10-15 · bac064ac2cf5](benchmarks/deepseek_v41/2026-10-15-smoke-deepseek-v4-1-flash-nvfp4-2rtx-4spark-v41-flash-nvfp4-max/report.svg) |
| [DeepSeek V4](docs/models/deepseek_v4.md) | DeepSeek-V4-Flash-0731 (mxfp4-g32) | 1× RTX + 2× Spark (1× RTX) | 2.1M tok / 16 req | 141 | C1: 141 · C4: 286 · C8: 358 · C16: 385 | 75.8 | 152 | 4,351 | 1.86 s | ✓ KL 0.010 · top-1 96.9% ✓ exact cache ✓ lossless spec | [2026-10-14 · 92ad5b03d029](benchmarks/deepseek_v4/2026-10-14-smoke-deepseek-v4-flash-0731-1rtx-2spark-v4-flash-min/report.svg) |
| [DeepSeek V4](docs/models/deepseek_v4.md) | DeepSeek-V4-Flash-0731 (mxfp4-g32) | 2× RTX + 4× Spark (2× RTX) | 2.1M tok / 16 req | 278 | C1: 278 · C4: 584 · C8: 858 · C16: 1,114 | 120 | 265 | 6,846 | 1.19 s | ✓ KL 0.011 · top-1 96.9% ✓ exact cache ✓ lossless spec | [2026-10-14 · 92ad5b03d029](benchmarks/deepseek_v4/2026-10-14-smoke-deepseek-v4-flash-0731-2rtx-4spark-v4-flash-max/report.svg) |
| [DeepSeek V4](docs/models/deepseek_v4.md) | DeepSeek-V4-Pro-0813-EXL3-K2-calibrated-v1 (exl3-k2) | 1× RTX + 4× Spark (1× RTX) | 2.1M tok / 16 req | 72.0 | C1: 72.0 · C4: 147 · C8: 179 · C16: 220 | 39.6 | 81.3 | 2,665 | 3.05 s | ✓ KL 0.058 · top-1 92.5% ✓ exact cache ✓ lossless spec | [2026-10-14 · 92ad5b03d029](benchmarks/deepseek_v4/2026-10-14-smoke-deepseek-v4-pro-0813-exl3-k2-calibrated-v1-1rtx-4spark-v4-pro-exl3-min/report.svg) |
| [DeepSeek V4](docs/models/deepseek_v4.md) | DeepSeek-V4-Pro-0813-EXL3-K2-calibrated-v1 (exl3-k2) | 2× RTX + 6× Spark (2× RTX) | 2.1M tok / 16 req | 98.4 | C1: 98.4 · C4: 197 · C8: 244 · C16: 296 | 51.1 | 113 | 2,617 | 3.10 s | ✓ KL 0.057 · top-1 92.5% ✓ exact cache ✓ lossless spec | [2026-10-14 · 92ad5b03d029](benchmarks/deepseek_v4/2026-10-14-smoke-deepseek-v4-pro-0813-exl3-k2-calibrated-v1-2rtx-6spark-v4-pro-exl3-max/report.svg) |
| [GLM 5.3](docs/models/glm5.md) | GLM-5.3-EXL3-K4-v1 (exl3-k4) | 1× RTX + 4× Spark (1× RTX) | 1.2M tok / 16 req | 48.7 | C1: 48.7 · C4: 79.9 · C8: 90.6 · C16: 114 | 35.2 | 62.9 | 2,723 | 2.96 s | ✓ KL 0.036 · top-1 96.6% ✓ exact cache ✓ lossless spec | [2026-10-14 · 92ad5b03d029](benchmarks/glm5/2026-10-14-smoke-glm-5-3-exl3-k4-v1-1rtx-4spark-glm53-exl3-min/report.svg) |
| [GLM 5.3](docs/models/glm5.md) | GLM-5.3-EXL3-K4-v1 (exl3-k4) | 2× RTX + 6× Spark (2× RTX) | 2M tok / 16 req | 54.5 | C1: 54.5 · C4: 91.5 · C8: 102 · C16: 131 | 37.3 | 67.6 | 3,136 | 2.57 s | ✓ KL 0.036 · top-1 96.4% ✓ exact cache ✓ lossless spec | [2026-10-14 · 92ad5b03d029](benchmarks/glm5/2026-10-14-smoke-glm-5-3-exl3-k4-v1-2rtx-6spark-glm53-exl3-max/report.svg) |
| [GLM 5.3](docs/models/glm5.md) | GLM-5.3-NVFP4 (nvfp4-g16) | 1× RTX + 4× Spark (1× RTX) | 1.2M tok / 16 req | 38.8 | C1: 38.8 · C4: 63.1 · C8: 79.3 · C16: 96.8 | 28.9 | 53.6 | 2,605 | 3.09 s | ✓ KL 0.041 · top-1 95.7% (info: calibrated W4A4) ✓ exact cache ✓ lossless spec | [2026-10-15 · 78d27e34a349](benchmarks/glm5/2026-10-15-smoke-glm-5-3-nvfp4-1rtx-4spark-glm53-nvfp4-min/report.svg) |
| [GLM 5.3](docs/models/glm5.md) | GLM-5.3-NVFP4 (nvfp4-g16) | 2× RTX + 6× Spark (2× RTX) | 2M tok / 16 req | 60.4 | C1: 60.4 · C4: 90.1 · C8: 98.3 · C16: 120 | 36.6 | 68.0 | 3,140 | 2.57 s | ✓ KL 0.039 · top-1 95.7% (info: calibrated W4A4) ✓ exact cache ✓ lossless spec | [2026-10-15 · 78d27e34a349](benchmarks/glm5/2026-10-15-smoke-glm-5-3-nvfp4-2rtx-6spark-glm53-nvfp4-max/report.svg) |
| [GLM 5.3 Flash](docs/models/glm5_flash.md) | GLM-5.3-Flash (fp8-block128x128/f32) | 1× RTX + 4× Spark (1× RTX) | 2M tok / 16 req | 80.9 | C1: 80.9 · C4: 121 · C8: 146 · C16: 173 | 67.8 | 95.1 | 5,313 | 1.52 s | ✓ KL 0.011 · top-1 97.0% ✓ exact cache ✓ lossless spec | [2026-10-14 · 92ad5b03d029](benchmarks/glm5_flash/2026-10-14-smoke-glm-5-3-flash-1rtx-4spark-glm53f-fp8-min/report.svg) |
| [GLM 5.3 Flash](docs/models/glm5_flash.md) | GLM-5.3-Flash (fp8-block128x128/f32) | 2× RTX + 4× Spark (2× RTX) | 2M tok / 16 req | 93.1 | C1: 93.1 · C4: 133 · C8: 181 · C16: 239 | 55.3 | 102 | 5,139 | 1.57 s | ✓ KL 0.010 · top-1 97.6% ✓ exact cache ✓ lossless spec | [2026-10-14 · 92ad5b03d029](benchmarks/glm5_flash/2026-10-14-smoke-glm-5-3-flash-2rtx-4spark-glm53f-fp8-max/report.svg) |
| [GLM 5.3 Flash](docs/models/glm5_flash.md) | GLM-5.3-Flash-EXL3-K3.25-v1 (exl3-k3+exl3-k4) | 1× RTX + 2× Spark (1× RTX) | 2M tok / 16 req | 89.6 | C1: 89.6 · C4: 171 · C8: 219 · C16: 269 | 84.3 | 148 | 4,739 | 1.70 s | ✓ KL 0.036 · top-1 95.5% ✓ exact cache ✓ lossless spec | [2026-10-14 · 92ad5b03d029](benchmarks/glm5_flash/2026-10-14-smoke-glm-5-3-flash-exl3-k3-25-v1-1rtx-2spark-glm53f-exl3-min/report.svg) |
| [GLM 5.3 Flash](docs/models/glm5_flash.md) | GLM-5.3-Flash-EXL3-K3.25-v1 (exl3-k3+exl3-k4) | 2× RTX (2× RTX) | 4.9M tok / 16 req | 179 | C1: 179 · C4: 385 · C8: 542 · C16: 736 | 132 | 267 | 7,933 | 1.02 s | ✓ KL 0.036 · top-1 95.7% ✓ exact cache ✓ lossless spec | [2026-10-14 · 92ad5b03d029](benchmarks/glm5_flash/2026-10-14-smoke-glm-5-3-flash-exl3-k3-25-v1-2rtx-glm53f-exl3-max/report.svg) |
| [GLM 5.3 Flash](docs/models/glm5_flash.md) | GLM-5.3-Flash-NVFP4 (nvfp4-g16) | 1× RTX + 2× Spark (1× RTX) | 2M tok / 16 req | 73.2 | C1: 73.2 · C4: 112 · C8: 129 · C16: 155 | 60.9 | 80.8 | 4,507 | 1.79 s | ✓ KL 0.041 · top-1 95.8% (info: calibrated W4A4) ✓ exact cache ✓ lossless spec | [2026-10-14 · 92ad5b03d029](benchmarks/glm5_flash/2026-10-14-smoke-glm-5-3-flash-nvfp4-1rtx-2spark-glm53f-nvfp4-min/report.svg) |
| [GLM 5.3 Flash](docs/models/glm5_flash.md) | GLM-5.3-Flash-NVFP4 (nvfp4-g16) | 2× RTX + 2× Spark (2× RTX) | 2M tok / 16 req | 138 | C1: 138 · C4: 172 · C8: 277 · C16: 343 | 84.4 | 159 | 7,207 | 1.12 s | ✓ KL 0.042 · top-1 95.3% (info: calibrated W4A4) ✓ exact cache ✓ lossless spec | [2026-10-14 · 92ad5b03d029](benchmarks/glm5_flash/2026-10-14-smoke-glm-5-3-flash-nvfp4-2rtx-2spark-glm53f-nvfp4-max/report.svg) |
| [GLM 5.3 Flash](docs/models/glm5_flash.md) | GLM-5.3-Flash-tr3-4bpw (exl3-k4) | 1× RTX + 2× Spark (1× RTX) | 2M tok / 16 req | 86.7 | C1: 86.7 · C4: 154 · C8: 193 · C16: 227 | 73.4 | 113 | 5,025 | 1.60 s | ✓ KL 0.024 · top-1 96.5% ✓ exact cache ✓ lossless spec | [2026-10-14 · 92ad5b03d029](benchmarks/glm5_flash/2026-10-14-smoke-glm-5-3-flash-tr3-4bpw-1rtx-2spark-glm53f-tr3-min/report.svg) |
| [GLM 5.3 Flash](docs/models/glm5_flash.md) | GLM-5.3-Flash-tr3-4bpw (exl3-k4) | 2× RTX + 2× Spark (2× RTX) | 2M tok / 16 req | 146 | C1: 146 · C4: 282 · C8: 376 · C16: 574 | 108 | 192 | 6,030 | 1.34 s | ✓ KL 0.024 · top-1 96.5% ✓ exact cache ✓ lossless spec | [2026-10-14 · 92ad5b03d029](benchmarks/glm5_flash/2026-10-14-smoke-glm-5-3-flash-tr3-4bpw-2rtx-2spark-glm53f-tr3-max/report.svg) |
| [MiMo V2](docs/models/mimo_v2.md) | MiMo-V2.6-Flash-MOPD (mxfp4-g32) | 1× RTX + 2× Spark (1× RTX) | 2M tok / 16 req | 98.4 | C1: 98.4 · C4: 189 · C8: 266 · C16: 324 | 62.1 | 130 | 4,761 | 1.70 s | ✓ KL 0.020 · top-1 95.1% ✓ exact cache ✓ lossless spec | [2026-10-14 · 92ad5b03d029](benchmarks/mimo_v2/2026-10-14-smoke-mimo-v2-6-flash-mopd-1rtx-2spark-mimo26-flash-min/report.svg) |
| [MiMo V2](docs/models/mimo_v2.md) | MiMo-V2.6-Flash-MOPD (mxfp4-g32) | 2× RTX + 4× Spark (2× RTX) | 2M tok / 16 req | 266 | C1: 266 · C4: 565 · C8: 809 · C16: 1,165 | 134 | 333 | 9,364 | 861 ms | ✓ KL 0.018 · top-1 95.8% ✓ exact cache ✓ lossless spec | [2026-10-14 · 92ad5b03d029](benchmarks/mimo_v2/2026-10-14-smoke-mimo-v2-6-flash-mopd-2rtx-4spark-mimo26-flash-max/report.svg) |
| [MiMo V2](docs/models/mimo_v2.md) | MiMo-V2.6-Pro-MOPD (mxfp4-g32) | 1× RTX + 6× Spark (1× RTX) | 2M tok / 16 req | 60.2 | C1: 60.2 · C4: 134 · C8: 130 · C16: 160 | 38.5 | 76.9 | 3,389 | 2.38 s | ✓ KL 0.016 · top-1 95.2% ✓ exact cache ✓ lossless spec | [2026-10-14 · 92ad5b03d029](benchmarks/mimo_v2/2026-10-14-smoke-mimo-v2-6-pro-mopd-1rtx-6spark-mimo-pro-min/report.svg) |
| [MiMo V2](docs/models/mimo_v2.md) | MiMo-V2.6-Pro-MOPD (mxfp4-g32) | 2× RTX + 6× Spark (2× RTX) | 2M tok / 16 req | 78.6 | C1: 78.6 · C4: 158 · C8: 163 · C16: 195 | 49.0 | 95.3 | 3,644 | 2.21 s | ✓ KL 0.018 · top-1 96.1% ✓ exact cache ✓ lossless spec | [2026-10-14 · 92ad5b03d029](benchmarks/mimo_v2/2026-10-14-smoke-mimo-v2-6-pro-mopd-2rtx-6spark-mimo-pro-max/report.svg) |
| [Qwen 3.8](docs/models/qwen4.md) | Qwen3.8-Flash-Next-EXL3-K4.25-PLE-FP8-v1 (exl3-k4+exl3-k5) | 1× RTX (1× RTX) | 446K tok / 16 req | 205 | C1: 205 · C4: 593 · C8: 959 · C16: 1,334 | 136 | 238 | 9,128 | 873 ms | ✓ KL 0.020 · top-1 96.4% ✓ exact cache ✓ lossless spec | [2026-10-14 · 92ad5b03d029](benchmarks/qwen4/2026-10-14-smoke-qwen3-8-flash-next-exl3-k4-25-ple-fp8-v1-1rtx-qwen38-exl3-min/report.svg) |
| [Qwen 3.8](docs/models/qwen4.md) | Qwen3.8-Flash-Next-EXL3-K4.25-PLE-FP8-v1 (exl3-k4+exl3-k5) | 2× RTX (2× RTX) | 2M tok / 16 req | 209 | C1: 209 · C4: 664 · C8: 1,034 · C16: 1,498 | 137 | 253 | 8,845 | 903 ms | ✓ KL 0.020 · top-1 96.0% ✓ exact cache ✓ lossless spec | [2026-10-14 · 92ad5b03d029](benchmarks/qwen4/2026-10-14-smoke-qwen3-8-flash-next-exl3-k4-25-ple-fp8-v1-2rtx-qwen38-exl3-max/report.svg) |
| [Qwen 3.8](docs/models/qwen4.md) | Qwen3.8-Flash-Next-NVFP4 (nvfp4-g16) | 1× RTX + 2× Spark (1× RTX) | 2M tok / 16 req | 120 | C1: 120 · C4: 269 · C8: 441 · C16: 481 | 87.2 | 132 | 2,040 | 3.91 s | ✓ KL 0.052 · top-1 93.9% (info: calibrated W4A4) ✓ exact cache ✓ lossless spec | [2026-10-15 · 3672f5c21e41](benchmarks/qwen4/2026-10-15-smoke-qwen3-8-flash-next-nvfp4-1rtx-2spark-qwen38-nvfp4-min/report.svg) |
| [Qwen 3.8](docs/models/qwen4.md) | Qwen3.8-Flash-Next-NVFP4 (nvfp4-g16) | 2× RTX (2× RTX) | 2M tok / 16 req | 220 | C1: 220 · C4: 612 · C8: 1,039 · C16: 1,412 | 152 | 255 | 11,337 | 704 ms | ✓ KL 0.039 · top-1 93.7% (info: calibrated W4A4) ✓ exact cache ✓ lossless spec | [2026-10-14 · 92ad5b03d029](benchmarks/qwen4/2026-10-14-smoke-qwen3-8-flash-next-nvfp4-2rtx-qwen38-nvfp4-max/report.svg) |

tok/s; C1 decode and concurrent code aggregate with thinking off (C1/C4/C8/C16 on PRO 6000; through C8 at ≤32 GiB, within server admission), 8K prefill cold. Quality: logit fidelity against the family golden reference, prefix-cache restore exactness, lossless speculation. Every row is `benchmarks/current-matrix.json` resolved against the reports under `benchmarks/`; a held or pending cell renders why it has no card. The 5090 column publishes real hardware only.

<!-- results:end -->

### Known limits

- **V4.1 NVFP4 is not recommended:** it fails the fidelity check (confident
  top-1 96.5–97.1% against 98%). Use the official MXFP4
  `deepseek-ai/DeepSeek-V4.1-Flash`, which passes every check.
  [Details](docs/models/deepseek_v41.md#known-limits)
- **V4 Pro EXL3 K2 minimum** fails fidelity at KL 0.0605 against 0.06; the
  maximum passes. [Details](docs/models/deepseek_v4.md#known-limits)
- **Qwen 1× RTX** default cells are text-only, and NVFP4 admits less than
  262,144 tokens of context. A fixed 262,144-token pool turns vision on for EXL3.
  [Details](docs/models/qwen4.md#known-limits)
- **V4 on one RTX** is pool-limited (Flash 663K, Pro 455K tokens) and leaves
  GPU1 underused on two; TP2 placement is v3 work.
- **MiMo sliced checkpoints:** copy the checkpoint's `dflash/` directory into
  the slice by hand until the next release.
  [Details](docs/sliced-checkpoints.md#known-limit-v200)
- **GLM Flash tr3** vision needs `CHAT_TEMPLATE_FROM=zai-org/GLM-5.3-Flash`; without
  it `VISION=auto` serves text only and says so.

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
   `ghcr.io/tpurtell/cuteafd-coordinator:v2.0.0` on the RTX host and
   `ghcr.io/tpurtell/cuteafd-spark-expert:v2.0.0` on each Spark; `docker pull`
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
