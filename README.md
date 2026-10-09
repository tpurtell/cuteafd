<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/brand/cuteafd-logo-color-dark.svg">
    <img src="assets/brand/cuteafd-logo-color.svg" alt="cuteafd" width="480">
  </picture>
</p>

## Usage History And Console Access

Supported coordinator images keep usage metadata in
`~/.cache/cuteafd/<instance>/usage/usage.sqlite` on the launch host (7 days,
256 MiB by default). Metadata never contains prompts or completions. The
separate full-log tier defaults to 24 hours and 1 GiB in `usage-log.sqlite`.
**With the full log on, user prompts and model outputs are stored in plain
text for the retention period.** Media are replaced by MIME/size/SHA-256
references; credentials and headers are not logged. Serving/gateway payload
capture hooks are a separate rollout; storage alone does not capture bodies.

The launcher prints a console unlock link once on host stdout. The persistent
secret lives at `~/.cache/cuteafd/console/secret` (0600 in an owned 0700
directory). Console cookies do not authorize API calls, and an API key does
not unlock console data. Rotate with `scripts/launch/console-secret.sh rotate`
or `run.sh --rotate-console-secret`; running coordinators reload within 10
seconds. New flags are omitted for older images and older WIP binaries.

Cookie-protected `/console/usage/settings` controls retention, caps, full-log
recording, client IP (off by default), and benchmark recording. Clear the full
log with `POST /console/usage/log/clear`, or both tiers with
`POST /console/usage/clear`. With the coordinator stopped, deleting
`~/.cache/cuteafd/<instance>/usage/usage-log.sqlite*` clears only payloads.
`--usage off` removes request accounting entirely; absent `--usage-dir`, tests
and standalone APIs use an in-memory store. Only daily aggregates may outlive
metadata retention (90 days by default), without request or session IDs.

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
Sparks) and maximum (2× RTX + 4 or 6 Sparks). The 5090 column is simulated on a
memory-capped RTX PRO 6000 until real cards arrive.

<!-- results:begin -->
<table>
<tr><th>Model · quant</th><th>5090</th><th>1× RTX</th><th>2× RTX</th></tr>
<tr>
<td width="19%" valign="top"><a href="docs/models/deepseek_v41.md"><b>DeepSeek V4.1</b></a><br><sub>deepseek-ai/DeepSeek-V4.1-Flash</sub><br><sub>mxfp4-g32</sub></td>
<td width="27%" valign="top"><a href="benchmarks/deepseek_v41/2026-10-09-smoke-deepseek-v4-1-flash-1rtx-3spark-v41-flash-rc3-sim5090/card.svg"><img src="benchmarks/deepseek_v41/2026-10-09-smoke-deepseek-v4-1-flash-1rtx-3spark-v41-flash-rc3-sim5090/card.svg" alt="deepseek-ai/DeepSeek-V4.1-Flash on simulated 5090: RTX PRO 6000 (188 SMs) capped at 31.8 GiB + 3 Spark"></a><br><sub>simulated 5090: RTX PRO 6000 (188 SMs) capped at 31.8 GiB + 3 Spark</sub><br><sub>Starts from rc3 (rc2: startup ring-budget failure).</sub></td>
<td width="27%" valign="top"><a href="benchmarks/deepseek_v41/2026-10-09-smoke-deepseek-v4-1-flash-1rtx-3spark-v41-flash-min-rc3/card.svg"><img src="benchmarks/deepseek_v41/2026-10-09-smoke-deepseek-v4-1-flash-1rtx-3spark-v41-flash-min-rc3/card.svg" alt="deepseek-ai/DeepSeek-V4.1-Flash on 1× RTX PRO 6000 @ 325 W + 3× DGX Spark"></a><br><sub>1× RTX + 3× Spark</sub></td>
<td width="27%" valign="top"><a href="benchmarks/deepseek_v41/2026-10-09-smoke-deepseek-v4-1-flash-2rtx-4spark-v41-flash-max-rc3/card.svg"><img src="benchmarks/deepseek_v41/2026-10-09-smoke-deepseek-v4-1-flash-2rtx-4spark-v41-flash-max-rc3/card.svg" alt="deepseek-ai/DeepSeek-V4.1-Flash on 2× RTX PRO 6000 @ 325 W + 4× DGX Spark"></a><br><sub>2× RTX + 4× Spark</sub></td>
</tr>
<tr>
<td width="19%" valign="top"><a href="docs/models/deepseek_v41.md"><b>DeepSeek V4.1</b></a><br><sub>nvidia/DeepSeek-V4.1-Flash-NVFP4</sub><br><sub>nvfp4-g16</sub></td>
<td width="27%" valign="top"><a href="benchmarks/deepseek_v41/2026-10-09-smoke-deepseek-v4-1-flash-nvfp4-1rtx-4spark-v41-flash-nvfp4-rc3-sim5090/card.svg"><img src="benchmarks/deepseek_v41/2026-10-09-smoke-deepseek-v4-1-flash-nvfp4-1rtx-4spark-v41-flash-nvfp4-rc3-sim5090/card.svg" alt="nvidia/DeepSeek-V4.1-Flash-NVFP4 on simulated 5090: RTX PRO 6000 (188 SMs) capped at 31.8 GiB + 4 Spark"></a><br><sub>simulated 5090: RTX PRO 6000 (188 SMs) capped at 31.8 GiB + 4 Spark</sub><br><sub><b>FAIL: not recommended.</b> Fails the calibrated confident-top-1 fidelity check (96.5% vs 98%). Use the official MXFP4 checkpoint (deepseek-ai/DeepSeek-V4.1-Flash), which passes every check. Uses 4 Sparks: TP3 fits by memory, but the launcher supports NVFP4 only at TP4.</sub></td>
<td width="27%" valign="top"><a href="benchmarks/deepseek_v41/2026-10-09-smoke-deepseek-v4-1-flash-nvfp4-1rtx-4spark-v41-flash-nvfp4-min-rc3/card.svg"><img src="benchmarks/deepseek_v41/2026-10-09-smoke-deepseek-v4-1-flash-nvfp4-1rtx-4spark-v41-flash-nvfp4-min-rc3/card.svg" alt="nvidia/DeepSeek-V4.1-Flash-NVFP4 on 1× RTX PRO 6000 @ 325 W + 4× DGX Spark"></a><br><sub>1× RTX + 4× Spark</sub><br><sub><b>FAIL: not recommended.</b> Fails the calibrated confident-top-1 fidelity check (96.7% vs 98%). Use the official MXFP4 checkpoint (deepseek-ai/DeepSeek-V4.1-Flash), which passes every check. Uses 4 Sparks: TP3 fits by memory, but the launcher supports NVFP4 only at TP4.</sub></td>
<td width="27%" valign="top"><a href="benchmarks/deepseek_v41/2026-10-09-smoke-deepseek-v4-1-flash-nvfp4-2rtx-4spark-v41-flash-nvfp4-max-rc3/card.svg"><img src="benchmarks/deepseek_v41/2026-10-09-smoke-deepseek-v4-1-flash-nvfp4-2rtx-4spark-v41-flash-nvfp4-max-rc3/card.svg" alt="nvidia/DeepSeek-V4.1-Flash-NVFP4 on 2× RTX PRO 6000 @ 325 W + 4× DGX Spark"></a><br><sub>2× RTX + 4× Spark</sub><br><sub><b>FAIL: not recommended.</b> Fails the calibrated confident-top-1 fidelity check (97.1% vs 98%). Use the official MXFP4 checkpoint (deepseek-ai/DeepSeek-V4.1-Flash), which passes every check.</sub></td>
</tr>
<tr>
<td width="19%" valign="top"><a href="docs/models/deepseek_v4.md"><b>DeepSeek V4</b></a><br><sub>deepseek-ai/DeepSeek-V4-Flash-0731</sub><br><sub>mxfp4-g32</sub></td>
<td width="27%" valign="top"><a href="benchmarks/deepseek_v4/2026-10-09-smoke-deepseek-v4-flash-0731-1rtx-2spark-v4-flash-rc3-sim5090/card.svg"><img src="benchmarks/deepseek_v4/2026-10-09-smoke-deepseek-v4-flash-0731-1rtx-2spark-v4-flash-rc3-sim5090/card.svg" alt="deepseek-ai/DeepSeek-V4-Flash-0731 on simulated 5090: RTX PRO 6000 (188 SMs) capped at 31.8 GiB + 2 Spark"></a><br><sub>simulated 5090: RTX PRO 6000 (188 SMs) capped at 31.8 GiB + 2 Spark</sub><br><sub>Fits a 32 GB card from rc3 (rc2: planner-only no-fit).</sub></td>
<td width="27%" valign="top"><a href="benchmarks/deepseek_v4/2026-10-09-smoke-deepseek-v4-flash-0731-1rtx-2spark-v4-flash-min-rc3/card.svg"><img src="benchmarks/deepseek_v4/2026-10-09-smoke-deepseek-v4-flash-0731-1rtx-2spark-v4-flash-min-rc3/card.svg" alt="deepseek-ai/DeepSeek-V4-Flash-0731 on 1× RTX PRO 6000 @ 325 W + 2× DGX Spark"></a><br><sub>1× RTX + 2× Spark</sub></td>
<td width="27%" valign="top"><a href="benchmarks/deepseek_v4/2026-10-09-smoke-deepseek-v4-flash-0731-2rtx-4spark-v4-flash-max-rc3/card.svg"><img src="benchmarks/deepseek_v4/2026-10-09-smoke-deepseek-v4-flash-0731-2rtx-4spark-v4-flash-max-rc3/card.svg" alt="deepseek-ai/DeepSeek-V4-Flash-0731 on 2× RTX PRO 6000 @ 325 W + 4× DGX Spark"></a><br><sub>2× RTX + 4× Spark</sub></td>
</tr>
<tr>
<td width="19%" valign="top"><a href="docs/models/deepseek_v4.md"><b>DeepSeek V4</b></a><br><sub>wrldsuksgo2mars/DeepSeek-V4-Pro-0813-EXL3-K2-calibrated-v1</sub><br><sub>exl3-k2</sub></td>
<td width="27%" valign="top"><a href="benchmarks/deepseek_v4/2026-10-09-smoke-v4-pro-exl3-no-fit-rc3-sim5090/report.json">doesn't fit 1× 5090 (32 GB) + 6 Sparks</a><br><sub>coordinator weights need 46.0 GiB, over the 32 GiB coordinator budget; rtx0 full memory layout needs 62369870091 bytes, budget 31031138713 bytes, shortfall 31338731378 bytes</sub><br><sub>Memory rejection with all six Sparks (rc3 planner); no server launched, no performance.</sub></td>
<td width="27%" valign="top"><a href="benchmarks/deepseek_v4/2026-10-09-smoke-deepseek-v4-pro-0813-exl3-k2-calibrated-v1-1rtx-4spark-v4-pro-exl3-min-rc3/card.svg"><img src="benchmarks/deepseek_v4/2026-10-09-smoke-deepseek-v4-pro-0813-exl3-k2-calibrated-v1-1rtx-4spark-v4-pro-exl3-min-rc3/card.svg" alt="wrldsuksgo2mars/DeepSeek-V4-Pro-0813-EXL3-K2-calibrated-v1 on 1× RTX PRO 6000 @ 325 W + 4× DGX Spark"></a><br><sub>1× RTX + 4× Spark</sub><br><sub>Fidelity FAIL: KL 0.0605 &gt; 0.06 (top-1 92.0%); threshold-straddling since rc1, gate unchanged.</sub></td>
<td width="27%" valign="top"><a href="benchmarks/deepseek_v4/2026-10-09-smoke-deepseek-v4-pro-0813-exl3-k2-calibrated-v1-2rtx-6spark-v4-pro-exl3-max-rc3/card.svg"><img src="benchmarks/deepseek_v4/2026-10-09-smoke-deepseek-v4-pro-0813-exl3-k2-calibrated-v1-2rtx-6spark-v4-pro-exl3-max-rc3/card.svg" alt="wrldsuksgo2mars/DeepSeek-V4-Pro-0813-EXL3-K2-calibrated-v1 on 2× RTX PRO 6000 @ 325 W + 6× DGX Spark"></a><br><sub>2× RTX + 6× Spark</sub></td>
</tr>
<tr>
<td width="19%" valign="top"><a href="docs/models/glm5.md"><b>GLM 5.3</b></a><br><sub>wrldsuksgo2mars/GLM-5.3-EXL3-K4-v1</sub><br><sub>exl3-k4</sub></td>
<td width="27%" valign="top"><a href="benchmarks/glm5/2026-10-09-smoke-glm53-exl3-no-fit-rc3-sim5090/report.json">doesn't fit 1× 5090 (32 GB) + 6 Sparks</a><br><sub>rtx0 full memory layout needs 38219720132 bytes, budget 31997506355 bytes, shortfall 6222213777 bytes</sub><br><sub>Memory rejection with all six Sparks (rc3 planner); no server launched, no performance.</sub></td>
<td width="27%" valign="top"><a href="benchmarks/glm5/2026-10-09-smoke-glm-5-3-exl3-k4-v1-1rtx-4spark-glm53-exl3-min-rc3/card.svg"><img src="benchmarks/glm5/2026-10-09-smoke-glm-5-3-exl3-k4-v1-1rtx-4spark-glm53-exl3-min-rc3/card.svg" alt="wrldsuksgo2mars/GLM-5.3-EXL3-K4-v1 on 1× RTX PRO 6000 @ 325 W + 4× DGX Spark"></a><br><sub>1× RTX + 4× Spark</sub></td>
<td width="27%" valign="top"><a href="benchmarks/glm5/2026-10-09-smoke-glm-5-3-exl3-k4-v1-2rtx-6spark-glm53-exl3-max-rc3/card.svg"><img src="benchmarks/glm5/2026-10-09-smoke-glm-5-3-exl3-k4-v1-2rtx-6spark-glm53-exl3-max-rc3/card.svg" alt="wrldsuksgo2mars/GLM-5.3-EXL3-K4-v1 on 2× RTX PRO 6000 @ 325 W + 6× DGX Spark"></a><br><sub>2× RTX + 6× Spark</sub></td>
</tr>
<tr>
<td width="19%" valign="top"><a href="docs/models/glm5.md"><b>GLM 5.3</b></a><br><sub>nvidia/GLM-5.3-NVFP4</sub><br><sub>nvfp4-g16</sub></td>
<td width="27%" valign="top"><a href="benchmarks/glm5/2026-10-09-smoke-glm53-nvfp4-no-fit-rc3-sim5090/report.json">doesn't fit 1× 5090 (32 GB) + 6 Sparks</a><br><sub>coordinator weights need 52.6 GiB, over the 32 GiB coordinator budget; rtx0 full memory layout needs 54111029708 bytes, budget 31997506355 bytes, shortfall 22113523353 bytes</sub><br><sub>Memory rejection with all six Sparks (rc3 planner); no server launched, no performance.</sub></td>
<td width="27%" valign="top"><a href="benchmarks/glm5/2026-10-09-smoke-glm-5-3-nvfp4-1rtx-4spark-glm53-nvfp4-min-rc3/card.svg"><img src="benchmarks/glm5/2026-10-09-smoke-glm-5-3-nvfp4-1rtx-4spark-glm53-nvfp4-min-rc3/card.svg" alt="nvidia/GLM-5.3-NVFP4 on 1× RTX PRO 6000 @ 325 W + 4× DGX Spark"></a><br><sub>1× RTX + 4× Spark</sub></td>
<td width="27%" valign="top"><a href="benchmarks/glm5/2026-10-09-smoke-glm-5-3-nvfp4-2rtx-6spark-glm53-nvfp4-max-rc3/card.svg"><img src="benchmarks/glm5/2026-10-09-smoke-glm-5-3-nvfp4-2rtx-6spark-glm53-nvfp4-max-rc3/card.svg" alt="nvidia/GLM-5.3-NVFP4 on 2× RTX PRO 6000 @ 325 W + 6× DGX Spark"></a><br><sub>2× RTX + 6× Spark</sub></td>
</tr>
<tr>
<td width="19%" valign="top"><a href="docs/models/glm5_flash.md"><b>GLM 5.3 Flash</b></a><br><sub>zai-org/GLM-5.3-Flash</sub><br><sub>fp8-block128x128/f32</sub></td>
<td width="27%" valign="top"><a href="benchmarks/glm5_flash/2026-10-09-smoke-glm-5-3-flash-1rtx-4spark-glm53f-fp8-rc3-sim5090/card.svg"><img src="benchmarks/glm5_flash/2026-10-09-smoke-glm-5-3-flash-1rtx-4spark-glm53f-fp8-rc3-sim5090/card.svg" alt="zai-org/GLM-5.3-Flash on simulated 5090: RTX PRO 6000 (188 SMs) capped at 31.8 GiB + 4 Spark"></a><br><sub>simulated 5090: RTX PRO 6000 (188 SMs) capped at 31.8 GiB + 4 Spark</sub></td>
<td width="27%" valign="top"><a href="benchmarks/glm5_flash/2026-10-09-smoke-glm-5-3-flash-1rtx-4spark-glm53f-fp8-min-rc3/card.svg"><img src="benchmarks/glm5_flash/2026-10-09-smoke-glm-5-3-flash-1rtx-4spark-glm53f-fp8-min-rc3/card.svg" alt="zai-org/GLM-5.3-Flash on 1× RTX PRO 6000 @ 325 W + 4× DGX Spark"></a><br><sub>1× RTX + 4× Spark</sub></td>
<td width="27%" valign="top"><a href="benchmarks/glm5_flash/2026-10-09-smoke-glm-5-3-flash-2rtx-4spark-glm53f-fp8-max-rc3/card.svg"><img src="benchmarks/glm5_flash/2026-10-09-smoke-glm-5-3-flash-2rtx-4spark-glm53f-fp8-max-rc3/card.svg" alt="zai-org/GLM-5.3-Flash on 2× RTX PRO 6000 @ 325 W + 4× DGX Spark"></a><br><sub>2× RTX + 4× Spark</sub></td>
</tr>
<tr>
<td width="19%" valign="top"><a href="docs/models/glm5_flash.md"><b>GLM 5.3 Flash</b></a><br><sub>wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3.25-v1</sub><br><sub>exl3-k3+exl3-k4</sub></td>
<td width="27%" valign="top"><a href="benchmarks/glm5_flash/2026-10-09-smoke-glm-5-3-flash-exl3-k3-25-v1-1rtx-2spark-glm53f-exl3-rc3-sim5090/card.svg"><img src="benchmarks/glm5_flash/2026-10-09-smoke-glm-5-3-flash-exl3-k3-25-v1-1rtx-2spark-glm53f-exl3-rc3-sim5090/card.svg" alt="wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3.25-v1 on simulated 5090: RTX PRO 6000 (188 SMs) capped at 31.8 GiB + 2 Spark"></a><br><sub>simulated 5090: RTX PRO 6000 (188 SMs) capped at 31.8 GiB + 2 Spark</sub></td>
<td width="27%" valign="top"><a href="benchmarks/glm5_flash/2026-10-09-smoke-glm-5-3-flash-exl3-k3-25-v1-1rtx-2spark-glm53f-exl3-min-rc3/card.svg"><img src="benchmarks/glm5_flash/2026-10-09-smoke-glm-5-3-flash-exl3-k3-25-v1-1rtx-2spark-glm53f-exl3-min-rc3/card.svg" alt="wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3.25-v1 on 1× RTX PRO 6000 @ 325 W + 2× DGX Spark"></a><br><sub>1× RTX + 2× Spark</sub></td>
<td width="27%" valign="top"><a href="benchmarks/glm5_flash/2026-10-09-smoke-glm-5-3-flash-exl3-k3-25-v1-2rtx-4spark-glm53f-exl3-max-rc3/card.svg"><img src="benchmarks/glm5_flash/2026-10-09-smoke-glm-5-3-flash-exl3-k3-25-v1-2rtx-4spark-glm53f-exl3-max-rc3/card.svg" alt="wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3.25-v1 on 2× RTX PRO 6000 @ 325 W + 4× DGX Spark"></a><br><sub>2× RTX + 4× Spark</sub></td>
</tr>
<tr>
<td width="19%" valign="top"><a href="docs/models/glm5_flash.md"><b>GLM 5.3 Flash</b></a><br><sub>nvidia/GLM-5.3-Flash-NVFP4</sub><br><sub>nvfp4-g16</sub></td>
<td width="27%" valign="top"><a href="benchmarks/glm5_flash/2026-10-09-smoke-glm-5-3-flash-nvfp4-1rtx-2spark-glm53f-nvfp4-rc3-sim5090/card.svg"><img src="benchmarks/glm5_flash/2026-10-09-smoke-glm-5-3-flash-nvfp4-1rtx-2spark-glm53f-nvfp4-rc3-sim5090/card.svg" alt="nvidia/GLM-5.3-Flash-NVFP4 on simulated 5090: RTX PRO 6000 (188 SMs) capped at 31.8 GiB + 2 Spark"></a><br><sub>simulated 5090: RTX PRO 6000 (188 SMs) capped at 31.8 GiB + 2 Spark</sub></td>
<td width="27%" valign="top"><a href="benchmarks/glm5_flash/2026-10-09-smoke-glm-5-3-flash-nvfp4-1rtx-2spark-glm53f-nvfp4-min-rc3/card.svg"><img src="benchmarks/glm5_flash/2026-10-09-smoke-glm-5-3-flash-nvfp4-1rtx-2spark-glm53f-nvfp4-min-rc3/card.svg" alt="nvidia/GLM-5.3-Flash-NVFP4 on 1× RTX PRO 6000 @ 325 W + 2× DGX Spark"></a><br><sub>1× RTX + 2× Spark</sub></td>
<td width="27%" valign="top"><a href="benchmarks/glm5_flash/2026-10-09-smoke-glm-5-3-flash-nvfp4-2rtx-4spark-glm53f-nvfp4-max-rc3/card.svg"><img src="benchmarks/glm5_flash/2026-10-09-smoke-glm-5-3-flash-nvfp4-2rtx-4spark-glm53f-nvfp4-max-rc3/card.svg" alt="nvidia/GLM-5.3-Flash-NVFP4 on 2× RTX PRO 6000 @ 325 W + 4× DGX Spark"></a><br><sub>2× RTX + 4× Spark</sub></td>
</tr>
<tr>
<td width="19%" valign="top"><a href="docs/models/glm5_flash.md"><b>GLM 5.3 Flash</b></a><br><sub>brandonmusic/GLM-5.3-Flash-tr3-4bpw</sub><br><sub>exl3-k4</sub></td>
<td width="27%" valign="top"><a href="benchmarks/glm5_flash/2026-10-09-smoke-glm-5-3-flash-tr3-4bpw-1rtx-2spark-glm53f-tr3-rc3-sim5090/card.svg"><img src="benchmarks/glm5_flash/2026-10-09-smoke-glm-5-3-flash-tr3-4bpw-1rtx-2spark-glm53f-tr3-rc3-sim5090/card.svg" alt="brandonmusic/GLM-5.3-Flash-tr3-4bpw on simulated 5090: RTX PRO 6000 (188 SMs) capped at 31.8 GiB + 2 Spark"></a><br><sub>simulated 5090: RTX PRO 6000 (188 SMs) capped at 31.8 GiB + 2 Spark</sub></td>
<td width="27%" valign="top"><a href="benchmarks/glm5_flash/2026-10-09-smoke-glm-5-3-flash-tr3-4bpw-1rtx-2spark-glm53f-tr3-min-rc3/card.svg"><img src="benchmarks/glm5_flash/2026-10-09-smoke-glm-5-3-flash-tr3-4bpw-1rtx-2spark-glm53f-tr3-min-rc3/card.svg" alt="brandonmusic/GLM-5.3-Flash-tr3-4bpw on 1× RTX PRO 6000 @ 325 W + 2× DGX Spark"></a><br><sub>1× RTX + 2× Spark</sub></td>
<td width="27%" valign="top"><a href="benchmarks/glm5_flash/2026-10-09-smoke-glm-5-3-flash-tr3-4bpw-2rtx-4spark-glm53f-tr3-max-rc3/card.svg"><img src="benchmarks/glm5_flash/2026-10-09-smoke-glm-5-3-flash-tr3-4bpw-2rtx-4spark-glm53f-tr3-max-rc3/card.svg" alt="brandonmusic/GLM-5.3-Flash-tr3-4bpw on 2× RTX PRO 6000 @ 325 W + 4× DGX Spark"></a><br><sub>2× RTX + 4× Spark</sub></td>
</tr>
<tr>
<td width="19%" valign="top"><a href="docs/models/mimo_v2.md"><b>MiMo V2</b></a><br><sub>XiaomiMiMo/MiMo-V2.6-Flash-MOPD</sub><br><sub>mxfp4-g32</sub></td>
<td width="27%" valign="top"><a href="benchmarks/mimo_v2/2026-10-09-smoke-mimo-v2-6-flash-mopd-1rtx-2spark-mimo26-flash-rc3-sim5090/card.svg"><img src="benchmarks/mimo_v2/2026-10-09-smoke-mimo-v2-6-flash-mopd-1rtx-2spark-mimo26-flash-rc3-sim5090/card.svg" alt="XiaomiMiMo/MiMo-V2.6-Flash-MOPD on simulated 5090: RTX PRO 6000 (188 SMs) capped at 31.8 GiB + 2 Spark"></a><br><sub>simulated 5090: RTX PRO 6000 (188 SMs) capped at 31.8 GiB + 2 Spark</sub></td>
<td width="27%" valign="top"><a href="benchmarks/mimo_v2/2026-10-09-smoke-mimo-v2-6-flash-mopd-1rtx-2spark-mimo26-flash-min-rc3/card.svg"><img src="benchmarks/mimo_v2/2026-10-09-smoke-mimo-v2-6-flash-mopd-1rtx-2spark-mimo26-flash-min-rc3/card.svg" alt="XiaomiMiMo/MiMo-V2.6-Flash-MOPD on 1× RTX PRO 6000 @ 325 W + 2× DGX Spark"></a><br><sub>1× RTX + 2× Spark</sub></td>
<td width="27%" valign="top"><a href="benchmarks/mimo_v2/2026-10-09-smoke-mimo-v2-6-flash-mopd-2rtx-4spark-mimo26-flash-max-rc3/card.svg"><img src="benchmarks/mimo_v2/2026-10-09-smoke-mimo-v2-6-flash-mopd-2rtx-4spark-mimo26-flash-max-rc3/card.svg" alt="XiaomiMiMo/MiMo-V2.6-Flash-MOPD on 2× RTX PRO 6000 @ 325 W + 4× DGX Spark"></a><br><sub>2× RTX + 4× Spark</sub></td>
</tr>
<tr>
<td width="19%" valign="top"><a href="docs/models/mimo_v2.md"><b>MiMo V2</b></a><br><sub>XiaomiMiMo/MiMo-V2.6-Pro-MOPD</sub><br><sub>mxfp4-g32</sub></td>
<td width="27%" valign="top"><a href="benchmarks/mimo_v2/2026-10-09-smoke-mimo-pro-no-fit-rc3-sim5090/report.json">doesn't fit 1× 5090 (32 GB) + 6 Sparks</a><br><sub>rtx0 full memory layout needs 33511016304 bytes, budget 31031138713 bytes, shortfall 2479877591 bytes</sub><br><sub>Memory rejection with all six Sparks (rc3 planner); no server launched, no performance.</sub></td>
<td width="27%" valign="top"><a href="benchmarks/mimo_v2/2026-10-09-smoke-mimo-v2-6-pro-mopd-1rtx-6spark-mimo-pro-min-rc3/card.svg"><img src="benchmarks/mimo_v2/2026-10-09-smoke-mimo-v2-6-pro-mopd-1rtx-6spark-mimo-pro-min-rc3/card.svg" alt="XiaomiMiMo/MiMo-V2.6-Pro-MOPD on 1× RTX PRO 6000 @ 325 W + 6× DGX Spark"></a><br><sub>1× RTX + 6× Spark</sub></td>
<td width="27%" valign="top"><a href="benchmarks/mimo_v2/2026-10-09-smoke-mimo-v2-6-pro-mopd-2rtx-6spark-mimo-pro-max-rc3/card.svg"><img src="benchmarks/mimo_v2/2026-10-09-smoke-mimo-v2-6-pro-mopd-2rtx-6spark-mimo-pro-max-rc3/card.svg" alt="XiaomiMiMo/MiMo-V2.6-Pro-MOPD on 2× RTX PRO 6000 @ 325 W + 6× DGX Spark"></a><br><sub>2× RTX + 6× Spark</sub></td>
</tr>
<tr>
<td width="19%" valign="top"><a href="docs/models/qwen4.md"><b>Qwen 3.8</b></a><br><sub>wrldsuksgo2mars/Qwen3.8-Flash-Next-EXL3-K4.25-PLE-FP8-v1</sub><br><sub>exl3-k4+exl3-k5</sub></td>
<td width="27%" valign="top"><a href="benchmarks/qwen4/2026-10-09-smoke-qwen3-8-flash-next-exl3-k4-25-ple-fp8-v1-1rtx-1spark-qwen38-exl3-rc3-sim5090/card.svg"><img src="benchmarks/qwen4/2026-10-09-smoke-qwen3-8-flash-next-exl3-k4-25-ple-fp8-v1-1rtx-1spark-qwen38-exl3-rc3-sim5090/card.svg" alt="wrldsuksgo2mars/Qwen3.8-Flash-Next-EXL3-K4.25-PLE-FP8-v1 on simulated 5090: RTX PRO 6000 (188 SMs) capped at 31.8 GiB + 1 Spark"></a><br><sub>simulated 5090: RTX PRO 6000 (188 SMs) capped at 31.8 GiB + 1 Spark</sub></td>
<td width="27%" valign="top"><a href="benchmarks/qwen4/2026-10-09-smoke-qwen3-8-flash-next-exl3-k4-25-ple-fp8-v1-1rtx-qwen38-exl3-min-rc3/card.svg"><img src="benchmarks/qwen4/2026-10-09-smoke-qwen3-8-flash-next-exl3-k4-25-ple-fp8-v1-1rtx-qwen38-exl3-min-rc3/card.svg" alt="wrldsuksgo2mars/Qwen3.8-Flash-Next-EXL3-K4.25-PLE-FP8-v1 on 1× RTX PRO 6000 @ 325 W"></a><br><sub>1× RTX</sub><br><sub>Text only: automatic admission keeps the 2M KV target and turns the vision tower off. With vision: <a href="benchmarks/qwen4/2026-10-09-smoke-qwen3-8-flash-next-exl3-k4-25-ple-fp8-v1-1rtx-qwen38-exl3-nospark-vision-rc3/report.svg">1× RTX, no Sparks (local experts), vision on RTX0, fixed 262,144 pool</a>.</sub></td>
<td width="27%" align="center">n/a: fits one RTX (no two-GPU split for Qwen)</td>
</tr>
<tr>
<td width="19%" valign="top"><a href="docs/models/qwen4.md"><b>Qwen 3.8</b></a><br><sub>nvidia/Qwen3.8-Flash-Next-NVFP4</sub><br><sub>nvfp4-g16</sub></td>
<td width="27%" valign="top"><a href="benchmarks/qwen4/2026-10-09-smoke-qwen3-8-flash-next-nvfp4-1rtx-2spark-qwen38-nvfp4-rc3-sim5090/card.svg"><img src="benchmarks/qwen4/2026-10-09-smoke-qwen3-8-flash-next-nvfp4-1rtx-2spark-qwen38-nvfp4-rc3-sim5090/card.svg" alt="nvidia/Qwen3.8-Flash-Next-NVFP4 on simulated 5090: RTX PRO 6000 (188 SMs) capped at 31.8 GiB + 2 Spark"></a><br><sub>simulated 5090: RTX PRO 6000 (188 SMs) capped at 31.8 GiB + 2 Spark</sub><br><sub>Effective context 243,456, below the 262,144-token agentic floor (automatic pool).</sub></td>
<td width="27%" valign="top"><a href="benchmarks/qwen4/2026-10-09-smoke-qwen3-8-flash-next-nvfp4-1rtx-qwen38-nvfp4-min-rc3/card.svg"><img src="benchmarks/qwen4/2026-10-09-smoke-qwen3-8-flash-next-nvfp4-1rtx-qwen38-nvfp4-min-rc3/card.svg" alt="nvidia/Qwen3.8-Flash-Next-NVFP4 on 1× RTX PRO 6000 @ 325 W"></a><br><sub>1× RTX</sub><br><sub>Effective context 217,856, below the 262,144-token agentic floor (automatic pool). Text only: automatic admission keeps the 2M KV target and turns the vision tower off.</sub></td>
<td width="27%" align="center">n/a: fits one RTX (no two-GPU split for Qwen)</td>
</tr>
</table>

| Family | Checkpoint | Hardware | KV / req | C1 code | Concurrent code (aggregate) | prose | JSON | 8K prefill | TTFT | Quality | Report |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| [DeepSeek V4.1](docs/models/deepseek_v41.md) | DeepSeek-V4.1-Flash (mxfp4-g32) | simulated 5090: RTX PRO 6000 (188 SMs) capped at 31.8 GiB + 3 Spark (5090) | 1.1M tok / 16 req | 112 | C8: 239 | 64.4 | 119 | 6,346 | 1.28 s | ✓ KL 0.017 · top-1 97.0% ✓ exact cache ✓ lossless spec | [2026-10-09 · v2.0.0-rc3](benchmarks/deepseek_v41/2026-10-09-smoke-deepseek-v4-1-flash-1rtx-3spark-v41-flash-rc3-sim5090/report.svg) |
| [DeepSeek V4.1](docs/models/deepseek_v41.md) | DeepSeek-V4.1-Flash (mxfp4-g32) | 1× RTX + 3× Spark (1× RTX) | 2.1M tok / 16 req | 119 | C8: 304 | 74.5 | 135 | 6,357 | 1.28 s | ✓ KL 0.015 · top-1 96.4% ✓ exact cache | [2026-10-09 · v2.0.0-rc3](benchmarks/deepseek_v41/2026-10-09-smoke-deepseek-v4-1-flash-1rtx-3spark-v41-flash-min-rc3/report.svg) |
| [DeepSeek V4.1](docs/models/deepseek_v41.md) | DeepSeek-V4.1-Flash (mxfp4-g32) | 2× RTX + 4× Spark (2× RTX) | 2.1M tok / 16 req | 158 | C8: 458 | 85.1 | 165 | 7,325 | 1.11 s | ✓ KL 0.014 · top-1 97.7% ✓ exact cache | [2026-10-09 · v2.0.0-rc3](benchmarks/deepseek_v41/2026-10-09-smoke-deepseek-v4-1-flash-2rtx-4spark-v41-flash-max-rc3/report.svg) |
| [DeepSeek V4.1](docs/models/deepseek_v41.md) | DeepSeek-V4.1-Flash-NVFP4 (nvfp4-g16) | simulated 5090: RTX PRO 6000 (188 SMs) capped at 31.8 GiB + 4 Spark (5090) | 1.1M tok / 16 req | 78.1 | C8: 185 | 44.8 | 98.9 | 5,267 | 1.54 s | ⚠ **FAILED** failed: Logit fidelity | [2026-10-09 · v2.0.0-rc3](benchmarks/deepseek_v41/2026-10-09-smoke-deepseek-v4-1-flash-nvfp4-1rtx-4spark-v41-flash-nvfp4-rc3-sim5090/report.svg) |
| [DeepSeek V4.1](docs/models/deepseek_v41.md) | DeepSeek-V4.1-Flash-NVFP4 (nvfp4-g16) | 1× RTX + 4× Spark (1× RTX) | 2.1M tok / 16 req | 77.0 | C8: 254 | 53.1 | 90.9 | 5,441 | 1.49 s | ⚠ **FAILED** failed: Logit fidelity | [2026-10-09 · v2.0.0-rc3](benchmarks/deepseek_v41/2026-10-09-smoke-deepseek-v4-1-flash-nvfp4-1rtx-4spark-v41-flash-nvfp4-min-rc3/report.svg) |
| [DeepSeek V4.1](docs/models/deepseek_v41.md) | DeepSeek-V4.1-Flash-NVFP4 (nvfp4-g16) | 2× RTX + 4× Spark (2× RTX) | 2.1M tok / 16 req | 93.8 | C8: 284 | 52.6 | 108 | 6,678 | 1.21 s | ⚠ **FAILED** failed: Logit fidelity | [2026-10-09 · v2.0.0-rc3](benchmarks/deepseek_v41/2026-10-09-smoke-deepseek-v4-1-flash-nvfp4-2rtx-4spark-v41-flash-nvfp4-max-rc3/report.svg) |
| [DeepSeek V4](docs/models/deepseek_v4.md) | DeepSeek-V4-Flash-0731 (mxfp4-g32) | simulated 5090: RTX PRO 6000 (188 SMs) capped at 31.8 GiB + 2 Spark (5090) | 886K tok / 8 req | 120 | C8: 279 | 48.8 | 117 | 4,249 | 1.91 s | ✓ KL 0.011 · top-1 96.8% ✓ exact cache | [2026-10-09 · v2.0.0-rc3](benchmarks/deepseek_v4/2026-10-09-smoke-deepseek-v4-flash-0731-1rtx-2spark-v4-flash-rc3-sim5090/report.svg) |
| [DeepSeek V4](docs/models/deepseek_v4.md) | DeepSeek-V4-Flash-0731 (mxfp4-g32) | 1× RTX + 2× Spark (1× RTX) | 650K tok / 8 req | 149 | C8: 403 | 61.8 | 149 | 4,183 | 1.94 s | ✓ KL 0.011 · top-1 97.5% ✓ exact cache | [2026-10-09 · v2.0.0-rc3](benchmarks/deepseek_v4/2026-10-09-smoke-deepseek-v4-flash-0731-1rtx-2spark-v4-flash-min-rc3/report.svg) |
| [DeepSeek V4](docs/models/deepseek_v4.md) | DeepSeek-V4-Flash-0731 (mxfp4-g32) | 2× RTX + 4× Spark (2× RTX) | 1.3M tok / 8 req | 199 | C8: 549 | 89.6 | 217 | 5,153 | 1.57 s | ✓ KL 0.012 · top-1 97.3% ✓ exact cache ✓ lossless spec | [2026-10-09 · v2.0.0-rc3](benchmarks/deepseek_v4/2026-10-09-smoke-deepseek-v4-flash-0731-2rtx-4spark-v4-flash-max-rc3/report.svg) |
| [DeepSeek V4](docs/models/deepseek_v4.md) | DeepSeek-V4-Pro-0813-EXL3-K2-calibrated-v1 (exl3-k2) | simulated 5090: RTX PRO 6000 (188 SMs) capped at 31.8 GiB + 6 Spark (5090) | — | — | — | — | — | — | — | doesn't fit: coordinator weights need 46.0 GiB, over the 32 GiB coordinator budget; rtx0 full memory layout needs 62369870091 bytes, budget 31031138713 bytes, shortfall 31338731378 bytes | [planner report](benchmarks/deepseek_v4/2026-10-09-smoke-v4-pro-exl3-no-fit-rc3-sim5090/report.json) |
| [DeepSeek V4](docs/models/deepseek_v4.md) | DeepSeek-V4-Pro-0813-EXL3-K2-calibrated-v1 (exl3-k2) | 1× RTX + 4× Spark (1× RTX) | 447K tok / 8 req | 71.6 | C8: 189 | 37.6 | 80.3 | 2,080 | 3.91 s | ⚠ **FAILED** failed: Logit fidelity | [2026-10-09 · v2.0.0-rc3](benchmarks/deepseek_v4/2026-10-09-smoke-deepseek-v4-pro-0813-exl3-k2-calibrated-v1-1rtx-4spark-v4-pro-exl3-min-rc3/report.svg) |
| [DeepSeek V4](docs/models/deepseek_v4.md) | DeepSeek-V4-Pro-0813-EXL3-K2-calibrated-v1 (exl3-k2) | 2× RTX + 6× Spark (2× RTX) | 1.3M tok / 8 req | 97.9 | C8: 240 | 47.4 | 102 | 2,450 | 3.33 s | ✓ KL 0.060 · top-1 92.6% ✓ exact cache | [2026-10-09 · v2.0.0-rc3](benchmarks/deepseek_v4/2026-10-09-smoke-deepseek-v4-pro-0813-exl3-k2-calibrated-v1-2rtx-6spark-v4-pro-exl3-max-rc3/report.svg) |
| [GLM 5.3](docs/models/glm5.md) | GLM-5.3-EXL3-K4-v1 (exl3-k4) | simulated 5090: RTX PRO 6000 (188 SMs) capped at 31.8 GiB + 6 Spark (5090) | — | — | — | — | — | — | — | doesn't fit: rtx0 full memory layout needs 38219720132 bytes, budget 31997506355 bytes, shortfall 6222213777 bytes | [planner report](benchmarks/glm5/2026-10-09-smoke-glm53-exl3-no-fit-rc3-sim5090/report.json) |
| [GLM 5.3](docs/models/glm5.md) | GLM-5.3-EXL3-K4-v1 (exl3-k4) | 1× RTX + 4× Spark (1× RTX) | 1.1M tok / 8 req | 45.1 | C8: 91.0 | 36.2 | 57.7 | 2,820 | 2.85 s | ✓ KL 0.037 · top-1 96.3% ✓ exact cache | [2026-10-09 · v2.0.0-rc3](benchmarks/glm5/2026-10-09-smoke-glm-5-3-exl3-k4-v1-1rtx-4spark-glm53-exl3-min-rc3/report.svg) |
| [GLM 5.3](docs/models/glm5.md) | GLM-5.3-EXL3-K4-v1 (exl3-k4) | 2× RTX + 6× Spark (2× RTX) | 1.3M tok / 8 req | 53.6 | C8: 106 | 40.5 | 68.4 | 1,861 | 4.34 s | ✓ KL 0.038 · top-1 96.2% ✓ exact cache ✓ lossless spec | [2026-10-09 · v2.0.0-rc3](benchmarks/glm5/2026-10-09-smoke-glm-5-3-exl3-k4-v1-2rtx-6spark-glm53-exl3-max-rc3/report.svg) |
| [GLM 5.3](docs/models/glm5.md) | GLM-5.3-NVFP4 (nvfp4-g16) | simulated 5090: RTX PRO 6000 (188 SMs) capped at 31.8 GiB + 6 Spark (5090) | — | — | — | — | — | — | — | doesn't fit: coordinator weights need 52.6 GiB, over the 32 GiB coordinator budget; rtx0 full memory layout needs 54111029708 bytes, budget 31997506355 bytes, shortfall 22113523353 bytes | [planner report](benchmarks/glm5/2026-10-09-smoke-glm53-nvfp4-no-fit-rc3-sim5090/report.json) |
| [GLM 5.3](docs/models/glm5.md) | GLM-5.3-NVFP4 (nvfp4-g16) | 1× RTX + 4× Spark (1× RTX) | 1.1M tok / 8 req | 44.4 | C8: 79.1 | 32.8 | 61.0 | 2,518 | 3.19 s | ✓ KL 0.032 · top-1 95.9% ✓ exact cache | [2026-10-09 · v2.0.0-rc3](benchmarks/glm5/2026-10-09-smoke-glm-5-3-nvfp4-1rtx-4spark-glm53-nvfp4-min-rc3/report.svg) |
| [GLM 5.3](docs/models/glm5.md) | GLM-5.3-NVFP4 (nvfp4-g16) | 2× RTX + 6× Spark (2× RTX) | 1.3M tok / 8 req | 62.5 | C8: 110 | 38.4 | 76.4 | 2,617 | 3.08 s | ✓ KL 0.033 · top-1 95.9% ✓ exact cache | [2026-10-09 · v2.0.0-rc3](benchmarks/glm5/2026-10-09-smoke-glm-5-3-nvfp4-2rtx-6spark-glm53-nvfp4-max-rc3/report.svg) |
| [GLM 5.3 Flash](docs/models/glm5_flash.md) | GLM-5.3-Flash (fp8-block128x128/f32) | simulated 5090: RTX PRO 6000 (188 SMs) capped at 31.8 GiB + 4 Spark (5090) | 1.5M tok / 8 req | 73.7 | C8: 132 | 60.9 | 87.6 | 5,881 | 1.36 s | ✓ KL 0.011 · top-1 97.5% ✓ exact cache ✓ lossless spec | [2026-10-09 · v2.0.0-rc3](benchmarks/glm5_flash/2026-10-09-smoke-glm-5-3-flash-1rtx-4spark-glm53f-fp8-rc3-sim5090/report.svg) |
| [GLM 5.3 Flash](docs/models/glm5_flash.md) | GLM-5.3-Flash (fp8-block128x128/f32) | 1× RTX + 4× Spark (1× RTX) | 2M tok / 8 req | 73.4 | C8: 131 | 59.8 | 78.6 | 5,878 | 1.37 s | ✓ KL 0.011 · top-1 97.5% ✓ exact cache | [2026-10-09 · v2.0.0-rc3](benchmarks/glm5_flash/2026-10-09-smoke-glm-5-3-flash-1rtx-4spark-glm53f-fp8-min-rc3/report.svg) |
| [GLM 5.3 Flash](docs/models/glm5_flash.md) | GLM-5.3-Flash (fp8-block128x128/f32) | 2× RTX + 4× Spark (2× RTX) | 2M tok / 8 req | 72.7 | C8: 139 | 63.1 | 86.1 | 6,162 | 1.31 s | ✓ KL 0.011 · top-1 97.0% ✓ exact cache | [2026-10-09 · v2.0.0-rc3](benchmarks/glm5_flash/2026-10-09-smoke-glm-5-3-flash-2rtx-4spark-glm53f-fp8-max-rc3/report.svg) |
| [GLM 5.3 Flash](docs/models/glm5_flash.md) | GLM-5.3-Flash-EXL3-K3.25-v1 (exl3-k3+exl3-k4) | simulated 5090: RTX PRO 6000 (188 SMs) capped at 31.8 GiB + 2 Spark (5090) | 1.5M tok / 8 req | 90.1 | C8: 149 | 66.0 | 107 | 5,054 | 1.60 s | ✓ KL 0.036 · top-1 95.4% ✓ exact cache | [2026-10-09 · v2.0.0-rc3](benchmarks/glm5_flash/2026-10-09-smoke-glm-5-3-flash-exl3-k3-25-v1-1rtx-2spark-glm53f-exl3-rc3-sim5090/report.svg) |
| [GLM 5.3 Flash](docs/models/glm5_flash.md) | GLM-5.3-Flash-EXL3-K3.25-v1 (exl3-k3+exl3-k4) | 1× RTX + 2× Spark (1× RTX) | 2M tok / 8 req | 93.2 | C8: 150 | 68.8 | 102 | 4,938 | 1.63 s | ✓ KL 0.036 · top-1 95.4% ✓ exact cache | [2026-10-09 · v2.0.0-rc3](benchmarks/glm5_flash/2026-10-09-smoke-glm-5-3-flash-exl3-k3-25-v1-1rtx-2spark-glm53f-exl3-min-rc3/report.svg) |
| [GLM 5.3 Flash](docs/models/glm5_flash.md) | GLM-5.3-Flash-EXL3-K3.25-v1 (exl3-k3+exl3-k4) | 2× RTX + 4× Spark (2× RTX) | 2M tok / 8 req | 112 | C8: 251 | 97.7 | 168 | 7,158 | 1.12 s | ✓ KL 0.039 · top-1 95.3% ✓ exact cache ✓ lossless spec | [2026-10-09 · v2.0.0-rc3](benchmarks/glm5_flash/2026-10-09-smoke-glm-5-3-flash-exl3-k3-25-v1-2rtx-4spark-glm53f-exl3-max-rc3/report.svg) |
| [GLM 5.3 Flash](docs/models/glm5_flash.md) | GLM-5.3-Flash-NVFP4 (nvfp4-g16) | simulated 5090: RTX PRO 6000 (188 SMs) capped at 31.8 GiB + 2 Spark (5090) | 1.5M tok / 8 req | 69.6 | C8: 125 | 54.6 | 77.9 | 5,236 | 1.54 s | ✓ KL 0.029 · top-1 96.4% ✓ exact cache | [2026-10-09 · v2.0.0-rc3](benchmarks/glm5_flash/2026-10-09-smoke-glm-5-3-flash-nvfp4-1rtx-2spark-glm53f-nvfp4-rc3-sim5090/report.svg) |
| [GLM 5.3 Flash](docs/models/glm5_flash.md) | GLM-5.3-Flash-NVFP4 (nvfp4-g16) | 1× RTX + 2× Spark (1× RTX) | 2M tok / 8 req | 68.5 | C8: 125 | 55.6 | 72.2 | 5,412 | 1.49 s | ✓ KL 0.029 · top-1 96.4% ✓ exact cache | [2026-10-09 · v2.0.0-rc3](benchmarks/glm5_flash/2026-10-09-smoke-glm-5-3-flash-nvfp4-1rtx-2spark-glm53f-nvfp4-min-rc3/report.svg) |
| [GLM 5.3 Flash](docs/models/glm5_flash.md) | GLM-5.3-Flash-NVFP4 (nvfp4-g16) | 2× RTX + 4× Spark (2× RTX) | 2M tok / 8 req | 116 | C8: 209 | 87.2 | 144 | 6,294 | 1.29 s | ✓ KL 0.027 · top-1 96.4% ✓ exact cache | [2026-10-09 · v2.0.0-rc3](benchmarks/glm5_flash/2026-10-09-smoke-glm-5-3-flash-nvfp4-2rtx-4spark-glm53f-nvfp4-max-rc3/report.svg) |
| [GLM 5.3 Flash](docs/models/glm5_flash.md) | GLM-5.3-Flash-tr3-4bpw (exl3-k4) | simulated 5090: RTX PRO 6000 (188 SMs) capped at 31.8 GiB + 2 Spark (5090) | 1.5M tok / 8 req | 71.6 | C8: 136 | 61.9 | 91.7 | 4,877 | 1.65 s | ✓ KL 0.024 · top-1 96.6% ✓ exact cache ✓ lossless spec | [2026-10-09 · v2.0.0-rc3](benchmarks/glm5_flash/2026-10-09-smoke-glm-5-3-flash-tr3-4bpw-1rtx-2spark-glm53f-tr3-rc3-sim5090/report.svg) |
| [GLM 5.3 Flash](docs/models/glm5_flash.md) | GLM-5.3-Flash-tr3-4bpw (exl3-k4) | 1× RTX + 2× Spark (1× RTX) | 2M tok / 8 req | 73.6 | C8: 138 | 62.6 | 90.0 | 4,855 | 1.66 s | ✓ KL 0.024 · top-1 96.6% ✓ exact cache | [2026-10-09 · v2.0.0-rc3](benchmarks/glm5_flash/2026-10-09-smoke-glm-5-3-flash-tr3-4bpw-1rtx-2spark-glm53f-tr3-min-rc3/report.svg) |
| [GLM 5.3 Flash](docs/models/glm5_flash.md) | GLM-5.3-Flash-tr3-4bpw (exl3-k4) | 2× RTX + 4× Spark (2× RTX) | 2M tok / 8 req | 110 | C8: 241 | 92.1 | 154 | 6,074 | 1.33 s | ✓ KL 0.025 · top-1 96.6% ✓ exact cache | [2026-10-09 · v2.0.0-rc3](benchmarks/glm5_flash/2026-10-09-smoke-glm-5-3-flash-tr3-4bpw-2rtx-4spark-glm53f-tr3-max-rc3/report.svg) |
| [MiMo V2](docs/models/mimo_v2.md) | MiMo-V2.6-Flash-MOPD (mxfp4-g32) | simulated 5090: RTX PRO 6000 (188 SMs) capped at 31.8 GiB + 2 Spark (5090) | 956K tok / 16 req | 74.3 | C8: 167 | 44.2 | 85.3 | 5,292 | 1.52 s | ✓ KL 0.019 · top-1 95.8% ✓ exact cache ✓ lossless spec | [2026-10-09 · v2.0.0-rc3](benchmarks/mimo_v2/2026-10-09-smoke-mimo-v2-6-flash-mopd-1rtx-2spark-mimo26-flash-rc3-sim5090/report.svg) |
| [MiMo V2](docs/models/mimo_v2.md) | MiMo-V2.6-Flash-MOPD (mxfp4-g32) | 1× RTX + 2× Spark (1× RTX) | 2M tok / 8 req | 71.8 | C8: 171 | 46.3 | 90.9 | 5,482 | 1.48 s | ✓ KL 0.019 · top-1 95.8% ✓ exact cache ✓ lossless spec | [2026-10-09 · v2.0.0-rc3](benchmarks/mimo_v2/2026-10-09-smoke-mimo-v2-6-flash-mopd-1rtx-2spark-mimo26-flash-min-rc3/report.svg) |
| [MiMo V2](docs/models/mimo_v2.md) | MiMo-V2.6-Flash-MOPD (mxfp4-g32) | 2× RTX + 4× Spark (2× RTX) | 2M tok / 8 req | 130 | C8: 304 | 76.6 | 152 | 8,434 | 957 ms | ✓ KL 0.022 · top-1 95.4% ✓ exact cache ✓ lossless spec | [2026-10-09 · v2.0.0-rc3](benchmarks/mimo_v2/2026-10-09-smoke-mimo-v2-6-flash-mopd-2rtx-4spark-mimo26-flash-max-rc3/report.svg) |
| [MiMo V2](docs/models/mimo_v2.md) | MiMo-V2.6-Pro-MOPD (mxfp4-g32) | simulated 5090: RTX PRO 6000 (188 SMs) capped at 31.8 GiB + 6 Spark (5090) | — | — | — | — | — | — | — | doesn't fit: rtx0 full memory layout needs 33511016304 bytes, budget 31031138713 bytes, shortfall 2479877591 bytes | [planner report](benchmarks/mimo_v2/2026-10-09-smoke-mimo-pro-no-fit-rc3-sim5090/report.json) |
| [MiMo V2](docs/models/mimo_v2.md) | MiMo-V2.6-Pro-MOPD (mxfp4-g32) | 1× RTX + 6× Spark (1× RTX) | 2M tok / 8 req | 63.4 | C8: 128 | 34.5 | 69.9 | 3,113 | 2.58 s | ✓ KL 0.016 · top-1 95.2% ✓ exact cache ✓ lossless spec | [2026-10-09 · v2.0.0-rc3](benchmarks/mimo_v2/2026-10-09-smoke-mimo-v2-6-pro-mopd-1rtx-6spark-mimo-pro-min-rc3/report.svg) |
| [MiMo V2](docs/models/mimo_v2.md) | MiMo-V2.6-Pro-MOPD (mxfp4-g32) | 2× RTX + 6× Spark (2× RTX) | 2M tok / 8 req | 67.1 | C8: 133 | 39.6 | 79.0 | 3,536 | 2.28 s | ✓ KL 0.018 · top-1 95.9% ✓ exact cache ✓ lossless spec | [2026-10-09 · v2.0.0-rc3](benchmarks/mimo_v2/2026-10-09-smoke-mimo-v2-6-pro-mopd-2rtx-6spark-mimo-pro-max-rc3/report.svg) |
| [Qwen 3.8](docs/models/qwen4.md) | Qwen3.8-Flash-Next-EXL3-K4.25-PLE-FP8-v1 (exl3-k4+exl3-k5) | simulated 5090: RTX PRO 6000 (188 SMs) capped at 31.8 GiB + 1 Spark (5090) | 273K tok / 8 req | 115 | C8: 391 | 82.2 | 137 | 3,632 | 2.20 s | ✓ KL 0.015 · top-1 96.2% ✓ exact cache ✓ lossless spec | [2026-10-09 · v2.0.0-rc3](benchmarks/qwen4/2026-10-09-smoke-qwen3-8-flash-next-exl3-k4-25-ple-fp8-v1-1rtx-1spark-qwen38-exl3-rc3-sim5090/report.svg) |
| [Qwen 3.8](docs/models/qwen4.md) | Qwen3.8-Flash-Next-EXL3-K4.25-PLE-FP8-v1 (exl3-k4+exl3-k5) | 1× RTX (1× RTX) | 368K tok / 8 req | 212 | C8: 733 | 146 | 251 | 6,331 | 1.26 s | ✓ KL 0.015 · top-1 96.2% ✓ exact cache ✓ lossless spec | [2026-10-09 · v2.0.0-rc3](benchmarks/qwen4/2026-10-09-smoke-qwen3-8-flash-next-exl3-k4-25-ple-fp8-v1-1rtx-qwen38-exl3-min-rc3/report.svg) |
| [Qwen 3.8](docs/models/qwen4.md) | Qwen3.8-Flash-Next-EXL3-K4.25-PLE-FP8-v1 (exl3-k4+exl3-k5) | 1× RTX, no Sparks (local experts), vision on RTX0, fixed 262,144 pool (1× RTX) | 256K tok / 8 req | 229 | C8: 718 | 150 | 245 | 6,393 | 1.25 s | ✓ KL 0.015 · top-1 96.2% ✓ exact cache ✓ lossless spec | [2026-10-09 · v2.0.0-rc3](benchmarks/qwen4/2026-10-09-smoke-qwen3-8-flash-next-exl3-k4-25-ple-fp8-v1-1rtx-qwen38-exl3-nospark-vision-rc3/report.svg) |
| [Qwen 3.8](docs/models/qwen4.md) | Qwen3.8-Flash-Next-NVFP4 (nvfp4-g16) | simulated 5090: RTX PRO 6000 (188 SMs) capped at 31.8 GiB + 2 Spark (5090) | 238K tok / 8 req | 130 | C8: 373 | 97.1 | 160 | 5,344 | 1.50 s | ✓ KL 0.037 · top-1 94.8% ✓ exact cache ✓ lossless spec | [2026-10-09 · v2.0.0-rc3](benchmarks/qwen4/2026-10-09-smoke-qwen3-8-flash-next-nvfp4-1rtx-2spark-qwen38-nvfp4-rc3-sim5090/report.svg) |
| [Qwen 3.8](docs/models/qwen4.md) | Qwen3.8-Flash-Next-NVFP4 (nvfp4-g16) | 1× RTX (1× RTX) | 213K tok / 8 req | 224 | C8: 693 | 153 | 257 | 8,700 | 918 ms | ✓ KL 0.032 · top-1 94.3% ✓ exact cache ✓ lossless spec | [2026-10-09 · v2.0.0-rc3](benchmarks/qwen4/2026-10-09-smoke-qwen3-8-flash-next-nvfp4-1rtx-qwen38-nvfp4-min-rc3/report.svg) |

tok/s; C1 decode and concurrent code aggregate with thinking off (up to C8, clamped to server admission), 8K prefill cold. Quality: logit fidelity against the family golden reference, prefix-cache restore exactness, lossless speculation. Simulated 5090 reports cap RTX PRO 6000 memory only; SM count (188 vs 170), L2, clocks and power are not emulated. Their speed is indicative and likely optimistic.

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
- **GLM Flash tr3** vision needs `CHAT_TEMPLATE_FROM=zai-org/GLM-5.3-Flash`.

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
supports_standalone_web_search = true
# Optional: the served model's real context window and output limit, so
# Codex compacts at the right point instead of using its built-in numbers.
model_catalog_url = "http://HOST:PORT/v1/codex/models.json"
```

Codex's responses-lite mode uses the client-executed `web.run` extension,
not a hosted Responses tool. `supports_standalone_web_search = true` enables
its authenticated `POST /v1/alpha/search` calls. Queries work with either search
provider; Exa also supports page `open`, and `find` searches opened text cached
for the session. `time` works locally; image search, click, screenshot, finance,
weather and sports return explicit unsupported tool output. Cached-mode searches
use the provider's index; page opens only reuse already-opened session pages,
never fetching uncached pages. Reference/page caches
are bounded and expire after an hour of inactivity; reopen URLs if refs expire.

**Realtime:** any client that accepts a custom URL can connect to
`ws://HOST:PORT/v1/realtime?model=...`. These run headless against it:
- the openai-python and openai-node SDKs (Node requires `wss://`);
- Agents JS/Python;
- Pipecat.

Runners are in `scripts/gateway/realtime-clients/`.

## Working on it

[`AGENTS.md`](AGENTS.md) is the standing guide for agents and collaborators
working on CuteAFD. [`PLAN.md`](PLAN.md) is the roadmap.
[WIP hardware cards](docs/wip-cards.md) runs candidate builds through the
release card matrix.

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
