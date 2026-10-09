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
- Exact prefix caching for agentic work: the deepest cached snapshot that
  prefixes a request is restored byte-identical, not approximated.
- Own your intelligence: your weights, your hardware, your rate limits (none),
  agentic coding at full speed on a machine you control, not a shared tenant.

## v2.0.0 changes since v1.0.0

- Bundled image input across shipped vision-capable families, V4.1 vision on Spark rank 0, and qualified MiMo audio enabled by default with its tower included in release images.
- One SM120 image for RTX PRO 6000 and RTX 5090; runtime SM/grid sizing, GPU/RDMA probing and 32 GB memory admission. The automatic KV target is 2M tokens on PRO cards and 1M on small cards, subject to family admission.
- Hugh Madden's GLM Flash ports #14-#24: pooled prefix marks and pinned host tier, lane/workspace and graph accounting, plus scoped compact-index, BF16-state and GB10 EXL3 schedule opt-ins. PR #25 is excluded.
- MiMo V2.6 Flash MOPD, bounded serving graphs, fidelity tiers (quick/standard/full) and published family goldens.
- Release-smoke cards add C8 aggregate code throughput alongside C1 and cold ~8K prefill. The grid is 5090 | 1x RTX | 2x RTX; Qwen's unsupported two-GPU split is n/a.
- Concurrent coordinator/Spark release builds, a published pinned toolchain-only dev image, and persistent architecture-specific Cargo/JIT/compiler caches.

The 5090 column is memory-only simulation unless explicitly marked real:
**simulated 5090: RTX PRO 6000 (188 SMs) capped at 31.8 GiB**.
SM count (170 vs 188), L2, clocks and power are not emulated; speed is indicative
and likely optimistic. Real 5090 cards replace current cells; simulations remain
in history. Planner-only no-fit cells link the reason and contain no performance.
Release smoke is not a full multimodal or full-context qualification. Family
Known limits retain unresolved numerical, stall and historical heap caveats.
Memory fit and executable kernel support are separate: V4.1 NVFP4 fits TP3 by
memory but requires the launcher's TP4 path (TP3 packages are native-only);
GLM/GLM Flash checkpoints support 1,048,576 context tokens and Qwen supports
262,144, while this image's compiled index extent for those families is 131,072.
Each family's Known limits and affected current-cell captions label the distinction.

These rc1 images and cards measure frozen source `29bc9e04`. Landing in the
next candidate: Hugh's #25 opt-in 128-row decode/verify, #26 GLM Flash Spark
transport warm-up and #27 prefix-mark count edge fix; they merged later and
are not included in rc1. rc1 remains a candidate for Hugh's real 5090 column,
not the final v2.0.0 target: the agentic context floor is 256K, above this image's
131,072-token index extent. rc2 with a 1M compiled extent is the v2.0.0 target.
V4's requested 1M context is retained as launch provenance; its effective runtime
context is 131,072, and the corrected cards label that manifest-enforced limit.

### Blockers Before v2.0.0

- Generic-family launches default to only 8,192 tokens (`scripts/launch/run-family.sh:279`), despite the planner's checkpoint-full context. Corrected rc1 cards explicitly set checkpoint context where supported; GLM/GLM Flash/Qwen are capped at the image's compiled 131,072-token extent. The later work/p0 fix `437d0a76` clamps the default to the compiled cap and logs it (131K, not 8K); it is not in rc1. rc2 targets a 1M compiled extent to meet the 256K agentic floor; scratch and image-size effects still need measurement. These cards do not qualify rc1's broken user default.
- MiMo Pro's one-RTX minimum fails automatic-pool admission by 65,484,228 B: the automatic pool fills the 97% ceiling before the 64 MiB Spark intake probe is charged (`rust/crates/cuteafd-daemon/src/families/mimo_v2/admission.rs:460`). The configured minimum uses a 1,048,576-token pool and preserves full context. Fix automatic sizing after reserving max(intake probe, small-card floor) as a startup peak.

The GLM Flash tr3 template needs an explicit vendor-template override for vision;
V4.1 NVFP4 fidelity is unqualified because its published reference config is
missing. V4 Pro EXL3 K2 straddles the 0.06 KL gate: minimum 0.0605 fails,
maximum 0.0596 passes. The minimum card stays FAIL and the threshold is unchanged;
this is not an established regression. These findings are recorded in family
Known limits, separately from measured performance and the final-release blockers.

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

## Models

Basic benchmark profile per family on its natural-minimum (1× RTX + fewest
Sparks) and maximum (2× RTX + 4 or 6 Sparks) hardware. Other reports:
[`benchmarks/`](benchmarks/README.md).

The v2.0.0-rc1 natural-minimum/maximum matrix contains 26 Release-smoke cards:
23 pass independently checked quick fidelity and exact-cache gates; two V4.1
NVFP4 cards have unsupported/unqualified fidelity; V4 Pro EXL3 K2 minimum
fails KL (0.0605 against 0.06). All three required publication spots pass.
These are C1, warmed C8 aggregate and cold ~8K measurements, not C16,
full-context or multimodal qualification. Configured context/template/pool
corrections do not qualify the broken defaults described above. The simulated
5090 matrix is pending; four planner-only no-fit reports contain no performance.

MiMo V2.6 Flash MOPD replaces the legacy Flash current rows. Its historical
[text-only bring-up qualification](docs/models/mimo_v2.md#flash-mopd-qualification-2026-10-05)
is separate from these rc1 measurements and retains its original conditions.
See the [v2 release scope](PLAN.md#release-v2-scope-decided-2026-10-05) and family Known limits.

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
   workspaces and Spark ranks. V4 Flash/Pro workspace formulas use the
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
