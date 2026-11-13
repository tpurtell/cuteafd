# CUTEAFD example configurations

These files are **opt-in, standalone `--config` files**. They overlay the
launcher defaults in `scripts/lib/release-common.sh` (DeepSeek V4.1) and
`scripts/launch/run-family.sh` (every other family); every key they do not
name keeps its default. A key the launched family never reads is a hard error
(`unused configuration key`, exit 2). Every key, its default, allowed values
and the families that accept it are in [Options](../../docs/options.md).
The files are never selected automatically and they do not change
`cuteafd.config`, the release images or the default serving selection.

Select one explicitly:

```bash
./run.sh --config examples/configs/glm53-exl3-min.config --dry-run
```

## Starting configs per family

One file per v2.0.0 release card: `<model>-<quant>-<min|max>.config`, on the
natural minimum (1 RTX + the fewest Sparks it fits) and the maximum
(2 RTX + 4 or 6 Sparks). Each names the checkpoint revision, speculator,
drafter, KV pool, media and expert placement the card ran with, so starting
from one reproduces the published card, except that V4.1 MXFP4 max now uses
six Sparks and keeps its four-Spark release layout as a reference.
Replace the `REPLACE-ME-spark-N` hosts
and the `192.0.2.N` (RFC 5737 documentation) RoCE addresses with your own,
download the checkpoint and its drafter, then `--dry-run` and `--restart`.
Cards and quality findings are in the main README and each family doc.

| Model · quant | Min | Max | Speculator | `POOL_TOKENS` |
| --- | --- | --- | --- | --- |
| [DeepSeek V4.1 Flash · MXFP4](../../docs/models/deepseek_v41.md) | [1 RTX + 3 Spark (TP3)](deepseek-v41-mxfp4-min.config) | [2 RTX + 6 Spark](deepseek-v41-mxfp4-max.config) | `DSPARK=on` | `auto` (exception, below) |
| DeepSeek V4.1 Flash · MXFP4, release-only FP8/FP8 replies | - | [2 RTX + 6 Spark](deepseek-v41-mxfp4-max-fp8-reply.config) | `dspark` | `auto` |
| DeepSeek V4.1 Flash · MXFP4, release-only NVFP4/NVFP4 replies | - | [2 RTX + 6 Spark](deepseek-v41-mxfp4-max-nvfp4-reply.config) | `dspark` | `auto` |
| DeepSeek V4.1 Flash · StridedPLE (byte-identical FP8 Engram) | [1 RTX + 3 Spark](v41-flash-strided-min.config) | [2 RTX + 6 Spark](v41-flash-strided-max.config) | `dspark` | `auto` |
| [DeepSeek V4 Flash · MXFP4](../../docs/models/deepseek_v4.md) | [1 RTX + 2 Spark](deepseek-v4-flash-mxfp4-min.config) | [2 RTX + 4 Spark](deepseek-v4-flash-mxfp4-max.config) | `dspark` | `auto` |
| [DeepSeek V4 Pro · EXL3 K2](../../docs/models/deepseek_v4.md) | [1 RTX + 4 Spark](deepseek-v4-pro-exl3-min.config) | [2 RTX + 6 Spark](deepseek-v4-pro-exl3-max.config) | `dspark` | `auto` |
| [GLM 5.3 · EXL3 K4](../../docs/models/glm5.md) | [1 RTX + 4 Spark](glm53-exl3-min.config) | [2 RTX + 6 Spark](glm53-exl3-max.config) | `dflash2` `incoai/GLM-5.3-DFlash2` | `auto` |
| [GLM 5.3 · NVFP4](../../docs/models/glm5.md) | [1 RTX + 4 Spark](glm53-nvfp4-min.config) | [2 RTX + 6 Spark](glm53-nvfp4-max.config) | `dflash2` `incoai/GLM-5.3-DFlash2` | `auto` |
| [GLM 5.3 Flash · FP8](../../docs/models/glm5_flash.md) | [1 RTX + 4 Spark](glm53-flash-fp8-min.config) | [2 RTX + 4 Spark](glm53-flash-fp8-max.config) | `dflash2` `incoai/GLM-5.3-Flash-DFlash2` | `auto` |
| [GLM 5.3 Flash · EXL3 K3.25](../../docs/models/glm5_flash.md) | [1 RTX + 2 Spark](glm53-flash-exl3-min.config) | [2 RTX + 4 Spark](glm53-flash-exl3-max.config) | `dflash2` `incoai/GLM-5.3-Flash-DFlash2` | `auto` |
| [GLM 5.3 Flash · EXL3 K4 + Core K6](../../docs/models/glm5_flash.md) | [1 RTX + 2 Spark](glm53-flash-exl3-k6-min.config) | [2 RTX, 0 Spark](glm53-flash-exl3-k6-max.config) | `dflash2` `incoai/GLM-5.3-Flash-DFlash2` | `auto` |
| [GLM 5.3 Flash · native EXLR8 K4](../../docs/models/glm5_flash.md) | [1 RTX + 1 Spark](glm53-flash-exlr8-k4-min.config) | [2 RTX, 0 Spark](glm53-flash-exlr8-k4-max.config) | `dflash2` `incoai/GLM-5.3-Flash-DFlash2` | `auto` |
| [GLM 5.3 Flash · NVFP4](../../docs/models/glm5_flash.md) | [1 RTX + 2 Spark](glm53-flash-nvfp4-min.config) | [2 RTX + 4 Spark](glm53-flash-nvfp4-max.config) | `dflash2` `incoai/GLM-5.3-Flash-DFlash2` | `auto` |
| [GLM 5.3 Flash · EXL3 K4 MC, original core](../../docs/models/glm5_flash.md) | [1 RTX + 2 Spark](glm53-flash-exl3-k4-mc-min.config) | [2 RTX, 0 Spark](glm53-flash-exl3-k4-mc-max.config) | `dflash2` `incoai/GLM-5.3-Flash-DFlash2` | `auto` |
| [MiMo V2.6 Flash MOPD · MXFP4](../../docs/models/mimo_v2.md) | [1 RTX + 2 Spark](mimo-v26-flash-mxfp4-min.config) | [2 RTX + 4 Spark](mimo-v26-flash-mxfp4-max.config) | `dflash2` (bundled `dflash/`) | `auto` |
| [MiMo V2.6 Pro MOPD · MXFP4](../../docs/models/mimo_v2.md) | [1 RTX + 6 Spark](mimo-v26-pro-mxfp4-min.config) | [2 RTX + 6 Spark](mimo-v26-pro-mxfp4-max.config) | `dflash2` (bundled `dflash/`) | `auto` |
| [Qwen 3.8 Flash Next · EXL3 K4.25](../../docs/models/qwen4.md) | [1 RTX, local experts](qwen38-exl3-min.config) | n/a (fits one RTX) | `mtp`, depth 3 | `auto` |
| [Qwen 3.8 Flash Next · NVFP4](../../docs/models/qwen4.md) | [1 RTX, local experts](qwen38-nvfp4-min.config) | n/a (fits one RTX) | `mtp`, depth 3 | `auto` |

⚠ The NVFP4 V4.1 cards fail the calibrated confident-top-1 fidelity check;
prefer the official MXFP4 release.

The speculator and pool keys repeat the launcher's bare defaults for the
family: dropping them gives the same launch, and
`scripts/tests/test_example_configs.py` keeps the two in step. The one named
exception is V4.1's `POOL_TOKENS=auto`: its cards ran with it, while V4.1's
release loader leaves the key empty when absent (the launch then resolves
auto). V4.1 launches through `run-family.sh` like every family (#82); roadmap
step S5 unifies this default with the other families'.

## Newly published checkpoints

These pinned recipes need a matching WIP build and have no benchmark cards yet.
Layouts, speculators and image keys match their earlier Core-K6 examples; this
is not a claim that the restored original core has the same memory or speed.
`CORE_VIEW=original` is explicit in each config. Mistral is a plain checkpoint;
GLM Flash, GLM 5.3 and Step-5 MCs also offer `k8-compact` and `k6-compact`,
but not whole-core `fp8-compact`. Their standalone GLM/Step EXL3 config paths
remain archived recipes; current MC cards use the dedicated files here.

| Model | Config | Layout |
| --- | --- | --- |
| [Mistral Large 4, EXL3 K4 + original mixed FP8/BF16 core](../../docs/models/mistral_large4.md) | [min](mistral-large4-exl3-k4-min.config) · [max](mistral-large4-exl3-k4-max.config) | 1 RTX + 5 Spark / 2 RTX + 6 Spark; bundled dSpark |
| [GLM 5.3 Flash, EXL3 K4 MC](../../docs/models/glm5_flash.md) | [min](glm53-flash-exl3-k4-mc-min.config) · [max](glm53-flash-exl3-k4-mc-max.config) | 1 RTX + 2 Spark / 2 RTX, no Sparks; DFlash2 |
| [GLM 5.3, EXL3 K4 MC](../../docs/models/glm5.md) | [min](glm53-exl3-k4-mc-min.config) · [max](glm53-exl3-k4-mc-max.config) | 1 RTX + 4 Spark / 2 RTX + 6 Spark; DFlash2 |
| [Step-5-Preview, EXL3 K4 MC](../../docs/models/step5.md) | [min](step5-exl3-k4-mc-min.config) · [max](step5-exl3-k4-mc-max.config) | 1 RTX + 3 Spark / 2 RTX + 6 Spark; native MTP3 |

## Families under qualification

These have no release card yet. Use a `./wip.sh` slot containing the path
under qualification (`--wip SLOT`); the image keys remain the promoted pair
for consistency, not a support claim. Speculator and placement choices are
listed per recipe. New families without a release serve command are refused
outside a matching WIP slot.

| Model · quant | Config | Layout |
| --- | --- | --- |
| GLM 5.3 · EXL3 K4 + native K6 core | [min](glm53-exl3-k6core-min.config) · [max](glm53-exl3-k6core-max.config) | 1 RTX + 4 Spark / 2 RTX + 6 Spark, DFlash2, pool-first |
| GLM 5.3 Flash · EXL3 K4 + native K6 core | [min](glm53f-exl3-k4-core-k6-min.config) · [max](glm53f-exl3-k4-core-k6-max.config) | 1 RTX + 2 Spark / 2 RTX, no Sparks; DFlash2, admitted KV pool |
| Kimi K3 · EXL3 K2 (#273) | [max](kimi-k3-exl3-max.config) | 2 RTX (TP2 attention) + 6 Spark (TP6 experts), 5 routed layers on the RTX |
| Kimi K3 · EXL3 K2 + K6 core (#1178) | [min](kimi-k3-exl3-k6-min.config) · [max](kimi-k3-exl3-k6-max.config) | 1 RTX + 6 Spark (110 GiB Spark budget) / 2 RTX + 6 Spark; native K6 core, automatic pool and placement, DFlash2 |
| Step-5-Preview · FP8 (#274) | [max](step5-fp8-max.config) | 2 RTX + 6 Spark |
| Step-5-Preview · NVFP4 (#274) | [min](step5-nvfp4-min.config) | 1 RTX + 4 Spark |
| Muse Spark · EXL3 K4 + Core K6 (#923) | [min](muse-spark-exl3-min.config) · [max](muse-spark-exl3-max.config) | 1 RTX + 4 Spark text qualified / 2 RTX + 6 Spark planned; pinned NVIDIA DFlash2, pool-first; media off |
| Step-5-Preview · EXL3 K4 + Core K6 (#911) | [min](step5-exl3-k6core-min.config) · [max](step5-exl3-k6core-max.config) | 1 RTX + 3 Spark · 2 RTX + 6 Spark |

## Spark tensor parallelism

Every Spark rank holds a tensor-parallel slice of every routed expert of its
layers; there are no replicated expert groups. The planner picks the Spark
count (`SPARK_COUNT` or `SPARK_HOSTS` only bounds it) and every rank count in
a family's package range runs on the one canonical Spark wire.

The Spark image is universal: it carries every native expert TP role its
build exported and advertises them in the `io.cuteafd.spark_tp_roles` label.
`run.sh` requires the role the planned deployment needs on every rank and
refuses before any service is stopped or replaced.

## DeepSeek V4.1 StridedPLE

StridedPLE packs each Engram row as 256 FP8 bytes followed by eight scale bytes.
`MAPPED_SCALE_PRELOAD_GIB` preloads both complete interleaved tables, across all
parts, when their full payload fits the cap and available host memory. Smaller
caps leave them mapped: there is no separate scale-only preload. `ENGRAM_WARM`
warms the complete mapped interleaved table. The original
checkpoint and its main release card remain unchanged.
