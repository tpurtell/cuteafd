# CUTEAFD example configurations

These files are **opt-in, standalone `--config` files**. They overlay the
launcher defaults in `scripts/lib/release-common.sh` (DeepSeek V4.1) and
`scripts/launch/run-family.sh` (every other family); every key they do not
name keeps its default. A key the launched family never reads is a hard error
(`unused configuration key`, exit 2). `KV_CACHE` is KV precision for Qwen
(`bf16|fp8`, default `fp8`, forwarded as `--kv-cache`) and MiMo (`int8|bf16`;
unset omits the flag and the daemon default `int8` applies). Other families
have no KV-precision choice, so they reject `KV_CACHE`. The files are never
selected automatically and they do not change `cuteafd.config`, the release
images or the default serving selection.

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
| DeepSeek V4.1 Flash · StridedPLE (byte-identical FP8 Engram) | [1 RTX + 3 Spark](v41-flash-strided-min.config) | [2 RTX + 6 Spark](v41-flash-strided-max.config) | `dspark` | `auto` |
| [DeepSeek V4 Flash · MXFP4](../../docs/models/deepseek_v4.md) | [1 RTX + 2 Spark](deepseek-v4-flash-mxfp4-min.config) | [2 RTX + 4 Spark](deepseek-v4-flash-mxfp4-max.config) | `dspark` | `auto` |
| [DeepSeek V4 Pro · EXL3 K2](../../docs/models/deepseek_v4.md) | [1 RTX + 4 Spark](deepseek-v4-pro-exl3-min.config) | [2 RTX + 6 Spark](deepseek-v4-pro-exl3-max.config) | `dspark` | `auto` |
| [GLM 5.3 · EXL3 K4](../../docs/models/glm5.md) | [1 RTX + 4 Spark](glm53-exl3-min.config) | [2 RTX + 6 Spark](glm53-exl3-max.config) | `dflash2` `incoai/GLM-5.3-DFlash2` | `auto` |
| [GLM 5.3 · NVFP4](../../docs/models/glm5.md) | [1 RTX + 4 Spark](glm53-nvfp4-min.config) | [2 RTX + 6 Spark](glm53-nvfp4-max.config) | `dflash2` `incoai/GLM-5.3-DFlash2` | `auto` |
| [GLM 5.3 Flash · FP8](../../docs/models/glm5_flash.md) | [1 RTX + 4 Spark](glm53-flash-fp8-min.config) | [2 RTX + 4 Spark](glm53-flash-fp8-max.config) | `dflash2` `incoai/GLM-5.3-Flash-DFlash2` | `auto` |
| [GLM 5.3 Flash · EXL3 K3.25](../../docs/models/glm5_flash.md) | [1 RTX + 2 Spark](glm53-flash-exl3-min.config) | [2 RTX + 4 Spark](glm53-flash-exl3-max.config) | `dflash2` `incoai/GLM-5.3-Flash-DFlash2` | `auto` |
| [GLM 5.3 Flash · EXL3 K4 + Core K6](../../docs/models/glm5_flash.md) | [1 RTX + 2 Spark](glm53-flash-exl3-k6-min.config) | [2 RTX, 0 Spark](glm53-flash-exl3-k6-max.config) | `dflash2` `incoai/GLM-5.3-Flash-DFlash2` | `auto` |
| [GLM 5.3 Flash · NVFP4](../../docs/models/glm5_flash.md) | [1 RTX + 2 Spark](glm53-flash-nvfp4-min.config) | [2 RTX + 4 Spark](glm53-flash-nvfp4-max.config) | `dflash2` `incoai/GLM-5.3-Flash-DFlash2` | `auto` |
| [GLM 5.3 Flash · tr3 4bpw](../../docs/models/glm5_flash.md) | [1 RTX + 2 Spark](glm53-flash-tr3-min.config) | [2 RTX + 4 Spark](glm53-flash-tr3-max.config) | `dflash2` `incoai/GLM-5.3-Flash-DFlash2` | `auto` |
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
| Kimi K3 · EXL3 K2 + K6 core (#273) | [max](kimi-k3-exl3-k6-max.config) | 2 RTX (TP2 attention) + 6 Spark (TP6 experts), 12 routed layers on the RTX, full 1M context, DFlash2 |
| Step-5-Preview · FP8 (#274) | [max](step5-fp8-max.config) | 2 RTX + 6 Spark |
| Step-5-Preview · NVFP4 (#274) | [min](step5-nvfp4-min.config) | 1 RTX + 4 Spark |
| Step-5-Preview · EXL3 K4 + Core K6 (#911) | [min](step5-exl3-k6core-min.config) · [max](step5-exl3-k6core-max.config) | 1 RTX + 3 Spark · 2 RTX + 6 Spark |

## TP x EP layout examples (DeepSeek V4.1)

The remaining files are V4.1 Spark topology experiments.

## Topology keys

| Key | Meaning |
| --- | --- |
| `SPARK_COUNT` | physical Spark ranks = `SPARK_TP * SPARK_EP` |
| `SPARK_TP` | tensor-parallel degree inside one replicated expert group |
| `SPARK_EP` | number of replicated expert groups (each group holds every expert) |

`SPARK_TP` and `SPARK_EP` are optional and **all-or-none**. When both are
absent the planner chooses the checkpoint's canonical topology. Native V4.1
three- and six-rank deployments select the ownership wire with both keys
(`SPARK_TP=3 SPARK_EP=1` or `SPARK_TP=6 SPARK_EP=1`).
The rank map is group-major:

```
group   = global_rank / SPARK_TP
tp_rank = global_rank % SPARK_TP
```

Approved native official layouts:

| File | Hardware | Topology | Spark role | Status |
| --- | --- | --- | --- | --- |
| `tp4ep1-explicit-native.config` | 2 RTX + 4 Spark | TP4 x EP1 = 4 | none (legacy shard) | explicit form of the default; control arm |
| `tp2ep2-native.config` | 2 RTX + 4 Spark | TP2 x EP2 = 4 | `tp2` | experiment completed; quality not accepted; release not qualified |
| `tp3ep1-native.config` | 1 RTX + 3 Spark | TP3 x EP1 = 3 | `tp3` | v10 TP3 profile; 5 RTX-local / 35 remote via the explicit-topology placement handoff; daemon support shipped since v9; this layout never qualified — see below |
| `tp3ep2-native.config` | 1 RTX + 6 Spark | TP3 x EP2 = 6 | `tp3` | experiment completed (all 40 remote); quality not accepted; release not qualified |
| `tp2ep3-native.config` | 2 RTX + 6 Spark | TP2 x EP3 = 6 | `tp2` | experiment completed; quality not accepted; release not qualified |
| `tp6ep1-native.config` | 2 RTX + 6 Spark | TP6 x EP1 = 6 | `tp6` | packaged; bounded final-image functional checks passed on 1 and 2 RTX; canonical six-rank qualifier not run |

All six native configurations have launcher support, including the dual-TP3
arms; the five six-rank/four-rank configurations actually run are recorded
under `runs/tp-ep-preflight/`.
These example files are **illustrative launch configurations, not the measured
artifacts**. See [docs/tp-ep-configuration.md](../../docs/tp-ep-configuration.md)
for the current results and the launcher limits (entry points differ; the
candidate launcher requires explicit topology and is single-rail A-only). No
release-support promise is made.

## v10 TP3 profiles

`tp3ep1-native.config` selects explicit native `TP3xEP1`, one unreplicated
group of three ranks. Packaging is not qualification: the file carries no
memory, correctness, performance or readiness claim — see the
[native TP3 status report](../../docs/release-v10-tp3-official-1x-3spark.md)
(its own banner governs). DeepSeek V4/V4.1 Flash EXL3 is unsupported.

The native profile pins `RTX_EXPERT_LAYERS=5`: the RTX holds the first five
routed layers and each Spark starts at layer 5 to hold the other 35. The coordinator derives the exact per-rank assignment from checkpoint headers
and sends typed admission before workers load; no launcher boundary file is used. The
5-local / 35-remote single-RTX placement has prior TP6xEP1 functional evidence
on the final v9 images and matched the measured v8 baseline dispatch; that is
TP6 placement evidence, not TP3 qualification. Remote TP3 weight is
`35 * 2,406,481,920 = 84,226,867,200 B`, leaving `24,892,452,864 B` under the
`109,119,320,064 B` floor — admission arithmetic only, not a runtime fit.

The `v1.0.0` pair is named by `cuteafd.config`, so a plain `./build.sh`
derives the `v1.0.0` tag from it. Pull the coordinator image on the RTX host
and the Spark image on every selected rank before `./run.sh`; the launcher
does not pull images.

Explicit topologies require native experts; NVFP4 also supports pure TP3/4/6.
DeepSeek V4/V4.1 Flash EXL3 checkpoints are rejected before any service change.

## Images and roles

Every example names the **same promoted release pair** that `cuteafd.config`
names. The Spark image is universal: it carries the default TP4 shard plus
the `tp2`, `tp3` and `tp6` expert TP roles and advertises them as
`io.cuteafd.spark_tp_roles=tp2;tp3;tp6` (see
[docs/release-v10-notes.md](../../docs/release-v10-notes.md)). One published pair
therefore serves every approved topology, and `run.sh` is what selects the mode:
it derives the needed role from `SPARK_TP`, requires that role in the image label
of **every** rank, and refuses before any service is stopped or replaced. A
topology that needs a role the image lacks is rejected even though the tag
resolves; the legacy default TP4 path never probes the label.

Do not pin a per-topology tag such as `cuteafd-coordinator:tp2ep3-candidate`.
`build.sh` takes the tag from the config it is given, so a name like that exists
only on a host where that exact file was built, and `run.sh` fails its image
check on every other host — the launcher cannot infer a tag from a topology.

Packaging is not qualification. The role check is a compatibility preflight: it
proves the image carries the shard family this topology asks for, nothing more.
Neither it nor a passing dry-run accepts the layout on quality, memory or
throughput; the per-file status above and
[docs/tp-ep-configuration.md](../../docs/tp-ep-configuration.md) carry that line.

To run a six-rank example, the Spark image must exist on all six ranks: a default
`./build.sh` builds and distributes to the four active ranks, while building with
one of the six-rank `--config` files selects all six. Otherwise pull the Spark
image on `rhea` and `moa` too. The coordinator image is x86_64 and runs only on
the RTX host.

## Fifth and sixth Sparks

The six-Spark files use `rhea` (global rank 4) and `moa` (rank 5), connected
2026-09-20; see [docs/cluster-hosts.md](../../docs/cluster-hosts.md). Under the
current permission they are ordinary benchmark/development hosts and are **not**
restricted to a single six-Spark serving set. These examples name a secondary
rail (`LANE_B`) on an isolated subnet with matching host numbers, but the release
transport still dials `LANE_A` and the candidate launcher is single-rail A-only
by design; `LANE_B` is unused unless a launch explicitly selects it. The six-rank
experiments have completed but are **not accepted as quality results and not
release-qualified**; no memory, correctness, performance or readiness claim is
attached.

## Admission

`run.sh` prints a Spark *weight-only* admission line derived from the actual
resolved RTX/Spark boundary. It is **not** a launch-feasibility claim: workspace,
load staging and runtime headroom are only known after the expert service
reports them. Weight-only overflow is rejected before any service change. See
[docs/tp-ep-configuration.md](../../docs/tp-ep-configuration.md).

The TP×EP overlays set `SPARK_DEVICE_BUDGET_BYTES=109119320064`, the tested
fleet-wide floor (minimum of `MemTotal - 20 GiB` across the six hosts); the
`tp4ep1` control carries no override and keeps the 100 GiB default fallback.

StridedPLE packs each Engram row as 256 FP8 bytes followed by eight scale bytes.
`MAPPED_SCALE_PRELOAD_GIB` preloads both complete interleaved tables, across all
parts, when their full payload fits the cap and available host memory. Smaller
caps leave them mapped: there is no separate scale-only preload. `ENGRAM_WARM`
warms the complete mapped interleaved table. The original
checkpoint and its main release card remain unchanged.
