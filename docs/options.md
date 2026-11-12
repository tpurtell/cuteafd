# Options

Every user-facing launch key and CLI option, grouped by area, with defaults and
the values each launch path accepts. The family columns in the last section are
tested against the launcher (`scripts/tests/test_options_doc.py`).

Release builds (`./build.sh`) and publication share the shipping matrix reported
by `scripts/build/release-scope.py default-matrix`. `CUTEAFD_RELEASE_MATRIX`
overrides that default; `--scope` selects families or cards for development builds,
while publication requires `all` and complete image inventories.

## Optional performance levers

Optional levers that are off by default and trade precision or host resources
for speed or memory at serve time. Every entry starts from the checkpoint's
native format or the most conservative behaviour; the effect is measured on the
stated hardware or marked unmeasured. Debug, profiling and capacity knobs are
documented in their own sections, not here. A PR that adds an optional,
default-off performance lever adds its row here in the same PR (AGENTS.md).

| Lever | Families | Enable | Effect | Cost |
| --- | --- | --- | --- | --- |
| `L2_PREFETCH` | V4.1, MiMo, Qwen4 | `L2_PREFETCH=auto` or a MiB count (unset = off; GLM 5.3 and GLM Flash already default it to `auto`) | Pulls the next layer's weights into L2 while a one-lane decode step waits on its Spark exchange. V4.1 1× RTX 5090 + 4 Sparks: every C1 cell +1.7–3.9%, C4/C8 and 8K prefill unchanged (#623); GLM 5.3 weighted +4.4%, GLM Flash C1 +6–15% (2026-09-30). | None in values (outputs byte-identical); only extra L2 traffic. |
| `QWEN_FP8_DECODE` | Qwen4 | `QWEN_FP8_DECODE=on` (unset = checkpoint BF16) | Holds the GDN/attention projections as one resident E4M3 copy; with `QWEN_FP8_HEAD=on`, 1× RTX code C1 222→247 tok/s, C4 441→489, 8K prefill 6100→6177 (2026-10-03). | Converts target projections to FP8 at load: golden KL 0.036→0.046, top-1 88.5%→86.7%; fails the KL gate, so opt-in only. |
| `CORE_VIEW` | GLM 5.3 / ML4, GLM Flash, Kimi K3, Step-5, Muse | `CORE_VIEW=k8-compact`, `k6-compact` or qualified `fp8-compact` (unset = original) | Selects a stored complete MC core, or explicit load-time FP8 within the family's documented runtime scope. The MC repositories listed below offer K8/K6; selected metadata is authoritative. Unmeasured; withdrawn HQA measurements do not qualify MC. | K8/K6 quantize target weights. Kimi K3.1 MC and GLM Flash MC refuse FP8; Muse FP8 changes q/k/v/o only. Plain ML4 accepts original only: native E4M3 q/o, dense/shared FFNs and index query; BF16 kv_a/kv_b/head/router/embeddings. |
| `CUTEAFD_GLM_BF16` | GLM 5.3 / ML4 | `CUTEAFD_GLM_BF16=fp8` (unset = native) | GLM 5.3 converts target BF16 MLA, indexer and dense/shared FFNs to block FP8. Unmeasured; no conversion of ML4's native mixed core. | Target-weight quantization; drafter and KV precision are independent. |
| `QWEN_FP8_HEAD` | Qwen4 | `QWEN_FP8_HEAD=on` (unset = BF16) | Converts the shared target/MTP head to one E4M3 backing. Historical combined measurement is in the decode row; standalone effect unmeasured. | Changes target logits; the draft reads the same head, not a separate draft-only FP8 copy. |
| `MIMO_FP8_HEAD`, `MIMO_FP8_O_PROJ` | MiMo | Set either to `on` (unset/auto = checkpoint) | Explicitly converts a BF16 target head or O projection; native FP8 stays FP8. Unmeasured. | Target-weight quantization, independent of drafter precision. |
| `GLM5_FLASH_FP8_HEAD` | GLM Flash | `GLM5_FLASH_FP8_HEAD=on` (unset/auto = checkpoint BF16 on TP1 and TP2) | Converts the shared target/draft head to E4M3 on either topology; unmeasured. `packed` is a separate exact BF16 option on one GPU. | Changes target logits; native trellis/EXLR8 heads ignore this setting. |
| `GLM5_FLASH_KDA_FP8` | GLM Flash | `GLM5_FLASH_KDA_FP8=row128` or `channel` (unset/auto = off) | Converts BF16 KDA projections to one resident E4M3 copy. Prior 1 RTX + 2 Sparks, with FP8 head and drafter: C4 code 113.8 to 131.8 tok/s; MC original unmeasured. | Prior top-1 89.1% to 85.7%; six-row verify/decode KL 0.0056 to 0.0121 nat. Does not change KV dtype. |
| `GLM5_FLASH_FP8_PREFILL` | GLM Flash | Add `kda-in` / `kda-o` to the default `mla,ffn` list; requires opted-in KDA FP8 | Adds W8A8 KDA prefill passes over already-FP8 weights. Unmeasured. Default MLA/shared-FFN passes likewise apply only to vendor FP8 or explicitly converted weights; they are a no-op on BF16 original/MC original operands. | FP8 activation quantization; this setting never converts BF16 target weights. |
| `SPARK_PREFILL_REPLY_FORMAT` | V4.1, V4, GLM 5.3 / ML4, GLM Flash, MiMo, Qwen4, Kimi K3, Step-5, Muse | `SPARK_PREFILL_REPLY_FORMAT=fp8` or `nvfp4` (unset = BF16; release builds only) | Spark expert partials return compressed instead of BF16: fewer exchange bytes per prefill wave. Unmeasured. | Reply values at FP8/NVFP4 precision instead of BF16. |
| `SPARK_DECODE_REPLY_FORMAT` | V4.1, V4, GLM 5.3 / ML4, GLM Flash, MiMo, Qwen4, Kimi K3, Step-5, Muse | `SPARK_DECODE_REPLY_FORMAT=fp8` or `nvfp4` (unset = BF16; release builds only) | Spark expert partials return compressed instead of BF16: fewer exchange bytes per decode wave. Unmeasured. | Reply values at FP8/NVFP4 precision instead of BF16. |
| `ENGRAM_WARM` | V4.1 | `ENGRAM_WARM=on` (unset = off) | After readiness, reads both Engram tables into page cache on a background thread so per-token table reads hit RAM on stores with slow random reads. Unmeasured. | Host page cache equal to the table bytes (about 203 GB for the official checkpoint), reclaimable; skipped when free memory is below the tables plus 16 GiB. |

Stored `k8-compact` and `k6-compact` views are offered by
`wrldsuksgo2mars/Kimi-K3.1-Pro-EXL3-K2-MC`,
`wrldsuksgo2mars/Muse-Spark-Open-EXL3-K4-MC` and
`wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K4-MC`; `cuteafd-mc.json` declares the
views actually present at the selected revision. Plain checkpoints do not acquire
stored compact views through this option. Flash `fp8-compact` on qualified plain
checkpoints explicitly converts its supported MLA/shared-FFN target projections;
MC Flash refuses it until whole-core conversion is supported. KDA and target-head
conversion remain separate opt-ins, and no precision lever changes KV defaults.

## How options are set

A launch reads one plain `KEY=VALUE` config file (`cuteafd.config` by default):

```sh
./run.sh --config examples/configs/glm53-exl3-min.config
./run.sh --config my.config --set CONCURRENCY=16 --set SPARK_COUNT=4
./run.sh --config my.config --dry-run --total-capacity   # validate and print the plan
```

- Keys are `[A-Z_0-9]+`. Values are unquoted; whitespace needs quotes.
- An empty value means "unset": the default applies.
- An unknown name fails at parse (`unknown configuration key`, exit 2).
- A known name the selected family's launch path never reads fails as
  `unused configuration key: KEY` (exit 2). That is the refusal mechanism every
  `✗` in the family table below means.
- `--set KEY=VALUE` appends an override to a copy of the file, so it gets the
  same validation. `SPARK_COUNT`, `SPARK_HOSTS`, `WIP_INSTANCE`, `WIP_ROOT` and
  `API_KEY_FILE` may also come from the environment.
- `CONCURRENCY` unset is a boot-time memory derivation; all C-dependent sizes
  follow it. Set it only to pin a value.

`./run.sh` options:

| Option | Meaning |
| --- | --- |
| `--config FILE` | Config file to load (default `cuteafd.config`). |
| `--set KEY=VALUE` | One override, repeatable. |
| `--family ID` | Force a family instead of reading the snapshot's `config.json`. |
| `--embedding-placement host\|gpu` | Same as `EMBEDDING=host\|gpu`. |
| `--wip SLOT` | Serve a `./wip.sh` slot's artifacts. |
| `--restart [--all]` | Replace running containers; `--all` also stops other instances. |
| `--dry-run [--total-capacity]` | Validate, plan and print without touching services. |
| `--rotate-console-secret` | Re-provision the console unlock secret. |

`./stop.sh [--config FILE] [--all]` stops the containers for that config
(`--all` also removes release expert workers on the configured hosts).

The `cuteafd` binary: `plan MODEL`, `serve` (family from `--snapshot` or
`--family`), `golden`, `expertd`, `gateway`, `usage`, `doctor`, `fabric`,
`bench [run|fidelity|publish|smoke|export|cancel|panels]`, `bench-rdma`,
`bench-rdma-ring`, `transport-capabilities`. `cuteafd plan --layout` reports fit
before any kernel work; `cuteafd bench --url URL --profile NAME` runs a profile
on a launched server. Family serve subcommands (`serve-native`, `serve-dsv4`,
`serve-glm`, `serve-glmf`, `serve-mimo`, `serve-qwen4`, `serve-qwen35`,
`serve-kimi`, `serve-step5`, `serve-muse`) take the flags the launcher maps the
keys below onto, and `cuteafd serve --family ID --help` lists a family's serve
flags. Users set keys, not these flags or the `CUTEAFD_*` environment variables
the launcher derives from them.

`cuteafd kimi-golden --rtx-expert-layers auto|all|N` defaults to `auto`, using
startup's real-device admission rather than forcing zero coordinator layers.
Golden's expert-layer hint must cover the planner-selected Spark layers under
`auto`; explicit `all` or `N` requires an exact list. Only the admitted list is
sent to workers and loaded. Golden receipts record the chosen coordinator count,
Spark layer IDs and inactive hinted/loaded counts.
`cuteafd kimi-golden --skip-routed-experts --nll` instead runs the router and
shared experts with zero routed contribution, no Spark peers, no expert catalog
or RTX onboarding. It accepts only a matching `cuteafd.kimi.skipped-routed/1`
reference generated by `python/reference/families/kimi_k3/golden.py
--skip-routed-experts`. The reference uses only the sealed eight-window **quick**
subset (legacy, a00, a04, a08, a20, c00, d00, e00); a20 scores beyond 8K history
and exercises sparse DSA selection. Full-model reference defaults are unchanged.
This all-layer dense/attention/DSA diagnostic is **not** a full-model golden.
K3.1 DSA is admitted only through the explicit golden role;
serving remains refused until qualification. The flag is unavailable to serve.

`cuteafd qwen35-golden --golden DATASET --window ID` also accepts the pinned
published compact fidelity dataset for graph qualification when decode graphs
are enabled. It compares full decode logits byte-for-byte across graph-off,
graph-on warm-up and graph-on measured passes, requires measured replays and
identical NLL, and optionally runs `--verify-check` and `--prefix-check`. This
same-engine oracle does not replace the standard card's external top-1024
quality, strict lossless speculation or prefix-cache gates. Full-logits golden
inputs retain their existing reference checks; compact graph mode does not
support draft/MTP comparisons or `--save-logits`.

## Identity and instance

| Key | Default | Values | Notes |
| --- | --- | --- | --- |
| `MODEL_ID` | `deepseek-ai/DeepSeek-V4.1-Flash-1017` | HF repo id | Checkpoint to serve; starting configs always set it. |
| `MODEL_REVISION` | `1bae0e1088993b9b82de08f59de5be5a34e42170` | empty or 40–64 lowercase hex | Empty takes the hub checkout. |
| `SERVED_MODEL_ID` | `MODEL_ID` | string | Public id in the API. |
| `MODEL_VARIANT` | `flash` | `flash`, `pro` | V4.1 only; `pro` requires `EXPERT_FORMAT=exl3`. |
| `EXPERT_FORMAT` | `native` | `native`, `exl3` | V4.1 routed-expert format. |
| `SPARKINFER_EXL3` | `disable` | `auto`, `disable`, `force` | V4.1; `disable` needs `native`, `force` needs `exl3`. |
| `INSTANCE` | unset | `[A-Za-z0-9_.-]` | Namespaces containers and state. |
| `WIP_INSTANCE`, `WIP_ROOT` | unset | instance / absolute path | `--wip` slot staging. |
| `CUTEAFD_EXPORT_HOST`, `WIP_BUILD_SEED` | empty / `kakapo` | host / seed | Site build settings; every launch path accepts them. |

## Hardware placement

| Key | Default | Values | Notes |
| --- | --- | --- | --- |
| `RTX_GPUS` | `auto` | `auto`, `1`, `2` | Coordinator GPUs the layout may use. |
| `COORDINATOR_GPUS` | lowest index | ordered ids, primary first | Explicit GPU pair. |
| `COORDINATOR_SPLIT` | `auto` | `auto`, `heads`, `off` | Split a two-GPU layout by heads; `off` serves from the primary. |
| `SPARK_COUNT` | every available | `0`–`8` | Upper bound on physical Spark ranks available to the deployment (experts, media or drafter); the planner may use fewer when ranks have no role. `0` is accepted for every family when the planner can fit all owners on the selected RTX GPUs. Only ranks assigned a role start containers, using their configured host indices (unused indices may be skipped); expert peers include expert ranks only. |
| `SPARK_HOSTS` | pool order | comma list | Explicit rank order, same length as `SPARK_COUNT`. |
| `SPARK_<rank>_HOST` | pool host | hostname | Ranks 0–7; required for a host outside the pool. |
| `SPARK_<rank>_LANE_A` | pool value | IPv4 | Rail A address for that rank. |
| `SPARK_<rank>_LANE_B` | unset | IPv4 | Rail B; all active ranks or none. |
| `EXPERT_BACKEND` | `auto` | `auto`, `local`, `spark` | `local` requires `SPARK_COUNT=0`. Dense Qwen is local-only; the planner admits other families by fit. |
| `RTX_EXPERT_LAYERS` | `auto` | `auto`, `all`, `N` | Pool-first onboarding of routed-expert layers on the RTX pair. |
| `RTX_EXPERT_PEER` | `off` | `off` | `on` is refused; TP2 halves replaced it. |
| `EMBEDDING` | `gpu` | `host`, `gpu` | Untied token-embedding placement. |
| `EXPERT_PORT` | `19441` | 1–65535 | Expert-worker port. |
| `SPARK_REDUCTION_MIN_ROWS` | `16` | positive int | Row threshold for the reduction path. |
| `HOST_ISA` | `auto` | `auto`, `x86-64-v2`, `x86-64-v4` | Native library ISA. |
| `RDMA_BOND_BALANCE` | `off` | `off`, `labels`, `probe` | RoCE v2 flow-label placement on a coordinator bond. |
| `MAPPED_SCALE_PRELOAD_GIB` | `0` | GiB | Host-RAM cap for preloading mapped tables (V4.1 Engram); 0 keeps all mapped. |
| `ENGRAM_WARM` | `off` | `on`, `off` | V4.1 only: read both Engram tables into page cache after readiness. |
| `CUDA_MODULE_LOADING` | CUDA default (`LAZY`) | `LAZY`, `EAGER` | V4.1 only; `EAGER` costs startup time and memory, avoids first-shape hangs. |
| `SPARK_INTAKE` | `auto` | `auto`, `gpu`, `pinned`, `host` | How routed partials reach the coordinator GPU. |
| `SPARK_PREFILL_REPLY_FORMAT` | unset (BF16) | `bf16`, `fp8`, `nvfp4` | Spark prefill replies; FP8/NVFP4 exported only in release builds. |
| `SPARK_DECODE_REPLY_FORMAT` | unset (BF16) | `bf16`, `fp8`, `nvfp4` | Same for decode replies. |
| `MIMO_WEIGHT_POLICY` | `auto` | `auto`, `checkpoint` | MiMo weight precision policy. |
| `MIMO_HOST_CACHE` | `off` | `on`, `off` | MiMo host cache. |
| `MIMO_PREFIX_DRAFT` | `off` | `on`, `off` | MiMo draft on the prefix path. |
| `MIMO_SNAPSHOT_WAIT` | `off` | `on`, `off` | MiMo snapshot wait. |
| `MIMO_PREFILL_CHUNK_S` | unset | seconds | MiMo prefill chunk time. |

## Memory and pools

One memory policy covers every family
([`cuteafd-core/src/memory_policy.rs`](../rust/crates/cuteafd-core/src/memory_policy.rs)).
Each device has one cap, and weights, KV, workspaces and the reserve all live inside it.
The planner fixes the layout (layer placement, onboarding, buffers) for `KV_TARGET_TOKENS`;
the pool then grows into what the fixed layout leaves, up to `KV_MAX_TOKENS`, and the launch
is refused below `KV_MIN_TOKENS`. The reserve is the sum of named, measured components
(CUDA context and cuBLAS, loaded code, graphs, runtime growth, Spark worker process residency)
plus `FAMILY_RESERVE` and the explicit per-coordinator-GPU `release padding` allowance; `GPU_RESERVE` / `SPARK_RESERVE` replace the reserve components,
not physical driver page tables or startup graph owners (including node padding).
Startup and `cuteafd plan` print one line per device with the cap,
weights, workspaces, every reserve component and the KV target, min, max and pool.
The coordinator charges the SM121 worker process census (1,596,606,464 B,
moa/driver 580.178.04, 2026-11-07, #1454) inside `SPARK_MEMORY_CAP`:
steady anonymous PSS + file PSS + context-phase consumption, subtracting
context-phase file PSS growth to avoid overlap. The worker's live admission
checks future allocations against `MemAvailable`, so it does not charge that
already-resident process/context a second time. The historical 109,760,512-B
unattributed startup remainder is not an owner and is not charged.
`driver page tables` is one opaque runtime owner with distinct backing models,
not a tracked workspace or replaceable reserve. RTX portable pinned backings pay
ceil(bytes/512) per mapping coordinator context; aliases do not pay again.
Spark expert arenas pay 48 B per 4096-B page
(40 B vmalloc metadata + 8 B secondary page tables), rounded once per backing.
This is calibrated from real-worker kernel fields at 40.508, 75.059 and
103.652 GiB (zero-intercept fits 10.052338 + 1.949590 MiB/GiB).
Source attributes 16 B/page to `nvidia_pte_t` and 8 B/page to the retained DMA
page array; 16 B/page of vmalloc remains unattributed. Neither the fit's
intercept nor whole-host reclaim residuals are charged as fixed allowances.
Driver bytes survive `GPU_RESERVE` / `SPARK_RESERVE`, including zero. Ring, workspace,
validation and upload owners also stay separate and are charged once.
Memory sizes take `B`, `MB`, `GB`, `MiB` or `GiB` (a bare number is bytes), not percentages.
`GRAPH_NODE_PADDING` adjusts graph-node and executable rates and accepts percentages.
`MEMORY_PADDING` defaults to `100MiB` (104,857,600 B) per coordinator GPU, inside
`GPU_MEMORY_CAP`. This visible release allowance covers still-unattributed memory;
each release lowers it as owners are attributed. It is charged once, separately
from measured owners and `FAMILY_RESERVE`. `MEMORY_PADDING=0` removes only padding;
`GPU_RESERVE` replaces the whole reserve including padding, even when set to zero.
Sparks are unchanged and use `SPARK_RESERVE`, not `MEMORY_PADDING`.

`GRAPH_NODE_PADDING` is separate: by default it adds **2.5% to both measured
coefficients**, bytes per kernel node and bytes per graph executable, rounding each
adjustment to the nearest whole byte. The model sums `nodes * (a + pad_a)` and
`executables * (c + pad_c)`, then packs once per physical device into 2 MiB chunks.
Both counts come from the declared set; rates are per family and architecture.
It applies even with `GPU_RESERVE` or `SPARK_RESERVE`. An integer adds that many
bytes to each coefficient (per node and per executable); per-architecture values
may mix forms, e.g. `rtx=2.5%,spark=100`. Missing Spark rates require a Spark census,
never an RTX substitution. Whole-byte GLM SM120 coefficients reproduce the 1M
calibration exactly; the independent 128K prediction is one 2 MiB chunk above
measurement (within the independent-validation bar of plus or minus one chunk).
Flash's archived calibration also retains a one-chunk overprediction; it has no
independent context validation yet.

For the archived GLM 5.3 count this adds 74,660,034 B (about 71 MiB) before chunk
rounding; Flash adds 57,692,320 B (about 55 MiB). Together with `MEMORY_PADDING`'s
100 MiB default this is explicit driver-drift headroom, not a second graph charge.
Successful captures above the prediction warn and continue; actual allocation
failures stop. This shared model is staged: existing whole-capture families do
not charge the node padding until their admission is migrated.

Each key is also a global `cuteafd` flag (`--gpu-memory-cap`, `--kv-max-tokens`, ...).

| Key | Default | Values | Notes |
| --- | --- | --- | --- |
| `KV_TARGET_TOKENS` | 2M; 1M on cards of 32 GB or less | positive count | KV the layout is sized for: layer placement, onboarding, buffers. |
| `KV_MIN_TOKENS` | min(256K, model max context, KV target), on every layout | positive count | Refuse automatic pools below this; missing the target is fine down to it. Automatic context serves min(model max context, admitted pool); an explicit context still requires the pool to cover it. |
| `KV_MAX_TOKENS` | the target | positive count, at least the target | After an automatic layout is fixed, the pool grows into free memory up to this. A pinned `RTX_EXPERT_LAYERS` already makes the pool the output (it fills what is left). |
| `GPU_MEMORY_CAP` | full device | size | Total per-GPU ceiling for plan, serve and golden; leaves SM count and L2 unchanged. |
| `SPARK_MEMORY_CAP` | the family's Spark default | size | Total per-Spark engine ceiling, including worker process residency and routed weights; the outside-cap gap leaves room for OS/services. |
| `GPU_RESERVE` | auto | size | Sets the total GPU reserve; built-in reserve components (including release padding) become 0, but physical driver page tables, named sampled pre-weight occupancy and the bounded graph-cache reserve stay charged. Still inside the cap. |
| `SPARK_RESERVE` | auto | size | Replaces Spark reserve components, not arena driver metadata or actual ring/workspace/staging owners; these survive even a zero override. |
| `MEMORY_PADDING` | `100MiB` | size | Explicit per-coordinator-GPU release padding, inside the cap; `0` disables only padding. `GPU_RESERVE` replaces it. Global flag: `--memory-padding`. |
| `GRAPH_DISABLE_LOW_MEM` | half of `MEMORY_PADDING` (50 MiB by default) | size or `gpuN=size,...` | Coordinator-only physical free-memory floor after the estimated graph and existing instantiate headroom. `0` disables the check. Uses the latest boundary or periodic sample minus ledger and pending graph growth, without a new miss/replay CUDA query; a skip runs eagerly without a ticket or OOM cooldown. Each GPU decides independently; all-rank admission is follow-up work. Independent of `GPU_RESERVE`. Global flag: `--graph-disable-low-mem`. |
| `GPU_GRAPH_RESERVE` | trace-derived family base + bounded extra | size or `gpuN=size,...` | Optional **whole** coordinator graph reserve per device; a user value is charged exactly, without extra, and conflicts with either extra key for that device. Unspecified devices use the checked-in prewarm arena plus its 2 MiB tail as the base, then add `min(GPU_GRAPH_RESERVE_EXTRA, GPU_GRAPH_RESERVE_EXTRA_CAP)`. Resolved checkpoint/layout defaults: GLM 5.3 one-rank `glm_prewarm_1.json` base 337,641,472 bytes, charged 675,282,944 bytes; two-rank `glm_prewarm_2.json` base 1,379,926,016 bytes (1,377,828,864 measured plus the 2 MiB tail), charged 1,879,926,016 bytes per rank, traced on K4+Core-K6 with 2 RTX + 6 Sparks, 73 replayed geometries and no pins; untraced coordinator rank counts retain the one-rank bootstrap geometry/base without borrowing its pins and warn at startup; GLM Flash per resolved layout in `glmf_prewarm.json`: `1-sparks` base 115,343,360 bytes (113,246,208 measured plus the 2 MiB tail), charged 230,686,720 bytes, and `2` rank 0 / rank 1 bases 127,926,272 / 119,537,664 bytes (125,829,120 / 117,440,512 measured); Mistral Large 4 uses `ml4_prewarm_1.json` (base 715,128,832 bytes) or `ml4_prewarm_2.json` (base 914,358,272 bytes), each plus 500 MB extra; V4 selects `v4_prewarm.json` by resolved shape (coordinator GPU count x Sparks attached: `rtx1-spark`, `rtx1-local`, `rtx2-spark`, `rtx2-local`); a traced shape's base is its startup warm's measured arena plus its 2 MiB tail — 671,088,640 bytes on `rtx1-spark`; 954,204,160 on each `rtx2-spark` rank; 914,358,272 and 924,844,032 on `rtx2-local`, charged 1,171,088,640 / 1,454,204,160 / 1,414,358,272 and 1,424,844,032 bytes at the default extra — covering the replayed width-0 startup rows and 53 / 42 / 37 recurring verify geometries out of the traces' 5,476 / 5,169 / 5,264 verify keys (a width-0 row the trace never replayed, or a verify geometry used only inside one sampling interval such as a single panel or the 32K context pass, stays lazy under the extra), and an untraced shape keeps its provisional numbers (charged 751,619,276 bytes on `rtx1-local` at the default extra) and logs that its shape has no trace card. Qwen4 selects its checked-in prewarm base by resolved capture shape (RTX count, Sparks attached) from `qwen_prewarm.json`, plus the shared extra. V4.1 uses layout/rank inventories in `v41_prewarm.json`: one-rank measured base 3,630,170,112 bytes, charged 4,132,267,264 bytes; two-rank measured bases 1,625,292,800 / 1,279,262,720 bytes, charged 2,127,389,952 / 1,781,359,872 bytes (each charge includes one 2 MiB tail and 500 MB extra). An unmeasured V4.1 layout borrows neither another layout's base nor its pins. MiMo selects `mimo_prewarm.json` by checkpoint variant, coordinator rank count and Spark attachment: Flash `mimof-48` 1 RTX + Sparks has a 423,624,704-byte base including its 2 MiB tail, charged 847,249,408 bytes. Pro `mimop-70` 2 RTX + Sparks uses its own per-rank measured row shares and tails; an untraced checkpoint/layout uses unpinned bootstrap, never another key's evidence. Charged once before KV, independent of `GPU_RESERVE`; cache bound equals charged total. GLM/Mistral prewarm fits the base (or user total): pinned entries first, then list order until the next entry does not fit; if the pinned set cannot fit, prewarm none. V4 warms every geometry its list holds that its pool can express. Skips warn and become runtime misses. Qwen4's traced list is already the base by construction, so it warms in pinned-first list order and an explicit total below the base evicts the rare tail. Zero runs eagerly. A scalar applies independently to each GPU; unspecified `gpuN` selectors retain defaults. Startup, Ready and `/v1/stats` report charge and policy sources. Kimi K3 selects the per-physical-rank base in `kimi_prewarm.json` by checkpoint-native core format (`k6` or `bf16`) and coordinator rank count, plus the same fixed tail and bounded extra; untraced core views use bootstrap keys. BF16 2 RTX + 6 Sparks uses measured bases 1,281,359,872 / 1,279,262,720 bytes, charged 1,783,457,024 / 1,781,359,872 bytes including one 2 MiB tail and 500 MB extra; its coverage-1 adaptive trace selects 168 geometries with 21 pins. Untraced core/layout combinations use unpinned bootstrap geometry and a provisional census base, never another key's measured base, list or pins. Kimi startup fits indivisible pinned geometries first, then replay-hot unpinned geometries using measured per-rank capture shares within the base, never runtime extra. Kimi prefill remains eager and startup expands only replayable rows 1..64. Dense Qwen uses checkpoint/quant/KV-specific single-RTX inventories in `qwen35_prewarm.json`; untraced variants bootstrap unpinned with a provisional 64 MiB base plus one 2 MiB tail. Muse uses `muse_prewarm.json` per rank count and resolved onboarded-layer prefix: (1,2) measured base 1,056,964,608 bytes, charged 1,559,061,760 bytes; (2,0) measured bases 1,287,651,328 / 1,302,331,392 bytes, charged 1,789,748,480 / 1,804,428,544 bytes (each includes one 2 MiB tail and 500 MB extra); an untraced layout has an empty, unpinned bootstrap (zero base), so set an explicit reserve for initial trace collection. Step-5 EXL3 K4+Core-K6 inventories in `step5_prewarm.json` have measured bases 3,407,872,000 bytes (1 RTX + 3 Sparks) and 2,835,349,504 / 2,629,828,608 bytes (2 RTX + 6 Sparks); each adds one 2 MiB tail and the shared 500 MB extra when `DECODE_GRAPHS=on`. Untraced Step-5 layouts have no borrowed base or pins; explicit totals or extra enable bounded bootstrap captures. Other families await cache integration. |
| `GPU_GRAPH_RESERVE_EXTRA` | 500 MB | size, percentage or `gpuN=value,...` | Runtime capture headroom beyond the family prewarm base, coordinator GPUs only. MB is decimal: 500 MB = 500,000,000 bytes (MiB is binary). Percentages are of the base. An explicit value has no implicit cap; otherwise the default cap is 100% of the base. Conflicts with an explicit `GPU_GRAPH_RESERVE` total on that device. |
| `GPU_GRAPH_RESERVE_EXTRA_CAP` | 100% of base for default extra; no cap for explicit extra | size, percentage or `gpuN=value,...` | Upper bound on extra runtime headroom; percentages are of the family base. Charged total = base + min(extra, cap). Conflicts with an explicit `GPU_GRAPH_RESERVE` total on that device. No Spark extra keys: workers have no graph owner. |
| `SPARK_GRAPH_RESERVE` | worker family default | size | One graph reserve per Spark worker, matching the scalar `SPARK_RESERVE` convention. Forwarded to workers for GLM 5.3, GLM Flash, Qwen4, V4.1 and MiMo; the live native worker currently runs eagerly with no capture/replay sites, so no worker graph owner is active. Enabling worker capture requires admission ownership and bounded-cache integration together. |
| `GRAPH_TRACE` | off | absolute path | GLM 5.3 / GLM Flash / Mistral Large 4 / V4 / V4.1 / MiMo / Kimi K3 / dense Qwen / Muse schema-2 graph identity, per-cycle/lifetime hit counts, eviction/recapture/ghost counts, first/last-hit time, rolling 60 x 10 s hit intervals, measured boundary capture-byte shares, permanent node/exec cost basis, width/device-span and device high-water snapshots, written as `<stem>.device-N.json`; parent directory must exist. A multi-GPU shape writes one file per coordinator GPU, and its V4 trace invocation passes every rank's file. `scripts/dev/graph-prewarm-from-trace.py --family glm5 --layout 1|2 --trace FILE --output FILE` (GLM 5.3 coordinator rank count; pass both device traces for layout 2, whose half-table keys are doubled to the built width), `--family glm5_flash --layout 1-sparks|2 --trace FILE --output FILE` (GLM Flash, per resolved card layout), or `--family mistral_large4 --layout 1 --trace FILE --output FILE` (one rank; `--layout 2` takes both devices' traces and doubles each traced half table back to the built width), `--family deepseek_v4 --layout rtx1-spark|rtx1-local|rtx2-spark|rtx2-local --list rust/crates/cuteafd-loader/src/serving_capacity/v4_prewarm.json` instead replaces that one shape's per-rank arrays and marks it traced, printing each heat class's keys and arena bytes (sustained pins; plain C1 and the layout's highest-lifetime-hit verify geometry pin whatever their heat; other regular and single-interval rows are prewarmed and stay evictable; rare stays lazy; a width-0 startup row needs a replayed hit like any other geometry; entries are ordered pinned first, then lifetime replay hits descending). `--family deepseek_v41 --layout 1|2 --prefill-rows N --trace FILE --output FILE` generates the V4.1 rank inventory; `--family qwen4` generates shape-keyed entries. `--family mimo_v2 --ranks N --variant mimof-48\|mimop-70 --sparks yes\|no --trace FILE --output FILE` generates MiMo row-only entries, priced from every selected row's indivisible segment/head siblings; capture-only rows remain lazy. Kimi pins plain C1, the dominant verify width and its speculative siblings, and each device/operation lifetime-hit top tier carrying 50% of replay hits (ties included); sustained rolling heat and repeated recapture can also pin. Startup `Other` counts never create request heat. The checked-in Kimi adaptive fixture selects every replayed geometry with coverage 1; zero-replay geometry stays lazy. Kimi uses `--family kimi_k3 --variant bf16\|k6 --ranks N --layout NAME` with every physical device trace for that layout; it sums completed-boundary capture shares for the selected operation/row/speculative geometries, including startup-only siblings, and rounds once per logical rank. Missing shares fail generation; zero shares are valid allocator reuse. `--family qwen3_5 --variant h5120-l64-bf16-kvfp8-rtx1 --trace FILE --output FILE` generates checkpoint/quant/KV-specific dense Qwen entries from replayed complete-layer geometries, including their zero-hit segment siblings in the price. `--family muse_spark --ranks N --onboard-layers N --trace FILE --output FILE` generates Muse row geometries and measured sibling costs, ordered pinned first then replay heat. Runtime headroom comes from the shared extra keys, not the trace. Pinned-hit times are boundary-attributed; ARC-hit times use the cycle/boundary clock. Latest complete device snapshots supersede older heat/cost inventories. A layout without a checked-in trace prewarms the family bootstrap set unpinned and warns; only trace-derived entries pin. `/v1/stats` exposes top-16 hit/churn identities and per-key telemetry payload bytes without tracing; review and qualify the proposal before checking it in. |
| `GRAPH_NODE_PADDING` | `2.5%` | percentage, integer bytes per coefficient or `rtx=VALUE,spark=VALUE` | Added to both measured node and executable rates, packed once per device into 2 MiB chunks. Independent of `GPU_RESERVE`, `SPARK_RESERVE` and `MEMORY_PADDING`; included once in the graph owner, not a reserve. Global flag: `--graph-node-padding`. |
| `FAMILY_RESERVE` | per-family constant (0 in every family) | size | A named per-GPU reserve added to the measured components; `GPU_RESERVE` overrides it. One key for coordinator GPUs only: no family keeps a per-family Spark margin (Sparks use `SPARK_RESERVE`). |
| `KV_POOL_TOKENS` | unset | multiple of 64 | V4.1 only: fixed KV pool. Conflicts with `POOL_TOKENS`. |
| `KV_POOL_SIZE` | unset | `B/MB/GB/MiB/GiB` | V4.1 only: fixed KV pool size. |
| `POOL_TOKENS` | `auto` where offered | `auto` or count | Planner-sized pool up to 2M tokens; a number pins it. |
| `HOST_CACHE_BYTES` | `auto` | `auto`, `0`, byte size | Prefix-cache host tier; `0` disables it. |

## Attention and KV

| Key | Default | Values | Notes |
| --- | --- | --- | --- |
| `CORE_VIEW` | unset (= `original`) | `original`, `k8-compact`, `k6-compact`, `fp8-compact` | Immutable dense-core view. MC metadata advertises K8/K6; plain checkpoints refuse these. Original is the complete checkpoint-native core. `fp8-compact` refuses for Kimi K3.1 (plain or MC) and GLM Flash MC (whole-core FP8 pending). Muse MC supports **attention-only FP8** (q/k/v/o), not whole-core FP8. Other qualified plain families retain their current conversion scope, including Step-5 MTP O. No conversion of an FP8-origin core. |
| `ATTENTION_PLACEMENT` | `auto` | `auto`, `context`, `layers`, `heads` | Per-family defaults and refusals are in the table below. |
| `ATTENTION_PLAN_ROWS` | `auto` | `auto`, positive int | Decode/verify reduction plan rows. |
| `MAX_CONTEXT_TOKENS` | `auto` | `auto`, 1–1048576 | `auto`: serve min(model maximum context, admitted pool). A number pins context and requires the pool to cover it. Plan and every serve command use `--max-context-tokens` with `auto` or a number; explicit CLI overrides the config. Compiled workspace/table extent is internal, not a user context setting. |
| `MAX_OUTPUT_TOKENS` | `131072` | integer | Generation cap. |
| `KV_CACHE` | family default | `bf16`, `fp8`; MiMo also `int8` | See the precision table below. |
| `SWA_KV_CACHE` | `int8` | `bf16`, `int8`, `fp8` | MiMo sliding/MTP rings. |
| `DRAFT_KV_CACHE` | `fp8` | `bf16`, `fp8` | Drafter ring storage (GLM 5.3, GLM Flash, MiMo, Muse). |
| `PREFIX_CACHE_ENTRIES` | `auto` | `auto`, non-negative | Snapshot entries; auto derives from context × concurrency. |
| `PREFIX_CACHE_MARK_MIB` | unset | MiB | Recurrent/ring state retained per mark. |
| `PREFIX_PARTIAL` | `off` | `on`, `off` | MiMo approximate partial reuse; off keeps exact restores only. |
| `PREFIX_POINT_BOUNDARIES` | `1` | `0`, `1` | Dense Qwen: retain the exact frontier before the generation message. |
| `GLM5_FLASH_PREFIX_MARKS` | unset | `arena`, `pool` | GLM Flash KDA state marks. |
| `GLM5_FLASH_KDA_STATE` | `f32` | `f32`, `bf16` | GLM Flash KDA recurrent state. |
| `GLM5_FLASH_INDEX_CACHE` | `auto` | `keys`, `compact` | GLM Flash index cache; auto picks compact on one GPU. |
| `GLM5_FLASH_DECODE_ROWS` | `auto` (64) | `auto`, `64`, `128` | GLM Flash decode rows. |
| `DECODE_LANES` | `1` | `1`, `2` | V4.1/MiMo forward decode lanes. |
| `DECODE_SHARE` | `0.2` | `[0, 1)` | Share of time running requests keep while prompts prefill; 0 prefills whole prompts. |
| `DECODE_GRAPHS` | MiMo, dense Qwen, Muse and Step-5 `on` | `on`, `off` | Dense Qwen graphs complete decode/verify layers on its supported single-RTX layouts; prefill stays eager. With graphs off, its graph reserve is zero and page-table geometry stays on the original eager path. MiMo bounded per-layer CUDA graph replay, keyed by checkpoint and layout. Muse Spark defaults to `on` for bounded rank-local decode/verify segments; its prefill stays eager. Step-5 bounds decode/verify sections around eager KV staging and host joins; prefill remains eager. Step-5 selects trace-owned row geometries per native format, dense core, RTX count and Spark count from `step5_prewarm.json`, fitting pinned-first then a heat-ordered prefix across all ranks; startup uses disposable full-step transactions without publishing KV state. The shipped EXL3 K4+Core-K6 layouts (1 RTX + 3 Sparks and 2 RTX + 6 Sparks) retain their historical measured geometry pins. The shipped min/max traces select 58/59 replayed row geometries, respectively; min pins sustained row46 and max has no sustained pins. Capture-only rows remain lazy. Other Step-5 layouts have no borrowed pins or measured base. Standalone `plan --mimo-decode-graphs on\|off` and `--step5-decode-graphs on\|off` match serving defaults. |
| `KV_DECODE_RESERVE_CHUNK` | unset | tokens | Decode-chunk KV reserve top-up. |
| `KV_RESERVE_DRAIN_BUFFER` | unset | 0 = whole remaining output | Drain buffer. |
| `KV_EARLY_TERMINATION_MIN_OUTPUT_TOKENS` | unset | `-1` or non-negative | Early termination floor. |

## Speculation

| Key | Default | Values | Notes |
| --- | --- | --- | --- |
| `SPECULATOR` | family default | `auto` (Qwen4), `off`, `dflash2`, `dspark`, `mtp` | Which fail is in the table below. |
| `SPECULATOR_MODEL_ID` | family default | HF repo id | External drafter checkpoint. |
| `SPECULATOR_MODEL_REVISION` | empty | 40–64 lowercase hex | Drafter revision. |
| `SPECULATOR_DEPTH` | 3 for local Qwen/Step-5, else 1 | positive int | Native MTP draft depth. |
| `DRAFT_LENGTH` | unset (adaptive) | positive int | Fixed verified draft tokens per step: min(value, native head width), further capped by the existing row budget at high concurrency. Overrides `DRAFT_POLICY`; statistics still recorded. Every family and speculator (DFlash2, dSpark, MTP, Markov). V4.1 adaptive maximum remains 5 when unset; fixed supports up to 7. Without a speculator it is ignored with one startup log; reports and cards say `no speculation`. |
| `SPECULATOR_FP8` | `auto` | `auto`, `on`, `off` | GLM/Qwen/MiMo drafter precision: `on` converts to single-copy FP8. |
| `DRAFT_POLICY` | unset | `shared`, `simple`, `simple-rc`, `simple-derived` | Drafter block policy diagnostic. |
| `DRAFT_ROW_COST` | unset | `linear`, `power`, `log` | Drafter row cost model diagnostic. |
| `DRAFT_SURVIVAL_LENGTHS` | `off` | `on`, `off` | Record draft survival lengths. |
| `DRAFT_CONTEXT_SLOTS` | `max(20, drafts)` | positive int | GLM/MiMo drafter context slots. |
| `DRAFT_SEQUENCES` | engine default | positive int | Maximum drafts in one batch. |
| `SPECULATION_TRACE` | unset | absolute path | JSONL speculation trace. |
| `GLM5_FLASH_DRAFT_LINEAR` | auto by drafter | `w8a16`, `wide`, `w8a8` | GLM Flash drafter GEMM. |

## Prefill

| Key | Default | Values | Notes |
| --- | --- | --- | --- |
| `PREFILL_PACKING` | engine default | `unrestricted`, `exact` | Pack short prompts; `exact` preserves expert arithmetic routes. |
| `PREFILL_BATCH_TOKENS` | `2048` | 80–4096 | Prefill chunk size. |
| `PREFILL_ROWS` | planner | multiple of 64, ≤4096 | Per-unit rows override. |
| `PREFILL_WAVES` | planner | 0–4 | Units in flight. |
| `PREFILL_SPAN_ROWS` | planner | positive int | GLM/MiMo shared span. |
| `FULL_PREFILL_LOGITS` | engine/hidden | `on`, `off` | Legacy all-row diagnostic reservation (2.5 GiB at 4096 rows); may reduce KV or onboarding. Kimi, GLM Flash, GLM 5.3 / Mistral Large 4, Step-5, dense Qwen, Qwen4, Muse, MiMo and V4 prefill fidelity need no flag: they stream selected logits from bounded existing head storage, preserving transformer prefill units. The flag remains temporarily for old/new exactness checks. |
| `TARGET_HEAD` | `tensor` | `tensor`, `exact` | BF16 target head: tensor cores with FP32 accumulation, or the exact FP32 GEMM diagnostic. |
| `FP8_EXPERT_PREFILL` | `auto` | `auto`, `w8a8`, `w8a16` | FP8 expert prefill arithmetic. |
| `EXPERT_INPUT` | family default | `fp8`, `bf16`, `bf16-decode` | MiMo Spark expert inputs. |
| `EXPERT_PARTIAL_BF16_MAX_ROWS` | engine | row count | BF16 expert partials up to this row count. |
| `MIMO_PREFILL_CHUNK_S` | unset | seconds | MiMo prefill chunk time. |
| `GLM5_FLASH_PREFILL_BATCH` | `off` | `on`, `off` | GLM Flash one pass for multiple prompts. |
| `GLM5_FLASH_EXL3_SCHEDULE` | `default` | `default`, `gb10` | GLM Flash Spark EXL3 decode schedule. |
| `GLM5_FLASH_EXL3_WORKER_PATH` | `async` | `async`, `blocking` | GLM Flash Spark EXL3 upload/wait path. |
| `GLM5_FLASH_EXL3_ROUTE_DUMP` | unset | absolute dir | Per-Spark EXL3 route dump. |
| `GLM5_FLASH_EXL3_ROUTE_DUMP_CALLS` | `200000` | positive int | Dump call cap. |
| `GLM5_FLASH_ROUTE_RING_CHECK` | `off` | `on`, `off` | Diagnostic router-id copy through the local ring. |
| `GLM5_FLASH_REPLAY_RECORDS` | `auto` | `shared`, `own` | GLM Flash replay records (`auto` selects `shared` on one GPU with an automatic pool and admitted Spark experts, else `own`). |

## Media

| Key | Default | Values | Notes |
| --- | --- | --- | --- |
| `VISION` | `auto` | `auto`, `off`, `rtx[:gpu]`, `spark[:rank]` | Vision tower placement; `auto` enables the family's bundled tower. `spark[:rank]` may use a media-only Spark when every expert layer is on RTX (include it in `SPARK_COUNT`); that rank starts only the tower, not expert transport. GLM Flash passes physical tower ranks independently of expert peers, including sparse ranks and replicas. |
| `AUDIO` | `auto` | `auto`, `off` | Bundled audio tower. |
| `VISION_REPLICAS` | `1` | 1–8 | Vision replicas. |
| `MEDIA_CACHE_BYTES` | engine | byte size | Media cache. |
| `MEDIA_FEATURES_DIR`, `AUDIO_FEATURES_DIR` | unset | absolute dir | Encoder feature caches. |
| `CHAT_TEMPLATE_FROM` | checkpoint | snapshot dir | GLM Flash template source when the checkpoint's template cannot mark images. |

Kimi K3.1 (`kimi_k31`) is recognized by the shared Kimi family for CPU header
and core-view inspection. Its Gated DSA device path and index-cache prefix
restore are not installed yet, so plan/serve refuse execution (#1777); dense
MLA is not a fallback. K3 DFlash2 hidden-state compatibility is unqualified,
and requesting a drafter for K3.1 refuses explicitly. MC `original`,
`k8-compact` and `k6-compact` retain native BF16 indexer weights;
`fp8-compact` refuses until whole-core conversion is implemented.

Kimi K3 and Step-5 are text-only; dense Qwen has media off. GLM 5.3 / ML4,
GLM Flash, MiMo, Qwen4 and Muse carry their own towers. Native GLM Flash EXLR8
K4 min/max examples use automatic bundled vision; the planner admits its resident
weights and scratch alongside the KV pool and routed experts.

## Precision

Target weights keep checkpoint precision unless an explicit conversion option is
selected: unset = checkpoint/native format. This includes ordinary BF16
checkpoints and the MC `original` view. Native FP8 checkpoint weights remain
FP8. Drafters may retain their qualified FP8 defaults: target verification
controls output. Qwen has one shared target/MTP head: BF16, the draft reads it;
there is no extra draft-only FP8 copy. KV cache dtype is independent and unchanged.

Keys here that are optional and off by default are also listed under
[Optional performance levers](#optional-performance-levers).

| Key | Default | Values | Notes |
| --- | --- | --- | --- |
| `EXLR8_MANIFEST` | published `bit325` | manifest name (`up1`, `bit325`, …) | Native EXLR8 experts; needs 1–2 RTX and 1–8 Sparks. |
| `EXLR8_PREFILL_ACTIVATIONS` | `a16` | `a16`, `a8` | Native EXLR8 trellis prefill activations; decode stays A16. |
| `V41_FP8_HEAD` | `auto` (engine `packed`) | `auto`, `off`, `draft`, `all`, `packed` | V4.1: unset/`auto` selects the exact packed BF16 target head and coarse dSpark plane on single/dual RTX, including push loading. `all` uses E4M3 for both; `off` keeps BF16; `draft` keeps BF16 plus E4M3 and requires mounted loading. Packed adds 300.214 MiB total over `all` (150.107 MiB/GPU on dual RTX); ordinary admission decides fit. |
| `CUTEAFD_GLM_BF16` | unset (= `native`) | `native`, `fp8` | GLM 5.3 target MLA, indexer and dense/shared FFNs preserve checkpoint BF16; `fp8` explicitly converts BF16 to block FP8. Also accepted as an environment variable; launch-key value wins. No effect on native FP8 checkpoints or drafter precision. |
| `FP8_SCALES` | `amax` | `amax`, `pow2`, `best` | Scale rule for FP8 copies made at load. |
| `L2_PREFETCH` | `off` | `off`, `auto`, MiB | Pull the next layer's weights into L2 during one-lane decode. |
| `QWEN_FP8_DECODE` | unset | `on`, `off` | Qwen GDN/attention projections to resident E4M3. |
| `QWEN_FP8_HEAD` | unset (BF16) | `on`, `off` | Explicit conversion of the shared target/MTP head to resident E4M3. |
| `MIMO_FP8_HEAD` | unset (= `auto`, checkpoint) | `auto`, `on`, `off` | Target head FP8 is opt-in; native FP8 stays FP8. Drafter defaults are independent. |
| `MIMO_FP8_O_PROJ` | unset (= `auto`, checkpoint) | `auto`, `on`, `off` | Explicit BF16 target O-projection conversion; native FP8 stays FP8. |
| `GLM5_FLASH_FP8_HEAD` (alias `GLMF_FP8_HEAD`) | unset (= `auto`, checkpoint BF16 on 1 or 2 GPUs) | `auto`, `on`, `off`, `packed` | GLM Flash BF16 checkpoint head: FP8 copy, BF16, or an exact 12-bit operand copy (one GPU). Native K6/EXLR8 heads ignore this conversion setting. |
| `GLM5_FLASH_FP8_PREFILL` (alias `GLMF_FP8_PREFILL`) | unset (engine `mla,ffn`) | pass list (`mla`, `ffn`, `kda-in`, `kda-o`), `all`, `off` | W8A8 only over already-FP8 weights, vendor-native or explicitly converted. Default MLA and dense/shared-FFN passes do nothing to BF16 original/MC original operands: those remain BF16. Added KDA passes require KDA FP8. No implicit target-weight conversion. |
| `GLM5_FLASH_KDA_FP8` (alias `GLMF_KDA_FP8`) | unset (`auto` = `off`) | `auto`, `off`, `row128`, `channel` | Explicit BF16 KDA projection conversion to resident E4M3; recurrent state and KV defaults are independent. |
| `GLM5_FLASH_KDA_SPLIT` | `auto` | `auto`, `partials` | GLM Flash KDA split output. |
| `GLM5_FLASH_TARGET_HEAD` | `auto` | `auto`, `tensor`, `exact` | GLM Flash's `TARGET_HEAD` spelling. |
| `TP2_ATTENTION`, `TP2_QUERY_PROJECTION`, `TP2_OUTPUT_PROJECTION` | `off` | `on`, `off` | V4.1 TP2 projections on the RTX pair. |
| `TP2_DSPARK_EXPERTS` | `on` with dual RTX + dSpark, otherwise `off` | `on`, `off` | V4/V4.1 TP2 draft experts; requires two RTX GPUs and dSpark. Explicit `off` retains GPU0-only draft experts. |
| `TP2_DFLASH` | MiMo/Kimi/Muse dual-RTX DFlash2: `on`; GLM Flash: planner choice; otherwise: `off` | `on`, `off` | DFlash tensor parallelism on an active two-RTX split with DFlash2. Muse defaults to `on` with dual-RTX DFlash2 (`off` remains available): rank-local query/KV heads and MLP neurons use the existing target peer exchange; selector and vocabulary head stay on GPU1. GLM Flash compares the whole GPU0 drafter with a head split: larger admitted pool wins, ties split for balance. Omit the key for this automatic choice; explicit `on` or `off` wins. Auto requires two GPUs and checkpoint-derived drafter owners (`--drafter-gib` keeps whole placement). GLM Flash selector weights and head-only workspace stay on GPU0. Plan accepts `--draft-fp8 true|false` (omission defaults to FP8 for GLM Flash) and `--draft-linear w8a16|wide|w8a8` and charges the same representation and scratch as serve. MiMo and GLM Flash split query/KV heads and MLP neurons, retaining replicated residual/convolution owners. Kimi splits cache and attention by Context; producer weights and FFN stay on GPU1. CLI false overrides config on. |

## Concurrency and serving

| Key | Default | Values | Notes |
| --- | --- | --- | --- |
| `ADDR` | `0.0.0.0:8000` | `HOST:PORT` | Listen address. |
| `CONCURRENCY` | boot-time derivation | 1–64 | Decode batch concurrency; every C-dependent size follows it. |
| `HTTP_QUEUE_DEPTH` | `CONCURRENCY` | positive int | Accepted request queue. |
| `HTTP_QUEUE_WAIT_MS` | `25000` | ms | Queue wait before refusal. |
| `API_KEY_FILE` | unset | readable file | API key; env `API_KEY_FILE` also works. |
| `ENABLE_BENCH` | `off` | `on`, `off` | Enable the in-server bench endpoints. |
| `GATEWAY` | binary default `on` | `on`, `off` | Messages/Responses/Realtime routes. |
| `USAGE` | `on` | `on`, `off` | Usage history tier. |
| `EXPERT_USAGE` | `on` | `on`, `off` | Device-side executed expert routes, separate prefill/decode tallies and 5 s / 60 s / total windows at `/v1/stats`. Active for V4.1, V4, MiMo, Kimi K3, GLM 5.3 and ML4 by default; `off` disables counting and telemetry owners. Families without an adapter remain effectively off and show the `/experts` placeholder. V4/GLM 5.3/ML4 `--skip-routed-experts` and MiMo `--skip-experts` disable telemetry before admission, even with `on`. Direct engine/plan CLI uses `--expert-usage=false` to opt out. Shared adapters extend coverage. Each canonical producer GPU owns counters; one worker merges their aggregate snapshots into one bounded engine history. `/experts` uses the console's admin cookie. Rejected speculative verification rows count; GLM 5.3/ML4 startup warm-up and graph-capture replays are excluded; graph padding and TP2 replicas do not. Window and benchmark phase boundaries are approximate asynchronous samples; short phases can share a sample. A 5 s snapshot-completion timeout logs an error and leaves history stale without stopping serving. Totals saturate at u64 limits; merged-counter overflow stops sampling with an error. No disk persistence. |
| `CONSOLE_ASSETS` | `off` | `on`, `off` | Serve console pages from a checkout; `on` needs `--wip`. The `/memory` dashboard uses the same admin console cookie as `/experts`; its read-only `/console/memory` snapshot combines all accepted Growth owner forms (including deferred owners), reconciled worker workspace and future-ring owners, background allocation inventory and published KV/prefix/graph counters. Named-owner bars use exact capacity scales; the domain selector expands non-weight/non-KV owners on a subtotal scale. It polls each second and retains five minutes only while open. Allocation inventory refreshes every 10 s; reconciliation remains stage-sampled, with Spark launch-relative figures diagnostic only. Missing fields are unavailable, never zero. |
| `CONSOLE_TEXT` | `on` | `on`, `off` | Live console token text for unlocked viewers. |
| `HOST_WAIT` | `block` | `spin`, `block` | Serving host-thread wait mode. |
| `IDLE_PARK_SECONDS` | `30` | positive number | Park polling threads after this idle time. |
| `THERMAL` | `on` | `on`, `off` | NVML thermal sampling for stats/bench reports. |
| `LOOP_ABORT_RUN`, `LOOP_ABORT_COPIES`, `LOOP_FAST_SPAN`, `LOOP_FAST_COPIES` | 512, 8, 1024, 8 | integers | Loop-abort rule guards; 0 disables. |
| `PROBE_DUMP_ROOT` | unset | existing absolute dir | Remote bench probe dump root. |
| `COORDINATOR_DOCKER_INFERENCE`, `COORDINATOR_DOCKER_DEV`, `SPARK_EXPERT_DOCKER_*` | release/dev image names | image ref | Image selection. |

## Profiling

| Key | Default | Values | Notes |
| --- | --- | --- | --- |
| `PROFILE` | unset | comma list of `census`, `transport-timing`, `expert-timing`, `expert-routes`, `draft-timing`, `nsys`, `ncu` | Wraps the coordinator for one launch; nsys and ncu are exclusive. |
| `PROFILE_DIR` | unset | absolute dir | Report destination; also runner-owned for `wip-cards --probe memory` without a profiler. |
| `PROFILE_NCU_KERNEL` | unset | kernel regex | Required by `PROFILE=ncu`; each match is replayed. |
| `PROFILE_NCU_COUNT`, `PROFILE_NCU_SET`, `PROFILE_NCU_SKIP` | 1, `full`, 0 | 1–9999 / section set / integer | ncu replay count, section set, warm-up matches skipped. |
| `PROFILE_NSYS_GPU_METRICS`, `PROFILE_NSYS_CPU`, `PROFILE_NSYS_GRAPH_TRACE` | unset | GPU list / `on`/`off` / `node`/`graph` | nsys extras. |
| `PROFILE_EXPERT_INPUT_SAMPLE`, `PROFILE_EXPERT_ROUTE_WEIGHTS` | unset | row count / `on`/`off` | Require `PROFILE=expert-routes`. |
| `PROFILE_SPARK`, `PROFILE_SPARK_RANKS` | unset | `nsys`, `ncu` / rank list | Profile Spark workers instead of the coordinator. |
| `PROFILE_SPARK_NCU_*`, `PROFILE_SPARK_NSYS_GPU_METRICS` | unset | as above | Spark-side profiling. |

The memory probe publishes measured per-logical-GPU sampler context bytes in
`PROFILE_DIR/observer-context.json`. The launcher declares that file to the
coordinator only; startup waits at most 30 seconds for the measurement and fails
if it is missing or invalid. Logical memory samples exclude those bytes from
used memory and from physical capacity, so the context owner is production-only
and an uncapped census leaves exactly the observer's bytes out of the pool.
An explicit smaller `GPU_MEMORY_CAP` uses existing physical headroom instead.
Raw physical samples and `memory.jsonl` remain unchanged. No observer declaration
means no adjustment.

`CUTEAFD_DIAG_KIMI_STARTUP_MEMORY` is a private coordinator-only Kimi
process environment diagnostic, not a launch configuration key. Pass it through
`wip-cards --arm-env`; unset or `off` does not sample or synchronize.
`pre-capture` samples in-process raw CUDA memory and native backing changes at
at most 64 synchronized phase boundaries, logs signed used/tracked/unclassified
deltas, then intentionally exits before target graph capture or Ready. It adds
no external observer, request probes, admission adjustment or automatic allowance.

## Family support: option values

Attention placement (`ATTENTION_PLACEMENT`; `auto` resolves through the
family's planner and runtime policy):

| Value | V4.1 | V4 | GLM 5.3 / ML4 | GLM Flash | MiMo | Qwen4 | Qwen dense | Kimi K3 | Step-5 | Muse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `auto` (default) | ✓ layers¹ | ✓ context | ✓ context | ✓ context | ✓ heads | ✓ layers | ✓ heads² | ✓ context⁴ | ✓ layers | ✓ heads |
| `heads` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓⁴ | ✗ | ✗ |
| `context` | ✗ | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | ✓⁴ | ✗ | ✗ |
| `layers` | ✗ | ✗ | ✓ (explicit; auto stays context)³ | ✓ (explicit; auto stays context) | ✗ | ✓ | ✗ | ✓ | ✗ | ✗ |

`✓` means the launch path accepts the value; `✗` means it stops with the
family's attention-placement error.

¹ V4.1 is `layers` on two GPUs and `heads` on one; an explicit `heads` is
accepted on either. ² Dense Qwen is a single-device executor: `auto` resolves
to `heads`, and an explicit `heads` is refused until a multi-GPU executor is
qualified. ³ Native GLM 5.3 EXLR8 runs `context` (the auto default) and
explicit `heads` or two-GPU `layers` (#1305, #1308); the planner refuses a
mode whose image lacks its programs. ⁴ Kimi K3 accepts explicit `heads`,
`context` and `layers` for MLA layers on two GPUs; `auto` selects `context`
on two GPUs and retains whole-owned layers on one. KDA stays head-split on
two GPUs, independent of the MLA placement.

Speculator (`SPECULATOR`):

| Value | V4.1 | V4 | GLM 5.3 / ML4 | GLM Flash | MiMo | Qwen4 | Qwen dense | Kimi K3 | Step-5 | Muse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `off` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `dflash2` | ✗ | ✗ | ✓ (default) | ✓ (default) | ✓ | ✗ | ✗ | ✓ (default) | ✗ | ✓ (default) |
| `dspark` | ✓ (default) | ✓ (default) | ✓ (Mistral Large 4 only) | ✓ | ✗ | ✗ | ✓ (default unless hidden is 5120) | ✗ | ✗ | ✗ |
| `mtp` | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ (default) | ✓ (local EXL3/NVFP4 default, depth 3) | ✗ | ✓ (default) | ✗ |

Qwen4 unset/`auto` selects native MTP (depth 3) only for qualified local
EXL3 or ModelOpt NVFP4 with one MTP stage and an admitted Spark expert
world of zero. Configured Spark inventory and admitted vision-only ranks do
not disable this default. Media-only Spark layouts still require an available
worker and complete encoder admission. Other formats and Spark expert layouts
default off. Explicit values win.

KV precision (`KV_CACHE`; `SWA_KV_CACHE` is MiMo only):

| Value | V4.1 | V4 | GLM 5.3 / ML4 | GLM Flash | MiMo | Qwen4 | Qwen dense | Kimi K3 | Step-5 | Muse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `bf16` | n/a (fixed) | n/a | n/a | n/a | ✓ | ✓ | ✓ | ✓ | ✓ | n/a |
| `fp8` | n/a | n/a | n/a | n/a | ✗ | ✓ (default) | ✓ (default) | ✓ (default) | ✓ (default) | n/a |
| `int8` | n/a | n/a | n/a | n/a | ✓ (default) | ✗ | ✗ | ✗ | ✗ | n/a |
| `SWA_KV_CACHE` `fp8` | n/a | n/a | n/a | n/a | ✓ | n/a | n/a | n/a | n/a | n/a |

V4.1, V4, GLM 5.3/ML4, GLM Flash and Muse fix their cache formats; GLM
families and Muse expose `DRAFT_KV_CACHE` (`bf16`/`fp8`, default `fp8`) for the
drafter ring instead.

Media:

| Key | V4.1 | V4 | GLM 5.3 / ML4 | GLM Flash | MiMo | Qwen4 | Qwen dense | Kimi K3 | Step-5 | Muse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `VISION` | ✓ (`rtx`/`spark`; local is `rtx:0`) | ✓ | ✓ | ✓ (`auto` needs a template that can mark images) | ✓ | ✓ | ✓ (`auto`/`off`; auto starts none) | ✓ (`auto`/`off`; auto starts none) | ✓ (`auto`/`off`; auto starts none) | ✓ |
| `AUDIO` | `auto`/`off` | `auto`/`off` | `auto`/`off` | `auto`/`off` | `auto`/`off` | `auto`/`off` | `auto`/`off` (auto starts none) | `auto`/`off` (auto starts none) | `auto`/`off` (auto starts none) | `auto`/`off` |

Native EXLR8 (`EXLR8_MANIFEST`): V4, GLM 5.3/ML4 and GLM Flash, MiMo (dense core
only) and Qwen4. Placement is 1–2 RTX plus 1–8 Sparks; GLM Flash TP2 also
allows 0 Sparks. The speculator must be `off` or `dflash2`. GLM needs its
attention heads or a two-GPU context/layers placement; native GLM EXLR8 takes
context (auto), heads and two-GPU layers (#1305, #1308).

## Family support: launch keys

Every key the launcher knows, checked against each launch path's real
accept/refuse behaviour. `✓` means the family's config validation accepts it;
`✗` means the launch stops with `unused configuration key: KEY` (exit 2).
`SPARK_<rank>_*` covers ranks 0–7. In the combined GLM 5.3 / ML4 column,
`CORE_VIEW` is accepted for both families. ML4 original keeps its native E4M3
weights and F32 block-128 scales; only `original` is accepted because its
original core is already FP8, not an MC publication.

| Key | V4.1 | V4 | GLM 5.3 / ML4 | GLM Flash | MiMo | Qwen4 | Qwen dense | Kimi K3 | Step-5 | Muse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Identity** | | | | | | | | | | |
| `MODEL_ID` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `MODEL_REVISION` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `SERVED_MODEL_ID` | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `MODEL_VARIANT` | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| `EXPERT_FORMAT` | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| `SPARKINFER_EXL3` | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| `INSTANCE` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `WIP_INSTANCE` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `WIP_ROOT` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `CUTEAFD_EXPORT_HOST` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `WIP_BUILD_SEED` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| **Placement** | | | | | | | | | | |
| `RTX_GPUS` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `COORDINATOR_GPUS` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `COORDINATOR_SPLIT` | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `SPARK_COUNT` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `SPARK_HOSTS` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `SPARK_<rank>_HOST` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `SPARK_<rank>_LANE_A` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `SPARK_<rank>_LANE_B` | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| `EXPERT_BACKEND` | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `RTX_EXPERT_LAYERS` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ |
| `RTX_EXPERT_PEER` | ✗ | ✓ | ✓ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| `EMBEDDING` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `EXPERT_PORT` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ |
| `SPARK_REDUCTION_MIN_ROWS` | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| `HOST_ISA` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ |
| `RDMA_BOND_BALANCE` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ |
| `MAPPED_SCALE_PRELOAD_GIB` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `ENGRAM_WARM` | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| `CUDA_MODULE_LOADING` | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| `SPARK_INTAKE` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ |
| `MIMO_WEIGHT_POLICY` | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| `MIMO_HOST_CACHE` | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| `MIMO_PREFIX_DRAFT` | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| `MIMO_SNAPSHOT_WAIT` | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| `MIMO_PREFILL_CHUNK_S` | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| **Memory and pools** | | | | | | | | | | |
| `KV_TARGET_TOKENS` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `KV_MIN_TOKENS` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `KV_MAX_TOKENS` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `GPU_MEMORY_CAP` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `SPARK_MEMORY_CAP` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `GPU_RESERVE` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `MEMORY_PADDING` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `GRAPH_NODE_PADDING` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `SPARK_RESERVE` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `GRAPH_DISABLE_LOW_MEM` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `GPU_GRAPH_RESERVE` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `GPU_GRAPH_RESERVE_EXTRA` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `GPU_GRAPH_RESERVE_EXTRA_CAP` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `SPARK_GRAPH_RESERVE` | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ |
| `GRAPH_TRACE` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `FAMILY_RESERVE` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `KV_POOL_TOKENS` | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| `KV_POOL_SIZE` | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| `POOL_TOKENS` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `HOST_CACHE_BYTES` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| **Attention and KV** | | | | | | | | | | |
| `CORE_VIEW` | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `ATTENTION_PLACEMENT` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ |
| `ATTENTION_PLAN_ROWS` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ |
| `MAX_CONTEXT_TOKENS` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `MAX_OUTPUT_TOKENS` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `KV_CACHE` | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ |
| `SWA_KV_CACHE` | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| `DRAFT_KV_CACHE` | ✗ | ✗ | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✓ |
| `PREFIX_CACHE_ENTRIES` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `PREFIX_CACHE_MARK_MIB` | ✗ | ✗ | ✗ | ✓ | ✓ | ✗ | ✓ | ✗ | ✗ | ✗ |
| `PREFIX_PARTIAL` | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| `PREFIX_POINT_BOUNDARIES` | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ |
| `GLM5_FLASH_PREFIX_MARKS` | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| `GLM5_FLASH_KDA_STATE` | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| `GLM5_FLASH_INDEX_CACHE` | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| `GLM5_FLASH_DECODE_ROWS` | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| `DECODE_LANES` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `DECODE_SHARE` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `DECODE_GRAPHS` | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | ✓ | ✗ | ✓ | ✓ |
| `KV_DECODE_RESERVE_CHUNK` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `KV_RESERVE_DRAIN_BUFFER` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `KV_EARLY_TERMINATION_MIN_OUTPUT_TOKENS` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| **Speculation** | | | | | | | | | | |
| `SPECULATOR` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `SPECULATOR_MODEL_ID` | ✗ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✗ | ✓ |
| `SPECULATOR_MODEL_REVISION` | ✗ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✗ | ✓ |
| `SPECULATOR_DEPTH` | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ | ✓ | ✗ | ✓ | ✗ |
| `DRAFT_LENGTH` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `SPECULATOR_FP8` | ✗ | ✗ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ |
| `DRAFT_POLICY` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ |
| `DRAFT_ROW_COST` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ |
| `DRAFT_SURVIVAL_LENGTHS` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ |
| `DRAFT_CONTEXT_SLOTS` | ✗ | ✗ | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| `DRAFT_SEQUENCES` | ✗ | ✗ | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| `SPECULATION_TRACE` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ |
| `GLM5_FLASH_DRAFT_LINEAR` | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| **Prefill** | | | | | | | | | | |
| `PREFILL_PACKING` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `PREFILL_BATCH_TOKENS` | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✗ | ✗ | ✓ | ✗ |
| `PREFILL_ROWS` | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✗ | ✓ |
| `PREFILL_WAVES` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ |
| `PREFILL_SPAN_ROWS` | ✗ | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| `FULL_PREFILL_LOGITS` | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `TARGET_HEAD` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ |
| `FP8_EXPERT_PREFILL` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ |
| `EXPERT_INPUT` | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ |
| `EXPERT_PARTIAL_BF16_MAX_ROWS` | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |
| `GLM5_FLASH_PREFILL_BATCH` | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| `GLM5_FLASH_EXL3_SCHEDULE` | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| `GLM5_FLASH_EXL3_WORKER_PATH` | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| `GLM5_FLASH_EXL3_ROUTE_DUMP` | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| `GLM5_FLASH_EXL3_ROUTE_DUMP_CALLS` | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| `GLM5_FLASH_ROUTE_RING_CHECK` | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| `GLM5_FLASH_REPLAY_RECORDS` | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| **Media** | | | | | | | | | | |
| `VISION` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `AUDIO` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `VISION_REPLICAS` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | ✓ |
| `MEDIA_CACHE_BYTES` | ✗ | ✗ | ✗ | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ |
| `MEDIA_FEATURES_DIR` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `AUDIO_FEATURES_DIR` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `CHAT_TEMPLATE_FROM` | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ |
| **Precision** | | | | | | | | | | |
| `EXLR8_MANIFEST` | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ |
| `EXLR8_PREFILL_ACTIVATIONS` | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ |
| `V41_FP8_HEAD` | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| `FP8_SCALES` | ✗ | ✗ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ |
| `L2_PREFETCH` | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ |
| `CUTEAFD_GLM_BF16` | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| `QWEN_FP8_DECODE` | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ |
| `QWEN_FP8_HEAD` | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ |
| `MIMO_FP8_HEAD` | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| `MIMO_FP8_O_PROJ` | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| `GLM5_FLASH_FP8_HEAD` | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| `GLM5_FLASH_FP8_PREFILL` | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| `GLM5_FLASH_KDA_FP8` | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| `GLM5_FLASH_KDA_SPLIT` | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| `GLM5_FLASH_TARGET_HEAD` | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| `GLMF_FP8_HEAD` | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| `GLMF_FP8_PREFILL` | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| `GLMF_KDA_FP8` | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| `TP2_ATTENTION` | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| `TP2_QUERY_PROJECTION` | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| `TP2_OUTPUT_PROJECTION` | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| `TP2_DSPARK_EXPERTS` | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| `TP2_DFLASH` | ✗ | ✗ | ✗ | ✓ | ✓ | ✗ | ✗ | ✓ | ✗ | ✓ |
| **Concurrency and serving** | | | | | | | | | | |
| `ADDR` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `CONCURRENCY` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `HTTP_QUEUE_DEPTH` | ✓ | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| `HTTP_QUEUE_WAIT_MS` | ✓ | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| `API_KEY_FILE` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `ENABLE_BENCH` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `GATEWAY` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `USAGE` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `EXPERT_USAGE` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `CONSOLE_ASSETS` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `CONSOLE_TEXT` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ |
| `HOST_WAIT` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ |
| `IDLE_PARK_SECONDS` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ |
| `THERMAL` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `LOOP_ABORT_RUN` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `LOOP_ABORT_COPIES` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `LOOP_FAST_SPAN` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `LOOP_FAST_COPIES` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `PROBE_DUMP_ROOT` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ |
| `COORDINATOR_DOCKER_INFERENCE` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `COORDINATOR_DOCKER_DEV` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `SPARK_EXPERT_DOCKER_INFERENCE` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ |
| `SPARK_EXPERT_DOCKER_DEV` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ |
| `SPARK_PREFILL_REPLY_FORMAT` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ |
| `SPARK_DECODE_REPLY_FORMAT` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ |
| **Profiling** | | | | | | | | | | |
| `PROFILE` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `PROFILE_DIR` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `PROFILE_NCU_KERNEL` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `PROFILE_NCU_COUNT` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `PROFILE_NCU_SET` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `PROFILE_NCU_SKIP` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `PROFILE_NSYS_GPU_METRICS` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `PROFILE_NSYS_CPU` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `PROFILE_NSYS_GRAPH_TRACE` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `PROFILE_EXPERT_INPUT_SAMPLE` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ |
| `PROFILE_EXPERT_ROUTE_WEIGHTS` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ |
| `PROFILE_SPARK` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ |
| `PROFILE_SPARK_RANKS` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ |
| `PROFILE_SPARK_NCU_KERNEL` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ |
| `PROFILE_SPARK_NCU_COUNT` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ |
| `PROFILE_SPARK_NCU_SET` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ |
| `PROFILE_SPARK_NCU_SKIP` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ |
| `PROFILE_SPARK_NSYS_GPU_METRICS` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ |

MiniMax is not a launch family yet; it gets a column here when it lands.
