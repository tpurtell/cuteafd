# CuteAFD plan

One attention/FFN-disaggregated engine for the open-weights MoE models that
soar on consumer Blackwell: strong RTX PRO 6000 head(s) for attention, dense
projections, routing, shared experts, sampling and speculation; a pool of DGX
Sparks for routed experts over RoCE. Rust host code, CuTe-DSL/Triton AOT
kernels from our b12x fork, hand CUDA where it pays. Load any checkpoint of a
supported family directly from the HF snapshot. Say precisely what is missing
when something is not supported. Never slower than the engine it replaces.

Companion notes for agents: `AGENTS.md`. Code is king;
measurements are short tables in commit messages and `docs/` stays tiny.

## What we start from (survey 2026-09-28)

- `../ds41rt` is the base. Live path is the `v41_*` modules (serve, experts,
  target pass, attention, index, engram, vision, dspark) plus 7 crates:
  core, ffi, loader, transport, hostcache, api, daemon. ~2.2k Rust tests,
  ~60 qualification scripts, 16-lane parallel expert loader, RoCE verbs
  transport with TP2/3/4/6 × EP, dual-RTX TP2 local experts, GPU sampling,
  xgrammar constraints, live console, v15 images published.
- `ds41rt-daemon/src/commands/real_full/` (~167k LOC) is the legacy DS4
  path. Nothing live references it. It is not ported; DS4 Flash/Pro are
  re-hosted on the new engine instead (the dsv4 programs replaced the
  legacy `ds4_*_aot` and `packed_fp8_mla_exact` kernels, removed 2026-09-30).
- `../ds4rt` has no engine code ds41rt lacks. It contributes the GPTQModel
  distributed quantization pipeline, Pro K2 evidence, and three API defaults
  (thinking on, 32K output budget, hidden internal model names).
- `../glmrt` is a port, not a merge (~150 divergent Rust files): GLM model
  code (MLA + DSA indexer, top-8 sigmoid router, dense first layers, shared
  expert path, native MTP), `dsa_indexer.cu`, mixed EXL3 K3/K4 top-8 kernels
  and loader layouts, DFlash2 block draft engine, GLM XML tool grammar,
  MoonViT vision, FP8/NVFP4/BF16 KV profiles.
- Kernel library: `../sparkinfer-glmrt` (b12x fork, master `7fcc094`,
  224 ahead of upstream) is the only fork; ds41rt already pins its head and
  it contains everything ds4rt/glmrt pinned. Publish there.
- `../GPTQModel` fork `main`; ds41rt imports `gptqmodel.utils.v41_*` which
  must be located (submodule copy or unpushed branch) before Phase 4.
- xgrammar v0.2.3 is pristine upstream; all customization is engine-side
  adapter code. Vendor as submodule with lock, same as today.
- Storage: sparknest FUSE at `/mnt/sparknest/hf-home` on every host. Local
  sealed copies read at NVMe passthrough speed; missing copies stream over
  the fabric at ~5 GB/s. `nest where` / `nest replicate` place copies.
  No Rust data-path client exists; the engine reads through the mount.

## Target models

Core (parity tier, must match or beat the parent engines):

| Model | Family | Format | Notes |
| --- | --- | --- | --- |
| deepseek-ai/DeepSeek-V4.1-Flash | deepseek_v41 | FP8 block + MXFP4 experts | engram ×2, dSpark 3-stage, vision. The regression anchor. |
| wrldsuksgo2mars/GLM-5.3-EXL3-K4-v1 | glm5 | EXL3 K4 experts | 78 layers, MLA+DSA, DFlash2 draft (`incoai/GLM-5.3-DFlash2`) |
| deepseek-ai/DeepSeek-V4-Flash-0731 | deepseek_v4 | FP8 block + FP4 experts | compress 4/128 alternating, nextn 1 |
| wrldsuksgo2mars/DeepSeek-V4-Pro-0813-EXL3-K2-calibrated-v1 | deepseek_v4 | EXL3 K2 experts | 61 layers, 7168 wide, ~96 GiB per Spark rank at TP4 |

Extend (new families; kernels largely exist in b12x already):

| Model | Family | New pieces |
| --- | --- | --- |
| zai-org/GLM-5.3-Flash, brandonmusic/GLM-5.3-Flash-tr3-4bpw | glm5_flash | hybrid KDA linear attention (34) + DSA MLA (11), mHC, EXL3 tr3 |
| XiaomiMiMo/MiMo-V2-Flash | mimo_v2 | GQA full + SWA with sink, no shared expert, FP8 |
| XiaomiMiMo/MiMo-V2.6-Pro-MOPD | mimo_v2 | 70 layers, 128 heads, mxfp4 store dtype, dflash dir |
| Qwen/Qwen3.8-Flash-Next (+ EXL3 K4.25 PLE variants) | qwen4 | GDN linear attention, n-gram memory tables, PLE, MTP; `../qflashrt` has a single-device port |

Speculation: one best speculator per family (native MTP/nextn, dSpark,
DFlash2). Adaptive width with calibrated confidence and cost model, as in
ds41rt `dspark_policy` and glmrt `dflash2_confidence`.
Qwen's launcher defaults to MTP3 only for local EXL3 (`ranks == 0`);
TP4's shipped default uses copy-window drafts. Enabling native MTP on TP4
with coordinator-local draft experts is a separate policy question, not a
requirement of vision or decode-bucket qualification.
Qwen TP4's former 32,768-token KV pool admitted only seven fresh requests
at the qualified C16 output budget; scheduler and EXL3 slots were not the
limit. Full-width distinct-prompt qualification and default auto admission
are recorded under Qwen TP4 admission below.

## Architecture

```
rust/crates/
  cuteafd-core       ids, geometry, placement math, admission/lanes, KV allocator, sampling params
  cuteafd-ffi        libloading C ABI; one module per kernel family, family-namespaced symbols
  cuteafd-loader     checkpoint catalog, family readers (HF config -> ModelSpec), LoadPlan,
                 capability check, fast sliced readers, mapped tables, sparknest placement
  cuteafd-transport  ExpertProtocolV2, verbs RoCE, TP x EP topology (unchanged from ds41rt)
  cuteafd-hostcache  pinned host RAM prefix snapshots (unchanged)
  cuteafd-engine     model-agnostic serve runtime: scheduler, lanes, prefix cache, memory,
                 speculative transaction framework, console state
  cuteafd-api        OpenAI chat + completions, constraints, tools, images, console
  cuteafd-families/  deepseek_v41, deepseek_v4, glm5, glm5_flash, mimo_v2, qwen4
                 each: spec reader, block execution, attention variants, speculator wiring,
                 chat template + tool parser + grammar generator
  cuteafd-spec/      speculators: nextn_mtp, dspark, dflash2
  cuteafd-daemon     CLI: serve, expertd, plan, inspect, doctor, bench-*
native/
  shared/        norm, sampling_gpu, embedding, router/route_reduce, expert_pack, peer_copy,
                 kv, engram/mapped tables, verbs, xgrammar adapter
  families/      deepseek_v41/ (hc, compressor, index*, sparse_attention, dspark*, vision),
                 deepseek_v4/ (mla_indexing, ds4_*_aot), glm/ (dsa_indexer, mixed exl3),
                 mimo/, qwen/
  aot/           b12x export manifests per family x format x SM x TP role
python/          reference impls, exporters, qualification tools
quantization/    unified distributed EXL3 pipeline (ds41rt + ds4rt + glmrt), gptqmodel fork
third_party/     sparkinfer, xgrammar, gptqmodel, transformers (submodules + tree locks)
build.sh wip.sh run.sh stop.sh   kept; config gains MODEL + family auto-detect
```

Traits the engine programs against (keep them few and concrete):

Decode buckets must not straddle registered projection arithmetic thresholds.
Families use a named Rust threshold registry and a fail-closed startup check;
CPU contracts compare it with the pinned fork's routing rules at each pin bump.
Exporter-embedded, object-attested route registries are possible hardening if
pin bumps ever bypass those tests; the current AOT manifest does not carry them.

- `Family`: reads `config.json` + `quantization_config` + safetensors index
  into a `ModelSpec` (layer kinds, attention kind per layer, MoE geometry,
  speculator, mapped tables, vision) and builds the per-layer `Block`s.
- `Attention` per layer kind: CED compressed MLA (V4.1), alternating
  compressed MLA (V4), MLA + DSA indexer (GLM), GQA full / SWA + sink (MiMo),
  KDA / GDN linear (GLM Flash, Qwen). Each owns its KV layout and formats.
- `ExpertBackend`: Spark TP×EP over RoCE, local RTX TP1/TP2, dSpark-style
  drafter experts. Format-specific pack/unpack lives with the AOT shim.
- `Speculator`: propose, verify, commit/rollback inside the existing
  speculative transaction protocol.
- `MappedTable`: engram / n-gram memory / PLE tables in host RAM with bounded
  prefetch and GPU staging; later optionally Spark-resident.

Load plan and capability check (`cuteafd plan`):

- Runs without GPUs. Enumerates every tensor to owner (RTX0/RTX1/host-mapped/
  Spark rank slice), source format, kernel requirement (family, format,
  shape class, SM, TP role) and read plan.
- Checks against the kernel capability registry embedded in the built
  image (what AOT families and roles it carries) and against
  `nest where` for copy placement.
- Unsupported is a first-class result, never a crash: prints a hint block
  for a code agent naming the tensors, formats, shapes, the exporter or
  kernel to add, and the fast-read path missing (fallback: generic pread).
- Plan is the contract between coordinator and Sparks; every rank verifies
  the same plan hash at startup, as ds41rt does with `plan.json`.

Loading:

- Generalize the six-region TP read plan into a sliced-extent reader: given
  tensor shape, dtype/packing, slice axis and rank, produce coalesced extents;
  16-lane pinned staging, fadvise/O_DIRECT, per-rank local reads. No speed
  regression on V4.1 (readiness 31.7 s dual / 58.6 s single today).
- Before loading, the coordinator asks sparknest where copies live. Default
  reads through the mount wherever the file is. Optional `--place` runs
  `nest replicate` of the shards a rank needs onto that rank so repeated
  starts hit NVMe passthrough.

Fabric discovery (transport, at every startup, coordinator and ranks):

- Probe and log per host: each RDMA device and port (`/sys/class/infiniband/
  */ports/*/{rate,state}`), its netdev, negotiated speed and MTU, the rail
  addresses/GIDs and whether rails sit on isolated subnets, and the PCIe
  link generation and width of the NIC and of the GPU. Publish this in the
  startup plan so every rank sees the whole fabric picture.
- Choose the queue-pair strategy from that data, not from a constant:
  rail count, which rail carries request/response vs. reduction traffic,
  in-flight depth and chunk sizes. Lesson learned: dual rail at 100G links
  with 200+ Gb/s of inbound PCIe traffic caused head-of-line blocking, so a
  second rail is only used when link rate and PCIe ingress justify it.
  Only one physical configuration exists at a time, but the engine records
  the inputs and the chosen strategy so the choice can be revisited.
- Today: Sparks negotiate 200 Gb/s on two ports (rail A 10.55.0.x, rail B
  10.55.1.x, separate subnets); raptor has one 400 Gb/s port carrying both
  rail subnets. The switch is moving from 100G to 200G; verify the
  effective rate rather than trusting the port. NCCL is not required, so
  its isolated-subnet rule is informational only.

API and dashboard: keep ds41rt `native_v41` router; add `/v1/completions`;
per-family chat templates (deepseek-recipe crates for DeepSeek; minijinja
over `tokenizer_config.chat_template` as the generic path) with per-family
tool-call parsers and xgrammar tag grammars; console made model-agnostic.
Status (2026-10-02): the live console (`/`) is family-neutral: `shared/console` (producer
`Ticket`/`Step`/`layer_mark`, console thread, wire schema) fed by every serve loop; each family
declares its header, speculator, micro-step stages and layer classes in a `console::Layout`.
Shared page shell, palette and SVG chart primitives at `/assets/cuteafd-ui.{css,js}` (also used
by `/bench`); logo and favicon at `/assets/cuteafd-{logo,mark}.svg`.

## Execution engines (decided 2026-09-29)

V4.1's execution stack is specialized to its causal encoder/decoder: KV and
index sources [2,8,14,20], engram gates at layers 1 and 14, dSpark taps from
layer 37, encoder/replay stages. Other families share none of that, so there
are two engines under one scheduler, API, transport, host cache and sampler:

- `deepseek_v41` keeps its specialized path; nothing regresses it.
- A generic per-layer engine runs every other family: each layer is a block
  (norm/HC in, family attention with its own window/compressed/index/state
  cache, router, shared expert, routed experts on Sparks, HC/residual out),
  plus head and native speculator. DeepSeek V4 is its first tenant, then GLM,
  MiMo, Qwen. It adopts V4.1's proven optimizations (chained stages, lanes,
  dual RTX, route packing) as it matures.

Kernels are shared by parameterizing dimensions; `cuteafd plan` feeds the
AOT exporters the geometry each image must carry.

## Phases

Each phase ends with: V4.1 Flash parity table (C1 code decode 1/2 RTX,
8K prefill, tool eval) unchanged or better, plus the phase's own table.
Commit small, push often. Opus executes all phases; Fable wrote the specs.

**Phase 0 — skeleton and anchor.**
Import, purge, restructure, and prove V4.1 parity on the new tree before
any new model work. Concrete recipe:

1. Import: `git -C ../ds41rt archive 3067d06 | tar -x -C .` then remove
   `.gitmodules` and re-add the four submodules at the same commits:
   sparkinfer `7fcc094e` (tpurtell/sparkinfer-glmrt, master),
   xgrammar `557becfb` (mlc-ai, v0.2.3), gptqmodel `5340775d`
   (tpurtell/GPTQModel, main), transformers `62d7ebd7` (malaiwah fork).
   Keep the `*.lock.json` tree locks and `verify-*-source.py`.
2. Purge in the import commit: `docs/` (940 evidence files), `runs/`,
   `ds41rt.build-v*.config`, `scripts/render-*`, `scripts/summarize-*`,
   `scripts/bench/` campaign files, `assemble-release-v2.py`,
   `update-ds41-v6-release-docs.py`, `release_semantic_quality.py`,
   `release_throughput_checks.py`, `migrate-layer-boundary-*`,
   `scripts/fixtures/release-*`, the matching `scripts/tests/test_v*` and
   `test_*release*` tests, and the legacy TCP scripts
   (`real-full-*`, `real-slice-*`, `start-spark-experts-tcp.sh`,
   `phase0-*`). Drop `README.md`, `DEVELOPER.md`, `AGENT_DEV_HINTS.md`,
   `architecture.md` (replaced by `AGENTS.md` here). Keep `LICENSE`,
   `THIRD_PARTY_NOTICES.md`, `docker/`, `examples/configs/`,
   `build.sh`/`wip.sh`/`run.sh`/`stop.sh`, `justfile`, `quantization/`,
   `python/`, `native/`, `rust/`.
3. Delete `rust/crates/ds41rt-daemon/src/commands/real_full/` and the
   legacy CLI commands (`Coordinator`, `Expertd`, real-full benches), then
   the Python tools and fixtures that only they used
   (`validate_ds4_*`, `validate_native_flash_*`, `tune_w8a16_*`,
   `tune_mtp_*`, legacy sparse-lm-head tuners). Native `ds4_*_aot.cu`,
   `mla_indexing.cu`, `packed_fp8_mla_exact.cu` stayed for Phase 1 (the
   unbound ones went in the layout purge).
   Build must pass after this step.
4. Rename: every `ds41rt` token → `cuteafd` (crates `cuteafd-*`, binary `cuteafd`, native lib
   `libcuteafd_native`, symbol prefix `cuteafd_`, env/config prefix
   `CUTEAFD_`, image names `cuteafd-{coordinator,spark-expert}`, build
   cache `~/.cache/cuteafd/builds`). Mechanical, one commit.
5. Restructure: `native/{shared,families/deepseek_v41,families/deepseek_v4}`;
   `v41_*` daemon modules → `cuteafd-families/deepseek_v41`; carve
   `cuteafd-engine` (scheduler, lanes, prefix, memory, speculative transaction,
   console state) out of `v41_native_serve` behind the traits in
   Architecture. Do this incrementally with V4.1 serving between steps.
6. Generalize: `cuteafd.config` takes MODEL (hf id or path), REVISION,
   TOPOLOGY; the family reader replaces the embedded official config and
   model-id check with schema validation. Add `cuteafd plan` with the kernel
   capability registry and the unsupported-hint block. Add sparknest
   placement awareness (`nest where`, optional `--place`).
7. Serve V4.1 Flash on 2×RTX + 4 Sparks and on 1×RTX; record the parity
   table against ds41rt v15 in the commit message. Tag `p0`.

**Phase 1 — DeepSeek V4 family.**
`deepseek_v4` family on the new engine: V4 Flash 0731 (native FP8/FP4) and
V4 Pro EXL3 K2 (TP4, ~96 GiB/rank; consider TP6 across all six Sparks).
Reuse `ds4_*_aot`, `mla_indexing` kernels; nextn MTP speculator. Port the
three ds4rt API defaults. Targets: Pro ≥ 50 decode / 2,500 prefill tok/s
(ds4rt floor 33 / 1,650).

**Phase 2 — GLM 5.3 (glm5) + DFlash2.**
Port from glmrt: DSA indexer kernel, top-8 router, dense/shared paths,
mixed EXL3 K3/K4 routes and loader layout, DFlash2 speculator, GLM tool
grammar, KV profiles. Target: ≥ glmrt's 25.96 weighted tok/s on 1 RTX,
better on 2 RTX with local expert layers.

**Phase 3 — new families.**
`glm5_flash` (GLM 5.3 Flash: KDA + DSA, mHC; b12x `kda_prefill`/`gdn_decode`),
`mimo_v2` (GQA + SWA sink; b12x paged FP8 KV attention), `qwen4`
(from `../qflashrt`: GDN, n-gram tables via `MappedTable`, PLE, MTP).
Each lands with a `plan` that says what is missing before any kernel work.
Finish downloads/placement first (MiMo checkpoints are incomplete today).

**Phase 4 — quantization and release.**
Unify `quantization/` (ds41rt K3.25/FP4PLE, ds4rt Pro K2, glmrt K3.25
mixed) on the gptqmodel fork; extend to new families; publish quants under
wrldsuksgo2mars. Official images `ghcr.io/tpurtell/cuteafd-{coordinator,
spark-expert}`, concise README with one headline table.

**Prefix cache for every family (top priority, design 2026-09-30).** Only
V4.1 has one (radix banks Prompt/Turn keyed by token ids, shared FP4 pages +
a 2.72 MB copied "front" of 128 SWA rows per layer, 128-token approximate
replay for partial hits, pinned-host tier in `cuteafd-hostcache`). Generic
version in the engine crate: `PrefixFamily` trait (layout, capture/restore of
the positional "mark", shareable page rows, commit point), one refcounted
`RefPagePool` with CoW tails replacing the five per-family free lists, a
device mark arena sized by lanes (recurrent marks are 110–141 MiB) with the
host tier holding the rest, Hugh-style victim order and 64-token hash-chained
host page identity, `After{greedy, logits}` for exact-length hits, drafters
restored cold with a `context_valid_from` mask. Prompt snapshot is the
guarantee; Turn snapshot opportunistic, later made exact by a template-aware
canonical turn snapshot. Order: S0 engine crate + agentic reasoning-on
benchmark (record/replay of multi-turn tool sessions; turn TTFT, hit rate,
decode tok/s), S1 MiMo V2.6 Pro (+V2 Flash), S2 GLM 5.3 Flash, S3 GLM 5.3,
S4 Qwen, S5 V4 Flash/Pro, S6 canonical turn snapshot, S7 embedding cache.
Gate: resume-at-P restores are byte-identical to straight prefill; greedy
text identical to cache-off; V4.1 parity unchanged.
Status: S0-S5 merged (S2 GLM 5.3 Flash: 256-row units = 4 MLA pages + the
pool page, 140.8 MiB KDA mark, kda_len commit point; S3 GLM 5.3: pages only,
host tier via a stand-in tail; S4 Qwen 3.8: the same 256-row units over 12
full-attention layers, 110.3 MiB GDN+PLE mark, state_len commit point, n-gram
history recomputed from the ids; S5 V4 Flash/Pro: 256-token units = the C4, index and C128 pages
of one index, 27.9 / 42.2 MB mark = every layer's last 128 window rows, dSpark rings included
(drafts warm), plus the compressors' FP32 rolling state; commit point = placement length), GPU-gated
with experts skipped; S5's live agentic gate ran on Sparks (V4 Flash TP4 + dSpark, one
session: 8/8 turns reuse the whole previous turn, later-turn hit ratio 0.93, TTFT 0.83 -> 0.20 s,
task solved); the others next (Qwen: one live session with local EXL3 experts on one RTX). The DSA index top-k (b12x `tiled_topk`: GLM 5.x, GLM 5.3
Flash, V4 Flash/Pro) is deterministic: ties go to the lower index and picks
come out in ascending order, so `--resume-at` is byte-exact past 2048 tokens
too. Still arrival-ordered: the fused decode route (`persistent_topk`, used
only by plans of 16 rows or fewer, not by the exported m64 programs).

**Repo layout and naming (study 2026-09-30; run at a quiet point after a merge
round, before prefix-cache S1 and planner S0 create new modules).** Family ids:
long ids `deepseek_v41`, `deepseek_v4`, `glm5`, `glm5_flash`, `mimo_v2`,
`qwen4` for directories, modules, plan ids, goldens; short tags `v41`,
`dsv4`(f/p), `glm`, `glmf`, `mimo`(p), `qwen4` only for C symbols, AOT
program prefixes and package dirs (baked into manifests; unchanged). Target:
`shared/` + `families/<id>/` in the daemon, ffi and loader; transport
`v41_expert` → `expert`; api `native_v41` → `openai` + `chat/<id>`;
`native/{shared,families/<id>}/{cuda,src,include}` + `cmake/{shared,families}`;
`python/reference/families/<id>`, `python/tools/{aot,bench,hf,qualify/<id>}`;
`scripts/{lib,build,launch,bench/<id>,qualify/<id>}`; multimodal input path in
the engine crate, per-family encoders beside their family. Steps: P1 purge
dead legacy (≈6.4k lines of unbound ds4/b12x/w8a16 kernels, python runtime,
stale justfile recipes, unused core modules), P2 baselines; M1–M9 pure
`git mv` commits with only forced build-file edits and root re-exports
(`crate::v41_*` paths keep compiling), verified per group, plus
`path-map.tsv` and a rebase-across-move script; then a naming pass (generic
types off V4.1 names, one `serve`/`expertd`/`golden` CLI with family
detection, `run.sh` absorbs `run-dsv4.sh`, cmake/config/env aliases for one
release; shared C symbols and headers drop `v41` with no aliases, since the
library and binary ship together; V4.1-specific names stay). ≈3 days move pass,
≈3–4 days naming pass; V4.1 parity before tagging.
Status: purge, move pass (M1–M8) and naming pass (N1–N9) done on
`work/restructure`, N10 (shared C symbols) on `work/p0`;
`scripts/build/path-map.tsv`, `rename-map.tsv` and `rebase-across-move.sh`
carry older branches across. Full native AOT build,
V4.1 parity vs 9015bff, GLM 5.3 golden NLL and a MiMo launch checked on
hardware before the merge into `work/p0`; M9 (fork layout) separately.

**Phase 5 — NVIDIA ModelOpt NVFP4 checkpoints (queued; after the families
above reach their performance targets).** nvidia/{DeepSeek-V4.1-Flash,
GLM-5.3, GLM-5.3-Flash, Qwen3.8-Flash-Next}-NVFP4. Common contract: U8
`[N,K/2]` E2M1 low nibble first, E4M3 per-16 scales in linear layout,
F32 `weight_scale_2`, static `input_scale` (W4A4 spec); routed experts are
FP4, most other weights BF16/FP8 as released. Design (study 2026-09-30):
- One `QuantOperand` descriptor in the loader, read from
  `hf_quant_config.json`/`config.json` by one ModelOpt reader; model code
  and engines never see the format.
- Routed experts: `ExpertFormat::Nvfp4` in the existing `fp8-<geom>`
  package (ABI word 3), W4A16 by default (E2M1×E4M3 is exact in BF16, so
  it reproduces NVIDIA's weights with unquantized activations); scale
  swizzle, when a kernel wants it, runs on the GPU at load
  (`nvfp4_scale.cu`, parameterized); no offline repack.
- Dense NVFP4/per-tensor-FP8 parts dequantize to BF16 at load today;
  item 10 of the v1 plan replaces that with compact consumers.
- V4.1: its NVFP4 release uses the native W4A4 44-slot family by default,
  with BF16 input rows quantized per route to E2M1 + E4M3 K16 on the Sparks.
  FC1 currently replaces per-expert static input_scale with the layer maximum;
  FC2 keeps the expert's scale. The earlier W4A8-downcast description was stale.
- W4A4 prefill (MmaMXF4NVF4Op, in-kernel per-16 quantization) only if it
  measures faster and stays within 0.005 nats KL of W4A16.
Stages: S0 loader + `plan`; S1 W4A16 experts (GLM 5.3 Flash first, then
Qwen on one RTX: 68 GB); S2 dense + MTP dispositions; S3 V4.1
convergence; S4 W4A4 prefill experiment. Gates per stage: oracle cosine,
KL vs golden within 0.005 of the FP8-expert path, tok/s ≥ it, readiness
not worse, V4.1 parity. GLM 5.3 NVFP4 experts (~407 GB) need TP6.
Status (2026-10-02): S0, S1 and the S4 kernels landed (`work/nvfp4`, fork
`cuteafd/nvfp4-w4a16`). The ModelOpt reader (`formats/modelopt.rs`) checks
every weight against hf_quant_config.json / config.json. `glm|glmf|qwen4:nvfp4`
build `fp8-<family>-nvfp4` packages (W4A16: GEMV, stream above 2048 rows);
`:nvfp4a4` builds W4A4 large-row steps (static input_scale, mxf4nvf4 MMAs,
above 512 rows); every `:nvfp4` entry also builds it and W4A4 is the default
(`CUTEAFD_NVFP4_ACTIVATIONS=a16` keeps W4A16). Native SM121
packages pass the CPU oracle on GB10 at tp2/3/4/6. GLM 5.3 Flash TP4 Sparks
(EXL3 K3.25 coordinator weights): NVFP4 W4A16 NLL 2.3880 / KL 0.0587 (EXL3
K3.25 2.4082 / 0.0616), C1 step 16.3 ms vs 15.3 (4.5 vs 3.25 bits read), 8K
prefill equal; W4A4 KL 0.0799, prefill 1.33x. Qwen on one RTX: NVFP4 8K
prefill 1.08x EXL3 (W4A4 1.36x), 1.6K 0.96x (W4A4 1.33x). W4A4 costs
+0.02 KL (over this plan's 0.005 bound) but is the checkpoint's calibrated
numerics: W4A4 is the default (TJ, 2026-10-02). S2: GLM 5.3 Flash NVFP4 dense MLPs run natively (one-expert
`fp8-glmfdense-nvfp4`): nvidia/GLM-5.3-Flash-NVFP4 serves alone (NLL 2.3896);
nvidia/GLM-5.3-NVFP4 serves on six Sparks (TP6 W4A16: NLL 2.4814 / KL 0.0571
vs golden 2.4677; 8K prefill 4.22 s, W4A4 3.20 s). Its per-tensor FP8 dense
MLPs prefill as static W8A8 on their own input_scale / weight_scale (plain
E4M3 MMAs; decode GEMVs read the same bytes under a uniform grid): NLL 2.4827 /
KL 0.0580, MLP 1.4x faster than block W8A8 at 4096 rows. Its BF16 attention,
indexer and shared experts quantize to FP8 blocks at load by default; with
`CUTEAFD_GLM_BF16=native` they run as-is on the BF16 programs (one or two
RTX): KL 0.0491, but C1 step 41.7 vs 34.3 ms with the Sparks (coordinator
alone 28.6 vs 19.2 ms one RTX, 19.8 vs 16.1 two), coordinator 8K prefill
3.35 vs 2.75 s (2.36 vs 2.07), weights 32.3 vs 17.6 GiB. FP8 blocks stay the
default (TJ, 2026-10-02): the official zai-org GLM 5.3 ships these tensors as
128x128 block FP8, so the load-time conversion matches the official format. W4A4 gate/up + SwiGLU + FP4 quant run fused (bit-exact;
layer 7-17% faster, GB10 13-17%). Open: W4A16 stream efficiency (GB10 4096
rows 14.3 ms/layer TP4 vs EXL3 9.1), SM121 route thresholds.

**Phase 6 — placement planner (design 2026-09-30).** One planner for every
family: (model, inventory of 1–2 coordinator GPUs — real or simulated by a
memory budget — and 1–8 Sparks, KV target, objective) → a hashed
`placement.json` that the loader, workers, engines and launchers all
consume. It generalizes V4.1's startup handoff (`v41_native_serve/placement.rs`),
live memory planner and 20/20 backbone split. Per MoE layer the experts are
resident on GPU0/GPU1 (full width, or TP2 where measured) or on a Spark group
with TP×EP and uneven whole-block slices; EP means expert subsets per group
(new: today groups are replicated). Cost model counts the busiest rank
(E[max] routed experts per group), fabric intake as a shared prefill
resource, and per-family coordinator step tables; search is exhaustive.
Cold components get dispositions: official vision/audio towers run whole on
a Spark when they fit (`ENCODERS=rtx0` pins them) with a hash-keyed
embedding cache refcounted by the prefix cache; MTP layers are unused when a
DFlash2 drafter drafts. Dual-GPU coordinator default is the layer-range split
(memory, ~1.9× GPU-bound prefill); TP2 of dense layers only if a P2P probe
shows it pays (GPU0/GPU1 cross the host bridge; ds41rt measured it a loss).
Evaluate per family, not globally: GQA models with many KV heads (MiMo V2.6
Pro: 128 q × 192, 8 KV heads, 16K-wide o_proj, ~18 GB streamed per token)
can head-split attention across two GPUs with KV partitioned (4+4 KV heads,
no replication) and one hidden all-reduce per layer — a possible C1 win the
DeepSeek MLA models never showed (ds41rt's V4.1 head split with replicated KV
measured −5…−8%; its attention weights are small next to hop and launch
costs). GLM 5.3 is the other candidate: MLA, but ~205 MiB of coordinator
weights per layer (o_proj 96, q_b 32, kv_b 14, shared expert 36), ~16 GB
streamed per token, and a small replicated latent (656 B/token/layer). DCP2
(KV split by sequence) is a capacity-only option and is not needed for V4.1
(compressed KV, 14M-token default pool). Order: P2P probe, then head-split vs
layer-range A/B for MiMo Pro and GLM 5.3.
Status (2026-10-01): `cuteafd fabric --p2p` measured GPU0<->GPU1 (NODE) hops of
3.3 us for 12 KiB (SM push + release flag, graph), a two-way exchange of 3.4 /
4.8 / 25 us for 12 KiB / 96 KiB / 1 MiB and 1.1 ms for a 48 MiB prefill chunk
(copy engine 0.9 ms); saturating host->GPU0 ingress roughly quadruples small
hops. So the head split pays and is the default with two RTX
(`--split-device`, run-family `RTX_GPUS`/`COORDINATOR_GPUS`, `COORDINATOR_SPLIT=off`
opts out): MiMo V2.6 Pro (coordinator-only decode -41%, 8K prefill -43%; with
6 Sparks C1 decode 30.9 -> 26.0 ms, prefill Spark-bound) and GLM 5.3
(coordinator-only 8K prefill -28%; with 6 Sparks decode -9.5%, prefill
Spark-bound) and DeepSeek V4 (4 Sparks, all experts remote: Flash decode -10%,
Pro decode -12%, prefill neutral). Shared plumbing in `shared/peer_split.rs`: per-slot release flags,
partials exchanged and summed in the same operand order on both GPUs (identical
residual streams), GPU1 queued a layer ahead of GPU0's Spark exchange, decode
graphs captured per GPU. The layer-range split is not needed for these three. V4.1 Flash re-measured on p8 images (2 RTX + Sparks, code,
dSpark on): its TP2 modes still lose (C1 187 -> 170 tok/s for TP2_ATTENTION and for
TP2 q+o projections; C4 501 -> 466 / 482). Each projection there ends in a
cross-device event and a host wait, so a gain needs V4.1's per-layer flow rebuilt
around device-side flags. The attention-only ceiling is about the DeepSeek V4 Flash
split (-10%, all experts remote), and less with RTX-resident expert layers, so
V4.1 keeps its layer split. Measured 2026-10-02 (C1 code, 6-row verify rounds of
~27 ms): each RTX is busy ~33% (C4 ~47%), the head-splittable work (q_b, sparse
core, wo_a, wo_b) is ~115 us of a ~170 us attention block per layer, and a 32-head
core saves only 9 of 24 us. With KV replicas, peer inputs and the exchange, a
V4-style split projects +2-4% C1, ~0-2% C4, less at C16, and the replicated 14M-token
pool (+5-8 GB per card) costs two RTX expert layers at the release config: not built.
V4.1's lever is its host-driven layer. Per layer at C1 (5-row verify, nsys): remote
(20 layers) 789 us = GPU to route ids 118, route D2H to host 12, Spark round trip and
collect 544, combine to next attention 99 (55 of kernels, the rest host launch
gaps); local TP2 (20) 450 us = 120, route to expert launch 35, experts 164, host-run
TP2 reduce 31, next attention 96. C4 lanes: remote 1086 (round trip 782), local 673.
A device-driven exchange (GPU-written requests + proxy post, GPU-landed replies with
a NIC-written flag, device wait, device-flag TP2 reduce) projects C1 +5-7%, C4
+3-6%; also capturing the whole verify step (no launch gaps) C1 +10-15%, C4 +8-12%.
Unknown: the host share inside the Spark round trip (worker-side timing needed).
Outside the layer loop: ~0.8 ms host gap per round after argmax, and two BF16
vocab heads (target, draft) of 450 us each.
The drafter follows the GPU that owns the last backbone layers (taps and head
live there); TP2 drafters are ≤1% on DFlash2 and not built unless the P2P
probe shows ≤15 µs hops; the win is lane B drafting on GPU1 while lane A
verifies on GPU0 at C≥2. Benchmark only the natural minimum and maximum
configs (AGENTS.md); the planner's estimates cover the rest.
One `ExpertRouter` replaces the six per-family stage/send/land/reduce copies
and is the device-driven Spark exchange below (V4.1 moves onto it too). Stages: S0 planner + `plan` (must
reproduce today's layouts), S1 manifest handoff + workers, S2 router in the
generic engines + GPU1 as expert host, S3 EP subsets (only if a quantized
model needs them; GLM 5.3 official FP8 is out of scope — EXL3 and NVFP4
quants cover it), S4 encoder service + multimodal input, S5 coordinator
range split, S6 eight Sparks.

**Memory audit and planner core (2026-10-03, `work/v1-memory`).** Every
device, pinned and RDMA allocation now goes through a process-wide ledger
(`cuteafd_ffi::memory_ledger`: thread-local category scopes, the checkpoint
tensor and resident format being uploaded; `cuteafd::memory` log reports;
`scripts/bench/memory-audit.py` tabulates and `--compare`s against the
planner). One launch per config (codex/v1 tree + ledger, after an 8K prefill,
a C4 and a C1 request; default pools). GiB per device:

| config | device | weights | emb | drafter | KV | marks | workspace | exchange | experts | runtime | used | free |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MiMo V2.6 Pro 2 RTX + 6 | GPU0 | 10.72 | 1.75 | 3.21 | 2.36 | 0.77 | 2.49 | 0.38 | | 1.37 | 23.0 | 71.9 |
| | GPU1 | 9.52 | | | 2.34 | 0.77 | 1.20 | 0.38 | | 1.00 | 15.2 | 79.7 |
| | Spark (TP6) | | | | | | 0.52 | 1.15 rings | 92.79 | 7.5 OS + 11.0 cache | 117.6 | 4.0 |
| MiMo V2.6 Pro 1 RTX + 6 | GPU0 | 20.24 | 1.75 | 3.21 | 4.70 | 1.54 | 2.90 | | | 1.23 | 35.6 | 59.4 |
| GLM 5.3 K4 2 RTX + 6 | GPU0 | 10.48 | 1.77 | 5.88 | 13.18 | | 5.59 | 0.56 | | 1.72 | 39.2 | 55.8 |
| | GPU1 | 8.49 | | | 13.18 | | 4.22 | 0.56 | | 1.26 | 27.7 | 67.3 |
| | Spark (TP6) | | | | | | 1.9 | 2.29 rings | 63.45 | 7.7 OS + 7.1 cache | 85.4 | 36.2 |
| GLM 5.3 K4 1 RTX + 4 | GPU0 | 17.56 | 1.77 | 5.88 | 13.18 | | 6.51 | | | 1.55 | 46.5 | 48.5 |
| | Spark (TP4) | | | | | | 1.9 | 2.29 rings | 84.59 | 7.7 OS + 3.4 cache | 103.3 | 18.3 |
| GLM 5.3 Flash 1 RTX + 2 / + 4 | GPU0 | 13.07 | 1.18 | 3.24 | 2.12 | 2.47 | 4.72 | | | 1.24 | 28.0 | 66.9 |
| | Spark (TP2 / TP4) | | | | | | 0.56 | 0.77 rings | 58.3 / 29.4 | 7.3 OS | 78.1 / 43.6 | 43.6 / 78.0 |
| V4.1 Flash 2 RTX + 4 | GPU0 / GPU1 | 5.16 / 3.91 | in weights | 0 / 8.12 | 7.34 / 4.90 | 0.1 | 9.48 / 5.70 | | ~67 each (20 layers TP2) | 2.0 / 1.9 | 93.1 / 93.5 | 1.8 / 1.5 |
| | Spark (TP4) | | | | | | 0.21 | 0.25 rings | 37.35 (20 layers) | 7.3 OS | 50.6 | 71.1 |
| V4.1 Flash 1 RTX + 4 | GPU0 | 9.29 | in weights | 8.21 | 15.60 | 0.1 | 21.08 | | 33.94 (5 layers) | 2.83 | 93.1 | 1.9 |
| | Spark (TP4) | | | | | | 0.21 | 0.48 rings | 65.37 (35 layers) | 7.4 OS | 79.2 | 42.4 |
| V4 Flash 2 RTX + 4 | GPU0 / GPU1 | 5.38 / 3.74 | 0.99 | | 2.08 / 1.96 | 1.0 / 1.0 | 4.14 / 3.55 | 0.25 / 0.31 | 73.47 / 0 | 1.0 / 0.9 | 88.3 / 11.4 | 6.7 / 83.6 |
| Qwen 3.8 EXL3 1 RTX | GPU0 | 8.21 | 1.18 | | 1.93 | 1.94 | 1.20 | | 62.61 | 1.41 | 78.5 | 16.5 |

Findings, against the suspects: no tensor is resident in two formats in any
family after codex/v1 (the ledger checks every upload by full tensor name);
no load-time conversion uploads at source size (GLM NVFP4 was the one case);
V4.1 Sparks hold only the remote layers (runtime placement handoff: 35 of 40
at one RTX, 20 at two). The waste is elsewhere:
1. Idle coordinator memory: the generic families' fixed pools (MiMo 131072,
   GLM 262144, GLM Flash 65536 tokens) leave 48–80 GiB of every GPU unused,
   while V4.1 fills its GPUs with expert layers and KV. Fixed for MiMo
   (`POOL_TOKENS=auto`, now the run-family default, through the codex capacity
   contract: 131072 -> 2,097,152 tokens, GPU0 23.0 -> 49.4 GiB, C1/C4/8K prefill
   unchanged) and GLM 5.3 Flash (default auto via `planned_pool_tokens`: 65536 ->
   2,097,152 tokens, 44 GiB still free, unchanged speed). GLM 5.3 takes
   `POOL_TOKENS=auto` (262144 -> 1,292,672 tokens, 65 GiB KV per GPU) but keeps
   its fixed default until item 5 is bounded (2.9 GiB left on GPU0 after a bench).
2. Spark page cache: ~10 GiB of the checkpoint stays cached per Spark after
   loading (CUDA free 4.0 of 121.6 GiB on MiMo Pro TP6); the worker's own
   fadvise does not reach sparknest's passthrough pages. Fixed: launchers
   drop Spark caches once every rank is resident (+9.5 GiB CUDA free).
3. Retained load staging: the copy_h2d pinned buffer stayed at the largest
   upload (0.45 GiB per Spark on MiMo, 1.77 GiB pinned on raptor for GLM,
   1.0 for V4 Flash). Fixed: released after loading.
4. Head-split workspaces sized for all heads: GLM 5.3 6.39/5.02 -> 5.59/4.22
   GiB, V4 Flash rank 1 3.80 -> 3.55; V4 Flash prefill logits for 4096 rows
   (2.1 GB) -> 64 rows with chunked golden downloads: GPU0 6.09 -> 4.14 GiB.
   DeepSeek V4 golden output identical; GLM 5.3 golden NLL 2.4686 in both (TP6 + head split; KL 0.03444, top-1 91.54%).
   GLM 5.3 2 RTX + 6 Sparks, 4 interleaved launches per arm: C1 54.3 vs 55.6
   (-2.4%, t=-1.6), C4 87.3 vs 88.2 (-1.1%, t=-1.1), 8K prefill TTFT medians
   3.454-3.58 vs 3.45-3.52 s: within run-to-run noise (C1 spans 50-58 in both).
5. Graph executables (untracked): GLM 5.3 grows from 0.89 to 2.62 GiB on
   GPU0 over one C1/C4 + prefill bench and keeps rising (graphs per layer x
   exact row count x table width); an auto pool sized with a 0.6 GiB graph
   reserve hit cudaGraphInstantiate OOM and poisoned the context. The planner
   reserves 3 GiB per GPU for GLM until the cache is bounded (bucket rows or
   cap entries; MiMo's codex plan bounds its own at 616/566 MiB).
Inventory for the planner (not fixed; bytes per device):
- Spark slices padded to the widest 128-row block: MiMo V2.6 Pro TP6 stores
  384 of 352/320 rows (7.7 GiB on ranks 0-3, 15.5 on 4-5, 62 GiB cluster);
  GLM 5.3 EXL3 TP6 384 of 341 (~7 GiB/rank); V4.1 TP4 640 of 576 (~3.7 GiB
  at 20 layers, 6.5 at 35). Uneven whole-block slices (384,384,384,384,256,
  256) free ranks 4-5 at no speed cost; kernels that run 32-row tails also
  cut the critical rank's rows 8-10% (MiMo Pro prefill is Spark-bound).
- sparknestd holds 7.2 GiB RSS on every Spark (host OS total ~13 GiB idle).
- RDMA rings: 1.15-2.31 GiB per Spark (depth 8 x 8 MiB slots per endpoint)
  and 6.9 (MiMo) / 13.7 (GLM) GiB pinned on raptor.
- GLM 5.3 head split replicates the MLA latent KV (13.18 GiB per GPU at 262K
  tokens) and 1.2 GiB of attention operands plus the indexer on GPU1.
- V4 Flash/Pro keep RTX-local expert layers on GPU0 only: GPU1 83.6 GiB free.
- V4.1 one RTX: decode and prefill target passes own complete workspaces
  (prefill pass 10.5 + backbone lanes 9.0 GiB) ~ three RTX expert layers.
- MiMo FP8 scales expanded per row twice (row and K-major): ~1 GiB.
- GLM DFlash2 drafter resident BF16 (5.88 GiB incl. 1.3 GiB buffers; the
  explicit FP8 representation is ~2.3 GiB smaller).
Planner core (S0): `cuteafd plan MODEL --layout [--rtx 1|2] [--pool-tokens N]`
lays every device out (weights by group and resident format, embedding,
drafter, KV records and state, prefix marks, workspaces, peer exchange,
runtime and graph allowance, Spark experts with padding, workspace and
rings) and sizes the pool from the tightest KV-owning GPU; MiMo weights come
from the codex resident layout, GLM Flash from its FP8-snapshot conversion.
Per-family costs are calibrated from the ledger (`plan::layout::family_costs`);
device totals at ready match the ledger within 2% (GLM 5.3 39.10/27.63 vs
39.18/27.70, MiMo Pro 22.81/14.99 vs 23.03/15.23, GLM Flash 27.55 vs 28.05), and
on the held-out auto-pool launches (MiMo Pro 49.42/41.86 vs 49.4/41.6, GLM Flash
50.93 vs 50.7, GLM 5.3 93.04/81.96 vs 92.1/80.6 with its full graph allowance).
Engines take admission from it: MiMo (capacity contract, pool 0 = auto),
GLM 5.3 and GLM 5.3 Flash (`--pool-tokens 0`). V4 Flash/Pro, V4.1 Flash
and Qwen 3.8 Flash Next now describe compressed/sparse KV, recurrent state,
mapped-table staging, prefix arenas, RTX expert arenas, native drafters,
workspaces and Spark ranks, with opt-in runtime admission on pool 0.
V4 Flash native TP2, V4.1 native TP4 and Qwen local EXL3 K4.25 natural-minimum
layouts are ledger-qualified after 8K prefill and C4; other configurations remain
estimates. Serving defaults are unchanged, including the omitted V4.1 pool
path. Next: graph-cache bounds and `placement.json` handoff (S1).
Follow-ups (2026-10-03, measurements pending in `~/.cache/cuteafd/builds/v1-memory/kit/out`):
- GLM 5.3 decode graphs bounded: steps pad to row buckets (exact to 16, then
  20..64) over a scratch page past the pool, page tables to power-of-two widths
  >= 16 pages; every shape captured at startup (2 RTX + 6 Sparks: 22,608
  graphs, 1.96 / 1.80 GiB per GPU, 5.8 s); real rows' logits byte-identical to
  unpadded steps; a failed capture runs its segment uncaptured (keeps the peer
  exchange in step). Measured vs base (2 launches each, 4 batches): C1 54.5 vs
  51.8 tok/s (no per-request captures), C4 85.8 vs 84.2, 8K prefill unchanged;
  untracked memory now flat after startup. GLM 5.3 POOL_TOKENS defaults to auto:
  262144 -> 1,292,672 tokens (65 GiB KV per GPU), 2.7 GiB left on GPU0 after the
  bench, C1 53.3 / C4 85.1 / prefill 2853 tok/s.
- V4.1 quick parity for the ledger (1 RTX + 4 Sparks, code, warm): C1 155.7 ->
  154.7, C16 1049.6 -> 1054.8, 8K prefill 6160 -> 6331 tok/s: no cost; tagging
  stays per allocation.
- Exact Spark slices: FP8/MXFP4/NVFP4 Spark packages also build tp<n>-w<width>
  layouts (ranks own whole 128-row blocks, no zero padding; MiMo V2.6 Pro TP6
  ranks 4-5 61.9 instead of 92.8 GiB); EXL3 already had them. Measured MiMo V2.6
  Pro 2 RTX + 6 Sparks, exact vs padded: rank 5 free 11.8 -> 43.2 GiB (rank 0
  unchanged), golden NLL 2.4150 -> 2.4088 (KL 0.0457 -> 0.0445; rank partials
  partition the rows differently), engine 8K prefill 3095 -> 3002 tok/s, served
  8K 2684 -> 2725, C1 63.9-69.1 -> 68.5-69.5, C4 102-108 both: neutral. Shortening the
  busiest rank (352/320 rows) needs 32-row tails in three MXFP4 kernels (fork
  master now has `work/mimo-perf`'s A8 down): the decode GEMV (`GroupedMxfp4Gemv`
  needs K % (128 x warps) for down), the BF16 stream down (`I % 128`, 128-K
  weight blocks) and the A8 stream down (128-K blocks via cp.async, u32 scale
  loads). Gate/up already tiles I in 32 rows (11 vs 12 CTA columns: -8%); down
  only gains if its last K block is predicated at 32 (TMA zero-fill covers the
  BF16 route's loads; the A8 route needs predicated cp.async), and scale rows
  of I/32 = 11 bytes need padding to 12 in the package layout. Expected: busiest
  rank -5..-8% expert time (MiMo Pro prefill is Spark-bound) and -7.7 GiB on
  ranks 0-3. V4.1 TP4 (576 -> 640)
  goes through the V4.1 packer: not done.
- V4.1 one RTX: row buffers at the live 2048-row chunk instead of the 4096 AOT
  capacity (as on two RTX), reindex selection shares the source's scratch:
  workspaces 21.08 -> 11.01 GiB, RTX expert layers 5 -> 6 (6.0 GiB still free),
  C1 154.7 -> 159.1, C16 1054.8 -> 1086.3, 8K prefill 6331 -> 6326 tok/s. Left:
  window-wave temporaries (3.5 GiB, per-layer streams; would make 7 layers) and
  engram gate sharing (0.6 GiB).

**Device-driven Spark exchange (decided 2026-10-02, `work/v41-device`).** No engine is
device-routed toward the Sparks today: every family (V4, GLM, GLM Flash, MiMo, Qwen) downloads
route ids, weights and wire rows per MoE layer, synchronizes the stream, builds the request on
the host, posts the RDMA sends, polls the CQ and only then queues the reduce; only RTX-local
expert layers read device routes (V4 `local.rs`). V4.1 has its own exchange (`NativeTp4Wave`,
`receive_owned` with per-frame H2D uploads, host ownership planner) plus host waits inside its
TP2 RTX expert layers (`chain::settle`, peer-copy `wait()`s). Measured on V4.1 (p8, 2 RTX + 4
Sparks, C1 code, nsys): a ~29 ms verify round keeps some GPU busy only 14.7 ms; the RTX-expert
layers 0-18 leave ~2.5 ms with both GPUs idle (host hops between attention, routes, TP2 slices,
peer reduce), the Spark layers ~10.9 ms (21 x ~470 us Spark round trips plus ~1 ms of host
hops). Decision: retrofit, not a port. V4.1 is not served through the V4 engine (no shared
attention, cache, encoder, engram or dSpark code; the V4 engine's Spark path is just as
host-driven); instead one shared exchange is built and both engines move onto it:
- GPU side: the router's ids, weights and wire rows are copied into a pinned, device-mapped
  mailbox and a kernel publishes a sequence number (release, system scope); the combine waits on
  the proxy's completion sequence with an acquire spin (`peer_exchange.cu`'s graph-safe pattern:
  sequences live in device memory, so replays advance them). Replies land GPU-direct in the
  intake planes (dma-buf), so a decode/verify step can be queued, and later captured, whole.
- Host side: one proxy thread per transport (the prefill lane pattern) spins on the mailbox,
  builds the request from it, posts the sends, polls the CQ, validates, and publishes
  completion; no inference-thread sync, D2H parse or launch on the critical path. GPU-initiated
  doorbells (IBGDA via mlx5dv) only if measured to pay over the proxy.
- Stages: D0 shared component (`shared/spark_intake` + `cuteafd-transport` device lane, native
  signal/wait kernels); D1 DeepSeek V4 decode/verify on it (opt-in `CUTEAFD_SPARK_DEVICE=1`),
  measured against the host path; D2 V4.1 Spark layers adopt SparkIntake/GPU landing and the
  device exchange, ownership choice on the device or precomputed; D3 V4.1 TP2 RTX expert layers
  without host waits (device-ordered peer copies and reduce); D4 whole-step graph capture
  (decode/verify), head split re-checked; D5 prefill; D6 other families adopt it. The Spark
  worker (`shared/experts/service.rs`) is host-driven too (CQ, launch, sync, send): its per-wave
  host overhead is recorded and a device-driven worker proposed if material. Gates per stage:
  golden NLL / byte-exact greedy vs the host path, prefix-cache restore exactness, quick C1/C4
  A/B on 1 RTX + 4 Sparks and 2 RTX + 4 Sparks, full V4.1 parity before proposing a default;
  the default stays byte-identical while off.
  MiMo (2026-10-02): decode/verify already run as captured per-layer segments between exchanges
  (`--decode-graphs`, opt-in): byte-exact, but neutral against the host exchange (the GPU bounds
  each segment), so D6 for MiMo is capturing those segments back to back.

Status (2026-10-02, `work/v41-device`, all opt-in): D0 done (`SparkDeviceLane` proxy +
`cuteafd_host_signal`/`cuteafd_peer_wait`; proxy spins only while announced waves are
outstanding, parks otherwise: idle 0.6% of a core, wake 5 us). D1 V4 Flash
(`CUTEAFD_SPARK_DEVICE=1`): top-1 100%, decode 13.2 -> 13.1 ms/tok, 6-row verify 4.7 -> 4.5
ms/tok. D2+D3 V4.1 (`CUTEAFD_V41_DEVICE=1`: device-ordered verify passes, TP2 layers without host
waits, staging fences, remote waves on the device exchange; only while the other lane is idle):
2 RTX + 4 Sparks code C1 185.5 -> 192.9 tok/s, C4 unchanged (old path), greedy byte-identical; 1 RTX
C1 +1%. Spark worker per 5-6-row wave: kernel ~510 of the ~544 us round trip (GB10 bandwidth,
~24 experts x 5.9 MB at TP4), host ~20 us: TP6 is the bigger remote lever. Write mode
(`CUTEAFD_SPARK_WRITE=1`, NIC-written rows + flags, GPU waits; needs this branch's Spark build) is
built but not yet run. Open: both lanes device-ordered corrupt C4 (`CUTEAFD_V41_DEVICE_LANES=1`,
cause not found); whole-step graphs (D4); nsys traces of the overlay image came out empty.

**Spark-side reduction (measured and parked 2026-10-01, `work/spark-reduce`).**
TP ranks reduce-scatter their routed partials by rows over an RC mesh between
the Sparks (`expertd --reduce-rail`, SEND_WITH_IMM tagged per wave, FP32 sum in
rank order) and each returns only its rows, so the coordinator lands one plane
instead of N. Correct (sums bit-identical to the coordinator reduce, oracle
cosine 0.999995, MiMo V2.6 Pro golden NLL unchanged at 2.4123), but MiMo V2.6
Pro TP6 8K prefill gained only ~3% on one 200 Gb rail and ~9-12% on two: every
rank waits for the slowest peer's slice, the exchange is bandwidth-bound
(NCCL's ceiling on rhea+moa: ~20 GB/s per direction on two rails, ~11 on one),
and it gets worse at 100 Gb. Coordinator-side intake and pipelining come first.

Ongoing, any phase: engram/n-gram tables are memory-mapped from the
checkpoint (`formats::mapped_table` + daemon `shared::mapped_table`: page
cache, bounded prefetch, gather pool/worker, pinned upload ring, stats; V4.1
engram and Qwen PLE use it); Spark-RAM replicas and fabric-fed tables are
explorations, kept behind options.

## Release v0 (2026-10-02)

Cut early from `work/p0` (p9 + benchmark dashboard + live console for every
family + README card grid) with the V4.1 8K prefill staging regression fixed.
The full Release smoke matrix (family × quant × natural-minimum and maximum
hardware) published as the README card grid is the v0 artifact. Images stay
local until TJ says to push them. No model license notes: we bundle no weights.

## Release v2 scope (decided 2026-10-05)

The next release is v2.0.0 (TJ: multimodal and the 5090 platform make it
major). TJ's decisions, recorded with Hugh Madden's 5090 feature requests
([#1](https://github.com/tpurtell/cuteafd/issues/1) GLM 5.3 Flash,
[#2](https://github.com/tpurtell/cuteafd/issues/2) V4.1,
[#3](https://github.com/tpurtell/cuteafd/issues/3) MiMo V2.6 Flash; replies
posted). Hugh donated his code: port it directly from ds41rt-rtx5090
v2.1.0, glm53f-afd v1.1.0 and mimo26f-afd v1.3.0, crediting repo, tag and
file in the commit. Our gates and policies still decide, and we don't
benchmark against his engines. Agent branches are listed with each item.

Policy decisions:
- **Fidelity set:** the current check scores one 512-token PLAN.md passage
  (±1.4-point top-1 standard error, no code or tool calls). A redesigned
  agentic-coding set (draft `docs/fidelity-design.md`) replaces it before
  any precision default changes on the new bar.
- **Precision bar:** a lossy default (FP8 KDA/head, A8, …) must pass the
  paired one-sided 95% non-inferiority bounds: top-1 loss <0.005 and KL
  increase <0.005 nat on both full-tier scoring shapes. Use only qualified
  goldens from official checkpoints, no external teacher (§11 of
  `docs/fidelity-design.md`). The cross-family absolute floor stays at
  top-1 >=90% / KL <=0.06 nat. Each config derives its own `expect` from
  its repeated baselines; V4.1's calibrated `expect` is top-1 >=94% /
  KL <=0.04 nat, not a common floor. Its FP8 vocabulary head
  passes full decode and prefill (top-1 upper bounds 0.000937 / 0.001854;
  KL upper bounds 0.000846 / 0.000305 nat). Quick is inconclusive, not fail;
  independent agentic replay remains required and defaults are unchanged.
- **Qwen reference ULP-sensitivity floor:** on SM121, changing official
  eager QSA from variable K to zero-padded K2560 gives mean full-vocabulary
  KL 0.0176858 (legacy) / 0.1790475 (a00), with seven confident top-1 flips
  on a00. The unchanged-source variable-K arm reproduces both original
  raw-F32 goldens byte-exactly; frozen attention padding agrees exactly on
  CPU with FP64 inputs, including a true-FP64 softmax diagnostic. This is
  consistent with GPU reduction-order perturbations amplifying through
  48 layers, not observed runner/checkpoint drift. Both official-math
  references remain valid; no regeneration, threshold relaxation or
  default promotion. This two-window sensitivity measurement is not a
  full-text-set noise bound; measuring that floor is a later item.
- **GLM 5.3 EXL3 K4 open finding:** copy-heavy context rows d03/d04/a25
  remain flagged. The aligned d04 versus c03 diagnostic shows an isolated
  attention spike at layer 22 and content-specific late divergence after
  about layer 46. The official d04 replay reproduces its confident answer;
  the engine keeps copy-source token 1103 at every full-index layer after
  physical-cache-slot IDs are mapped to logical positions. Late selected
  sets overlap the reference by about 94.5-97.2% (c03: 97.4-98.5%);
  selection drift is within about 5.5%, not a copy-source drop. A quant-ladder
  lead: the EXL3 K4 package stores indexer wq_b/wk as FP8 E4M3, versus BF16
  in the official checkpoint (weights_proj is BF16 in both). Queued, not
  run: repack those indexer weights in BF16, rescore d03/d04/a25 and
  controls, and compare context KL. A forced layer-50 attention check
  cannot be reconstructed from current dumps:
  per-layer K/V history and the MLA query/cache were not saved. Quantization
  amplification is a hypothesis, not an engine-correctness verdict; no
  further causal-debug hardware run is scheduled. Cross-architecture
  reference sensitivity is also observed: d03 scored 1403 (input 1402)
  gives token 6337 at p=0.94066 in the original SM121 golden, versus 7388
  at p=0.68815 from the truncated SM120 residual with CPU official-head
  replay (KL 2.37465). The neighboring d03 scored 1404 reproduces token
  1419 (KL 0.00003495); d04 scored 2214 reproduces token 2638 (KL 0.001403,
  not below 1e-4). Architecture, truncation and CPU head arithmetic differ;
  this is reference sensitivity evidence, not an isolated architecture
  cause or an engine-error exemption. Coordinator adjudication retains
  qualification and fidelity-side upload readiness with this note; upload
  still requires TJ's license decision and explicit approval.
- **v2.0.0 payload (TJ, 2026-10-07):**
  - Multimodal: vision by default for MiMo V2.6 Flash/Pro, GLM 5.3 Flash and
    Qwen 3.8 (qualified WP-7 availability and single-image gates), with the
    encoder on an expert Spark (D1). V4.1 too: its vision tower moves to a
    Spark by default like every other family (TJ, 2026-10-08).
  - GLM 5.3 Flash beast mode: startup graphs, real-row MoE dispatch,
    precreated workspaces, Hugh's Wave A, then his Waves B/C (1.6M-token
    pool, 1M extent, C16 speed) as they land.
  - Fixes: the MiMo V2.6 fused-QKV scale grid (cuteafd#3), Qwen
    free-memory pools (C16 admitted 7 -> 16), and everything else merged
    into work/p0 since v1.
  - Quality on the dashboard: Quick (card), Standard and Full fidelity tiers
    against the public dataset, with golden NLL for every family.
  - Golden measurements: every served family qualified on the public
    dataset, with MiMo Flash re-qualified and MiMo Pro added after the QKV fix.
  - A full set of card updates in three columns: **5090** | **1× RTX** |
    **2× RTX**. 1× RTX uses the fewest Sparks the model fits on (usually 4);
    2× RTX uses 4 or 6, whichever divides it sensibly; any column whose GPUs
    hold the whole model is a 0-Spark card (likely Qwen and GLM Flash on
    some quants; `cuteafd plan --layout` decides). Supplementary cards where
    useful, e.g. GLM Flash on 2× RTX with 0 Sparks.
  - 5090 support, fully tested for the models Hugh runs: PLAT-1 (one SM120
    image), PLAT-2 (the 32 GB plan) and PLAT-3 (GeForce defaults), with 5090
    cards produced by Hugh's agent at release-candidate time
    (hughmadden/cuteafd-collab item T-3).
  - Audio: MiMo V2.6 Flash/Pro `input_audio` (WP-10, work/mm-mimo-audio;
    the CPU reference is on work/p0 at 8acabb99), AUDIO=auto following VISION
    once its gates pass. Video (native in MiMo, GLM Flash and Qwen) only if
    image and audio are done while we still wait on Hugh's v2 pieces; plan
    NVDEC decode on the encoder's device, gated against a CPU reference.
  - Lower priority (TJ, 2026-10-07): **scoring without a launch flag.**
    Full-tier prefill scoring needs all-row prefill logits, which today must
    be admitted at launch (FULL_PREFILL_LOGITS=on reserves the extra
    logits/workspace before KV sizing), so the bench console can't run Full,
    or Standard's prefill half, against a default server. Instead, borrow the
    memory from the KV pool per scoring request: a scoring request reserves
    whole free KV pages, enough for its all-row logits and enlarged workspace,
    for its duration, through the same admission path as any KV allocation.
    It waits or returns a 429 with a reason when the pages aren't free, and
    it never evicts live sequences or shrinks admitted requests. The pages
    return when it finishes. The engine must run the all-row prefill in that
    borrowed storage with the same kernels and geometry as the launch-flag
    path, so the scores are byte-identical (gate: borrowed vs launch-admitted
    scoring, identical per-row records on Quick/Standard/Full). Then the
    console gets proper Full measurements on any server, and
    FULL_PREFILL_LOGITS remains as an explicit reservation for dedicated
    fidelity hosts.
- **Gate provenance:** every gate seal JSON records the exact source commit
  and a dirty flag, including untracked files, alongside the binary hash.
  Rebuilt gates use task-private targets; never repin a changed shared binary.
- **Exact speculation:** attempt byte-exact greedy speculation (drafts
  on/off, C1→C4) per family when it doesn't cost C1. Where it does, keep the
  faster path and accept proven rounding.
- **Context and KV:** the planner's default admits at least one request at
  the model's full context. The default KV target is 2M tokens on an RTX
  PRO 6000 and 1M on a 5090 (launch flag); the planner adapts expert-layer
  onboarding and cold-component placement to meet it, and reports any
  shortfall.
- **Multimodal** is in scope for every model: official bundled encoders
  only, planner-placed (the tower can live on a Spark, likely by default,
  since it's rarely used), and an image-embedding cache tied into the
  prefix cache, so long sessions carry many images without re-encoding.
- **MiMo Flash** moves to `XiaomiMiMo/MiMo-V2.6-Flash-MOPD` (same geometry
  as Hugh's V2.6 Flash RL).
- **Sampling:** keyed Gumbel-max draws with coupled drafts (#3 FR-M.11) go
  in if they are at least as fast; seeded output then repeats across
  batches, caches and draft settings.
- **Thinking off** maps to the template's Low effort as a setting (GLM
  default Low); `"minimal"` never reaches the template as Max.

Work, in priority order:
1. **In flight (wave 1):** Qwen 32 GB + 1 Spark minimum with a generic
   coordinator memory budget (`work/qwen-min-5090`); V4.1 Spark load speed —
   the serial WILLNEED prefetch cost ~94% of each Spark layer's load
   (`work/v41-spark-load`); V4.1 NVFP4 W4A4 wide-row efficiency (FP4 at 3–5%
   of peak at 2048/4096 rows; Spark decode already at 70–75% of GB10
   bandwidth; `work/v41-nvfp4-w4a4`); GLM Flash split FP8 root cause and
   per-layout precision under the bar (`work/glmf-split-fp8`); Hugh's V4.1
   patches 0001–0003 plus the ~1M self-eviction fix (`work/hugh-v41-ports`);
   MiMo V2.6 Flash MOPD bring-up (`work/mimo-flash-mopd`).
2. **One SM120 image for the PRO 6000 and the 5090 (PLAT-1, #2 FR-D.1/2):**
   remove the 188-SM guards and size grids from `cudaDevAttr`; sparse-MLA
   blocks in whole waves (#1 FR-G.10); interim: images carry
   `physical_sms` and `run.sh` refuses a mismatch.
3. **Full-context planner default + 32 GB plans (PLAT-2):** per-family
   profiles (V4.1 capacity 1024 / 256 on 32 GB, #2 FR-D.3; GLM Flash
   #1 FR-G.4; MiMo #3 FR-M.5), 1M program extents (#1 FR-G.1). Cold
   components (vision towers, rarely used) move off the GPU by default. The
   token embedding is not cold: every token reads one row (a gather of
   ~8–10 KB on the decode critical path). It stays on the GPU by default; a
   host-mapped embedding (like Engram) is a planner lever only when memory
   binds (32 GB cards, ~1.2–1.3 GB saved) and its measured C1 cost is
   within ~0.5% (TJ, 2026-10-05).
   **32 GB candidate issues (2026-10-08, work/plat2-32gb):** one serving heap
   abort (`corrupted size vs. prev_size while consolidating`) in the initial
   C16 attempt; not reproduced in four corrected launches plus ten distinct
   C16 soak batches with MALLOC_CHECK_=3/GDB. Separately, one complete CPU
   loader suite aborted with `corrupted double-linked list`; 50 exact parallel
   MALLOC_CHECK_=3 repeats, three matching-image ASAN full suites and 24 ASAN
   Engram/media/tokenizer subgroup runs passed. Baseline parallel/serial and
   exact GDB repeats also passed. Memcheck/Helgrind were time-boxed; huge
   fixture mappings limit coverage. Neither abort is fixed or proven related;
   the heap hunt is closed for v2 unless it recurs. MALLOC_CHECK_=3 serving
   soaks remain required; no memory-safety qualification is claimed. Independently, lazy target graph retention grew to ~12.6K counted
   executables with unchanged device owners: untracked residency 3.24 ->
   5.16 GB after batches 1 -> 5, leaving only 0.35 GiB of the logical 31.8 GiB
   budget. The small-card fixed eight-shape bank (`1,6,16,24,32,40,43,48`),
   exact index-selection retention/eager overflow and measured 2 GiB graph
   envelope passed three interleaved C1/C16 comparisons and ten distinct
   MALLOC_CHECK_=3 C16 batches on pin c595ba7: C1 +0.56%, C16 ratio 0.98194 (pair ratios
   1.01965/1.02588/0.95888), minimum warmed logical free 4.084 GiB;
   all four selection owners captured 256 exact fingerprints without recapture.
   Accepted for the small-card profile; finite lazy warm-up remains, not a
   startup freeze. PRO policy remains unchanged. MiMo host/GPU timing was
   byte-identical with no measured host penalty (+0.51% median); its 256K
   retrieval passed, but long-prompt free 2.5605 GiB failed the margin.
   Growth was already-admitted shared KV shadow/lane/drafter storage, not a
   missing reservation. Both small-card families now use an absolute 2.9 GiB
   floor with the 97% ceiling; MiMo's measured contract predicts 962,560
   aggregate tokens across 16 slots, an 86,016-token full-context shortfall.
   Matching pin 682f5ad confirmation passed: V4.1 production-default auto
   pool 3,342,848 tokens, C1/C16 code smoke and ten distinct C16 batches,
   minimum observed logical free 3.9863 GiB, no observed selection recaptures
   or capture failures. MiMo retained Int8 KV and the predicted 962,560-token
   aggregate pool; its 64 MiB startup probe shares the floor slack only while
   the probe is live (max(probe, floor), not their sum). Steady headroom stays
   2.9 GiB. A 262,048 API-token request returned the exact requested secret
   with 3.5976 GiB minimum observed logical free. The completed 640-token
   host/GPU code outputs are byte-identical and pass structural checks, but
   share an incorrect interval-merging assert; the checker does not execute
   generated code. Neither heap abort recurred in these gates. PRO-default
   parity/readiness on the rebased fcb6706d build remains the coordinator's
   merge gate; no full-1M prompt or real-5090 qualification is claimed.
   MiMo's launcher/planner auto concurrency is 16 only for physical or logical
   budgets at most 32 GiB; PRO retains 8 and explicit overrides are preserved.
   The direct serve-mimo CLI's upstream default of 4 is unchanged.
4. **Multimodal for every family** with planner placement and the
   embedding cache (#3 FR-M.12 for MiMo). Towers to add: MiMo V2.6
   Pro/Flash MOPD (vision 28×1280, plus audio), Qwen 3.8 (vision 27×1152),
   GLM 5.3 Flash (vision 24×1024; we build it, TJ 2026-10-05, though the
   rest of GLM Flash is Hugh's). V4.1 vision already works and is the
   template. Design first (docs/multimodal-design.md), then one shared
   encoder service and image-embedding cache, then the families.
   **D1 decided (TJ, 2026-10-06): the encoder goes on a Spark that already
   holds experts.** One new image per turn costs ~150 ms of stall (one
   1024-token image, measured), and history images come from the
   embedding and prefix caches. That beats spending RTX memory or a whole
   Spark on the tower. The −91% decode under back-to-back 4096-token
   encodes is a stress case, not a gate for this default. MiMo and GLM Flash
   launches now default to `VISION=auto`; Qwen joins them on the qualification
   below. Other generic families stay `off` until their towers are qualified.
   **Placement policy (TJ, 2026-10-08):** every family, V4.1 included, moves
   multimodal towers (vision, audio) to Sparks automatically by default;
   KV, layers and graphs are the better use of RTX memory. Hot but
   offloadable parts (the token embedding) stay on the GPU on an RTX PRO 6000
   unless benchmarks show no decode/prefill regression; on 32 GB cards the
   default is the embedding in host RAM (Hugh's tradeoff).
   **Merge note (Qwen WP-7 + GLM Flash WP-9):** the cold_steps echo allowlist in
   `cuteafd-api/src/openai/probe.rs` is `mimo_v2|qwen4` on WP-7 and
   `mimo_v2|glm5_flash` on WP-9. Each branch lists only families whose
   replay executor it contains. When merging both, resolve it to the union
   `mimo_v2|qwen4|glm5_flash`, with positive tests for all three. Never
   list a family whose executor isn't present.
   The same applies to `scripts/launch/run-family.sh:279`, where
   MEDIA_CACHE_BYTES is forwarded: WP-9 has `mimo_v2 || glm5_flash` and WP-7
   adds `qwen4`, so resolve that to all three.
   **Remote integration qualified:** MiMo serving uses `RemoteEncoder`
   with checked `vision_peers`, revision and `encoder_plan_hash`; readiness
   waits for every replica. Remote failure marks `/health` vision failed
   and rejects cached/new images with 503 while text continues. Flash-min
   C1 three-arm parity, chart/G6, exact G7 a/e, single-image interference
   and vision-only connection-loss gates pass. Explicit `VISION=off` skips
   tower startup; checkpoints without a tower remain text-only.
   **GLM Flash startup promotion `5e6652d3` (2026-10-07):** the matching
   SM120/ARM64 SM121 build and five-host installed audit pass. The same-image
   C1 confirmation is 66.88 -> 70.54 emitted tok/s (1.05483x); startup C16 is
   176.64 tok/s. Ready-to-text and both ordinary-media passes add zero graph
   captures; all reachable LM/drafter workspaces are resident at readiness,
   tracked growth is zero, and the 2 MiB global residual is within 16 MiB.
   Readiness is 72 -> 78 s: graph capture measures 4.21 s; the remaining
   approximately 1.8 s is unisolated launch/load/poll variation, not attributed
   to workspace precreation (both arms precreate the same storage).
   TJ accepts the 4.21 s graph-capture cost: runtime performance takes priority
   over readiness, while wasteful load transforms into required tile formats
   must still be avoided. The readiness-only follow-up is cancelled, not run.
   Startup graphs are qualified for the `work/p0` merge, subject to the batch
   V4.1 parity gate. **GLM Flash vision qualified (2026-10-07):** retained
   G1-G7 matching-native evidence plus the frozen three-session baseline/off/auto
   text parity pass: median paired C1/C16 ratios are 0.99764/0.99731 for off and
   1.00272/0.99656 for auto (bar 0.98). Default-quota served-image interference
   at 256/1024/4096 tokens passes; first-delta latencies 569/1274/3159 ms include
   admission and LM prefill, not isolated encoder stall. The separately sealed
   loss/off v3 retry passes encoder-only connection loss, cached/new image 503,
   continued active/new text and unchanged experts. Explicit `VISION=off` admits
   zero encoder bytes, rejects images and leaves media counters zero. No runtime
   graph captures are added by media/loss/off. GLM Flash now defaults to
   `VISION=auto` in the launcher and direct CLI; off remains explicit. The planner
   derives qualified resident weights from checkpoint headers (1,128,026,176 B)
   plus fixed-capacity scratch (791,907,584 B): 1,919,933,760 B admitted, Spark-first
   when capacity permits. Unsupported towers remain unsupported, not admitted;
   checkpoints without a tower stay text-only. Quantized checkpoints lacking a
   compatible bundled template still require explicit `CHAT_TEMPLATE_FROM`.
   The original failed v1 loss evidence stays intact; v3 closes the remaining
   gates. Shared code gets the quick A/B pair on an affected model (AGENTS.md).
   **Qwen image capacity:** 1024 merged tokens per image, `detail=low` 256,
   BF16 residual; omitted `VISION` now defaults to `auto` in the launcher and
   direct CLI (RTX on the zero-Spark minimum); explicit `off` remains unchanged.
   Frozen full-image v4 three-session availability medians pass: auto/off C1
   1.00208 and C16 0.99546 (bar 0.98). Single-image interference at 256/1024,
   the distinct 1024-cap witness at default quota, encoder-only loss with
   cached/new image 503 and surviving active/new text, unchanged experts,
   official-off checks and zero runtime graph captures pass. Planner admission
   equals native: 898,680,904 B weights + 447,778,048 B scratch = 1,346,458,952 B.
   The original sealed rolling-traffic analyzer remains FAIL; TJ classifies
   rolling submissions as stress, not a v2 promotion gate (2026-10-07).
   Known limit: under continuous image traffic (2 of 16 slots resubmitting
   image chats), text decode drops to ~22% of the off rate; likely the
   prefill_share floor; not addressed in v2. This is relative to off C16
   scaled by 14/16. CPU inspection confirms image LM prefills use shared
   rounds at decode_share=0.2; about 74-76% of the 90 s window is image LM
   prefill busy time, not the roughly 40 ms encoder alone. The fixed floor
   is a hypothesis, not proven: settle clears debt when the queue empties.
   GLM Flash shares this policy; its retained gates cover single images,
   not rolling traffic. No scheduler change is made for v2.
   Qwen tower >1024 tokens: BF16 fails calibrated G2 at
   2048/4096; FP32 residual fixes 4096 but regresses 256/1024 worst-row.
   Bounded diagnostic: patch row 863 has dominant channel 514; block-27 cosine
   0.99997 collapses at merger LayerNorm (0.588 versus BF16's 0.870).
   Hypothesis: a massive-activation outlier and LN amplification, not a
   demonstrated row-handling bug. Larger image caps remain unqualified.
   **WP-7 concurrency workload:** the Python concurrent benchmark defaults to
   one identical prompt for every request (`scripts/bench/deepseek_v41/bench-concurrent-api.py:45`).
   Qwen's retained C16 Copy comparison generated mostly lockstep lazy outputs
   but diverse bucket outputs, so its ratio does not isolate matched expert
   work. WP-7 task runners must pass `--distinct-prompts` for both Copy and
   plain C16 arms, with the same deterministic nonce and per-request index;
   qualify three interleaved pairs and their median before promotion. Preserve
   the identical-prompt evidence as a separate workload, not a diverse-load
   performance gate. The shared Rust `cuteafd-bench` concurrency panel already
   uses a unique nonce per request and alternating code/summary prompts.
   **Qwen TP4 admission:** published WP-7 C16 figures used the former
   32,768-token default KV pool and were admission-limited to seven active
   requests, not by EXL3 slots. The retained 104/105-token prompts plus a
   4096-token output budget and 64-row verify slack reserve seventeen
   256-token units per request; only seven fit in 128 units. Qwen's launcher
   now selects `POOL_TOKENS=auto` (explicit fixed pools still override), using
   existing free-memory admission after resident weights with workspace,
   state, prefix, graph and headroom reserves. A fresh identical-config A/B
   uses 73,728 tokens in both arms so all sixteen requests fit. The distinct
   three-pair full-width medians pass the 0.98 floor for plain and Copy;
   startup graphs now default on for serving, with `QWEN_STARTUP_GRAPHS=off`
   restoring lazy capture. Golden/diagnostic engines keep exact-shape behavior.
   Before KV allocation, admission solves the candidate pool's enumerated
   graph count (actual context, layers, sequence count and speculative modes)
   times the 2026-10-07 measured 149,712 bytes/graph, plus a margin of the
   larger of 10% or 256 MiB; headroom remains a separate reserve. The prior
   fixed 0.5 GiB serving estimate is not used for startup graphs. The
   default-unset confirmation after V4 Pro calibration passes C1 Copy and
   plain C16 single-pair floors, with zero capture deltas in both candidate
   warmup/measurement intervals and byte-identical paired plain outputs.
   Auto KV admits 2,097,152 tokens; enumerated and captured graph counts agree,
   measured graph bytes fit the graph reserve, and post-startup CUDA free
   exceeds the separate headroom. Readiness is measured at the first genuine
   completion, not API-open. This confirms the promoted image, not a new
   three-pair performance measurement. The separate v4 availability and
   single-image qualification above promotes VISION to auto.
   **Qwen multimodal maximum:** requested 2 RTX + 4 Sparks, effective 1 RTX
   + 4 Sparks, correctness only; Qwen has no coordinator head split and the
   second RTX is idle. Qualify Spark `VISION=auto` startup/readiness, image
   QA, prefix a/e, one-image stall and proxy encoder loss on that layout.
   Placing the encoder on the idle second RTX is a future planner option,
   not part of this bring-up; do not claim two-RTX LM parity.
   **Media-cache merge policy:** shared cache admission distinguishes a single
   image larger than total capacity (permanent 400, `image needs N bytes > media
   cache capacity M`) from capacity sufficient but pinned entries preventing
   admission (transient 503 with `Retry-After: 1`). Qwen implements both; apply
   the same pressure mapping to MiMo at integration (currently 400); GLM Flash
   already maps pressure to 503. Qwen warns at startup when configured cache
   bytes are below max admissible tokens times BF16 feature-row bytes, naming
   both byte counts; add the warning to MiMo and GLM Flash at integration.
   Do not reject/clamp deliberately tiny G7 eviction-test quotas. The shared
   `ImageTooLarge` variant must stay in each family's permanent-400 fallback.
5. **Platform robustness:** GeForce defaults (probed pinned intake, no
   P2P/GPUDirect; PLAT-3), RDMA device from the fabric address and bond
   balance (#2 FR-D.4), per-Spark free-memory guard and page-cache drop
   without `nest` (PLAT-5), `/health` 503 on expert failure, an optional
   API key, keyed bench controls (PLAT-6, #1 FR-G.15), malformed tool calls
   returned as content (#1 FR-G.14). Bond balance, in part: with the
   coordinator key `RDMA_BOND_BALANCE=probe` the expert QPs connect with
   RoCE v2 flow labels (the UDP source port) the coordinator chooses; it reads
   4 MiB from each candidate label's worker QP, sees which bond member's
   `rx_bytes_phy` received it, and splits every transport's (lane's) flows
   evenly across the members, then each rank's (`cuteafd-transport/src/bond.rs`):
   a lane's ranks answer a wave together, so a lane whose flows share a member
   overruns it even when the two lanes' totals balance. Default `off` keeps the kernel's
   QP-number labels, which re-roll the split at every start; `labels` fixes
   labels without measuring. On one RTX 5090 + four DGX Sparks, `probe`
   split both lanes 2+2 at 10 of 10 starts (`off`: 1 of 10). Readiness bug (2026-10-05, fidelity
   agent): `/v1/models` reports ready before the Spark experts finish
   loading on the V4.1 launch path; readiness must wait for every expert
   rank. Admission bug (2026-10-05, host-embedding agent): MiMo V2.6 Pro
   `admission.rs:356-370` appends `startup.spark_intake_probe_temporary`
   (64 MiB) after `resolve_capacity`, so an automatic pool that fills the
   budget (466 KB margin) is admitted and then refused at startup
   (shortfall 66.6 MB). Reserve every startup-phase temporary before pool
   resolution, in every family's admission. KV admission bug (2026-10-05,
   fidelity agent): DeepSeek V4 Pro EXL3-K2 TP4 on one RTX with
   max-context = pool = 16384 tokens, max-output 1024, max-sequences 1
   admitted 82 sequential set probes, then the next allocation needed 66
   pages with 65 of 65 free. Admission must reserve prompt + max_tokens
   in pages, rounded to page boundaries, before accepting a request.
   GLM 5.3 prefill tail bug (2026-10-06, prefill-score agent): with
   pipelined lanes, `prefill_capacity()` is lanes × `prefill_rows`, but a
   chunk of 257–511 rows (e.g. a 511-token prompt, or the remainder after
   1024) is below `2 × MIN_LANE_ROWS`. It takes the serial path, where
   `glm5/engine.rs:661` requires `t <= prefill_rows` (256), and the request
   fails with a worker error. Split such tails into serial chunks of at most
   `prefill_rows`, or run them as one lane. The default is 4096 rows, so
   this needs a small `--prefill-rows` (256 here); the same mismatch
   applies to any tail between `prefill_rows` and `2 × MIN_LANE_ROWS`.
   Root-owned outputs (2026-10-06): `run.sh` and `scripts/launch/run-family.sh`
   start coordinator containers as root. Everything they write to mounted
   host directories (bench dir, fidelity `--dump-dir`) becomes root-owned.
   Agents then can't archive or delete it without agent-sudo (76 GB of MiMo
   calibration dumps this time). Run as the host user (`--user`, with
   `HOME`/`USER` set) once GPU, RDMA and memlock access are checked under it.
   Proven (2026-10-06): a GLM 5.3 Flash coordinator run as `--user 1000:1000`
   on GPU0 + 4 Sparks completed full fidelity scoring over RDMA. All of its
   dump files were owned by uid 1000, and its decode results were identical
   to the root-run baseline.
   **V4 Flash decode differs between launches (2026-10-06, fidelity):** two
   launches of the same config with identical settings (GPU0 + 2 Sparks,
   TP2, release native, `--full-prefill-logits`) scored the 17,401-position
   decode-shaped panel at 96.552% and 96.937% top-1: 67 net agreements,
   hundreds of per-position flips. MiMo Flash, GLM Flash and Qwen repeat
   their aggregates exactly under the same harness. The DSA index top-k is
   already deterministic (315b821). Suspects: atomic or unordered
   accumulation in V4 expert combine or exchange, and launch-time kernel or
   split choices. This widens paired precision noise for V4 Flash and
   blocks byte-exact speculation there.
   **Separate task: GLM Flash speculative decode is not launch-deterministic
   (2026-10-07, WP-9):** greedy, temperature 0 / seed 0, identical prompt and
   tokenizer diverge at token 6 between launches on the unchanged lazy path
   (retokenized SSE text, not a raw generated-ID trace). Lazy/startup also
   diverge; the lazy/lazy control rules out attributing this to bucket padding
   alone. Find the input: adaptive draft length or copy policy reading timing,
   or nondeterministic Spark reduce order. The draft cost model demonstrably
   observes live draft and verify wall time and replans within a request;
   timing -> proposal width -> arithmetic is a candidate, not a causal trace.
   Principled fix: make verify logits width-invariant for real rows, extending
   aligned-bucket invariance across widths and buckets so greedy output is
   independent of scheduling. Alternative: seed from a fixed cost table and
   update it only between requests, not mid-stream. Goal: same prompt, same
   config, same token stream. Until fixed, every performance A/B on a
   speculative-decoding family needs at least three launches per arm; retain
   every pair and judge the median paired emitted-throughput ratio. This is a
   separate determinism task, not part of WP-9 vision/default promotion.
6. **GLM Flash (owned by Hugh, 2026-10-05; we only finish `work/glmf-split-fp8`
   and run a quick A/B on an affected model for his shared-code PRs):** compact pooled-key index cache (#1 FR-G.3, ~half the KV),
   four prefill lanes and two decode lanes (FR-G.8, G.11), BF16 KDA state
   and row-independent kernels for exact speculation (FR-G.2, G.6), the GB10
   EXL3 decode schedule (FR-G.7), 32 GB plan, API items, teacher gate.
7. **MiMo:** decode expert rows and the 16-stream scheduler (#3 FR-M.7/8),
   copy windows (FR-M.10), the RAM snapshot tier (FR-M.9) — port from
   mimo26f-afd.
8. **V4.1 carry-overs:** cold-prefill graph reuse in the plan (#2 FR-D.8,
   item 4m), Engram counters / shard dir / table warm (FR-D.10), preflight,
   plan-only boot and ready probe (FR-D.11), whole-step graphs and
   device-side draft acceptance, W4A4 decode rows, deterministic prefill.
9. **Open issue: one V4.1 heap abort on the 32 GB profile (2026-10-08).**
   A candidate small-card launch (31.8 GiB logical budget, capacity 1024,
   dSpark, C16, work/plat2-32gb) aborted once in C16 with glibc
   `corrupted size vs. prev_size while consolidating` (host heap); C1 was fine.
   Not reproduced since in 4 repeats (baseline 0143dcc4 and candidate,
   fixed and automatic pools, under GDB and plain with `MALLOC_CHECK_=3`) nor
   in a 10-batch C16 soak under GDB. No core was captured. The audit of new
   host-to-native writes found no size mismatch. Treat as open: rerun
   `MALLOC_CHECK_=3` soaks on small-card profiles before qualifying them.
   The same runs showed lazily captured graph executables (≈12,600 at plateau,
   ≈5.2 GB untracked) consuming the 32 GB card's margin; the small-card profile
   moves to a fixed graph set reserved before KV.

9. **Open issue: synchronized Spark response gaps.** GLM 5.3 Flash split
   (2 RTX + 4 Sparks, 2026-10-05 01:26 UTC): one BF16 launch had ~440 ms
   inter-wave response gaps on all four workers at once (normal 15–18 ms),
   with identical routes, GPU work and Spark clocks, and no overlapping
   cluster job. This caused the historical "split FP8 −24%" reading (not
   precision). Suspect a coordinator or transport timeout/retry path; it
   may affect every family. Investigate if it recurs.
10. **Build hygiene (merged 2026-10-05, `work/build-hygiene`; verify on the
    first real `./build.sh`: GPU access under `--user` and the relocated
    CARGO_HOME were only checked by inspection):** `./build.sh` took the hardware locks and pinned GPU0
    through its CPU, download and AOT export phases; it should take
    `build.lock` for those and touch hardware only where it measures.
    Build containers run as root, leaving root-owned `target*` directories
    agents can't delete; run them as the host user (UID 1000 on raptor,
    1001 on the Sparks).
11. **V4.1 NVFP4 decode (parked 2026-10-05):** NVFP4 trails official MXFP4
    by 16% C1 / 14% C4 on 1 RTX + 4 (tokens per round 3.94 → 2.87, Spark
    expert kernel +32% per layer). ncu: the cooperative NVFP4 kernel runs one
    90 KB CTA per SM at 6% occupancy, versus MXFP4's fused-slice kernel at
    2 CTAs per SM. A noncooperative NVFP4 slice kernel (fork
    `work/v41-nvfp4-slice` 3173cc2e) reached 3 CTAs per SM but ran 28–38%
    slower at 1–16 rows. Code audit (2026-10-09): standard cooperative FC1
    route packing quantizes BF16 to FP4 once per token × route, not per FC1
    slice. Native MXFP4 takes pre-quantized FP8 wire rows. Fix the shared-max
    FC1 scale approximation first; then gate FP8 wire rounding and occupancy
    changes separately in v3. Wide-row M32 tiles lost
    on both GPUs. Official MXFP4 stays the recommended V4.1 checkpoint.
12. **V4.1 C1 after request history (open, 2026-10-05):** after smoke +
    golden-probe requests, work/p0 decodes C1 ~1.7–2% slower than v1.0.0 on
    2 RTX + 4 Sparks (warm C1 190.4 → 187.2, ratio 0.983; fixed card 0.990;
    output byte-identical). On a fresh server C1 is equal. GPU code is
    byte-identical (all CUDA fatbins and embedded CuTe cubins); the
    difference is in the Rust daemon and needs the history. Ruled out:
    native pin, host snapshot arena (identical 7.2 GB, no evictions), Engram
    page-cache residency (±20 MB of 3.2 GB), copy-window bookkeeping (a
    copy-off fast path changed nothing, 0.999). Clue: dSpark's fitted draft
    cost after history is ~1,547+40 µs vs v1.0.0's ~1,492+37 µs. Next, if
    pursued: bisect the policy merges since v1.0.0 (569f4f6 draft policy
    core and verify-cost fit) under the same smoke-conditioned protocol.
    Harness: ~/.cache/cuteafd/builds/c1bisect. Correction (c1policy): 569f4f6
    predates v1.0.0, and the gap does NOT reproduce on 1 RTX + 4 (warm C1
    0.999, identical draft-cost fits 1,648/1,646 µs, 4.71 tokens per round
    both). So it's 2-RTX-head-split-specific. Remaining Rust candidates on
    the serving path: d60f989 (Hugh ports) and b615cc7 (Qwen merge); bisect
    on 2 RTX when GPU1 is free. Evidence: ~/.cache/cuteafd/builds/c1policy.
13. **Parked:** EXL3 × A8 (fails KL), MXFP4 tails, V4.1 exact slices,
   Spark-side reduce, split intake.

## Release v1 scope (decided 2026-10-04)

v1 ships when these are done; everything else below moves to v1.x/v2.
- **In v1:** device-driven exchange for V4.1 and MiMo V2.6 Pro (default only
  if it beats the current default and is hang-free; otherwise opt-in);
  byte-exact prefix-cache restores at turn end for V4 / V4.1; planner core for
  every family (per-device memory layout + admission); all Release smoke
  cards green with MOPD as the MiMo Pro default
  and a refreshed README; the Spark kernel wins already
  landed; known-issue notes (NVFP4 local experts on one RTX, Qwen with Sparks).
- **Status 2026-10-04:** the device exchange is merged opt-in and hang-free
  (`CUTEAFD_V41_DEVICE=1`; +1.5% C1 on 2 RTX, flat elsewhere; MiMo/GLM
  adoption on `work/device-mimo-glm` gains nothing — their segments are
  GPU-bound). It ships opt-in in v1. Turn-end prefix restores proved exact
  (the check was wrong; fixed). V4.1 FP8 vocabulary head: Claude accepted the
  target-head quality result (KL +0.000534 nat, NLL unchanged in practice,
  golden top-1 472 → 465 / 512). `all` now uses one shared FP8 residency;
  promotion requires C1/C4 ≥ 1.02 and other parity ≥ 0.98 on both layouts.
  The earlier single-copy gate missed C1 on dual RTX. RC1 code nevertheless
  defaults to `all`; prior notes incorrectly said BF16. The release-prep
  matched dual-RTX ABAB recheck does not reproduce the historical C1 drop,
  so RC2 retains that FP8 default. `draft` retains dual residency.
- **Cut to v1.x:** MXFP4 32-row tails and V4.1 exact Spark slices measured
  flat; their opt-in implementations remain on `work/mxfp4-tails` and
  `work/v41-exact`. Neither blocks v1.0.0.
- **Cut to v1.x/v2:** whole-step graphs (D4: context-length-dependent index
  graphs, per-request pointers in graph keys, host-built per-layer metadata,
  warm re-captures) and device-side draft acceptance; deterministic
  (batch-invariant) prefill and verify;
  multimodal input (v2); `placement.json` handoff and cold-component placement;
  V4.1 NVFP4 W4A4 revisit and W4A4 decode; EXL3 × A8 (an independent SM120
  implementation is the interesting part — not a port of b12x PR #342, whose
  ShapleyMcg licence covers re-implementations made with reference to it);
  parked Spark-side reduce / split intake.
- **RTX 5090 support: Hugh** (external collaborator). Brief: one SM120 build
  serves RTX PRO 6000 (188 SMs, 96 GB) and RTX 5090 (170 SMs, 32 GB) with no
  regression on the 6000; remove SM-count assumptions (hard-coded `4*188` grid
  clamps; the per-tensor FP8 GEMM grid sized for 188 SMs; any L2-size
  assumptions); simulate a 5090 on a 6000 via the planner's device inventory
  (`cuteafd plan MODEL --layout`, 32 GB budget) and validate on real 5090s;
  start from `work/p0`, branch `work/rtx5090`, follow AGENTS.md.

### v1.0.0 publication (2026-10-05)

TJ approved tagging v1.0.0, advancing `main` and `work/p0`, and publishing
both versioned and `latest` images. The runtime pair is built from clean
`release/v1.0.0` source `d5705aa6249d7d7c303895056dea7e5542dcd6a6`,
with universal Spark TP2/TP3/TP4/TP6 coverage and the unchanged SparkInfer
`f6bb38bc56fdcd695791c1ebf91d0dbf133cd599` pin and tree lock. The tag also
includes the deployment/documentation update naming the v1.0.0 image pair.

The final images passed serial Release spot-smokes for V4.1 Flash maximum,
Qwen NVFP4 minimum with resident mixed-format MTP3, and GLM Flash NVFP4
maximum. Every entry held the hardware locks, had bounded launch/benchmark
timeouts, and stopped its coordinator and workers before releasing locks.
A clean V4.1 smoke remained below the historical RC2 card, so matched
RC2/final ABAB controls used the same two RTX GPUs at 325 W, four Sparks,
configuration and fixed 320-token C1 code prompt. All four outputs were
byte-identical and the final images did not regress. The accompanying
nonced smoke prefill requests also ran at similar rates in both images;
this does not explain the shared shortfall against the historical card.
Measurements and conditions are in the publication commit and annotated tag.

The 28/28 preparation cohort is retained; these three spot-smokes do
not rerun the whole matrix. Golden fidelity and byte-exact cache restores
pass on the final images. Known speculative/batch rounding limits, Qwen
EXL3's documented default-speed gap and unsupported Spark MTP remain.
MXFP4 tails and V4.1 exact slices stay deferred to v1.x. Runtime defaults
are unchanged by publication. Logs and reproducible configs are retained
under `~/.cache/cuteafd/builds/v1-publish/`.

### v1 candidate changelog (preparation, 2026-10-04)

Changes since v0.1.0 touch every model family. The `release/v1` candidate
uses local `cuteafd-coordinator:v1.0.0-rc1` and
`cuteafd-spark-expert:v1.0.0-rc1` images. The family × quant ×
natural-minimum/maximum Release smoke matrix attempted all 28 cards (27 passing).
`CUTEAFD_RELEASE_ROW=v1 cuteafd bench publish` generated the family changelog
cards and README grid from RC1 exports; the benchmark index retains historical
reports. Qualification and blockers are recorded below. TJ explicitly
deferred V4.1 parity to the next
release; this candidate also skips `bench-ab`. Claude reviews the candidate,
then TJ approves tagging and publishing images.

- Shared runtime: stop-token grammar completion and structured stream
  errors, drained cancellation/staging lifetimes, bounded host cache and
  generic KV admission; family memory layouts and opt-in automatic admission.
- DeepSeek V4.1: coordinator-first loading, smaller one-RTX workspaces,
  per-width head graphs, opt-in device exchange and default single-copy FP8
  vocabulary head. V4 / V4.1 turn-end restore checks now
  compare each restored state to its own byte-exact snapshot.
- DeepSeek V4: qualified native Flash TP2 with legacy requests and exact
  local-expert placement overrides; compressed-cache/drafter memory planning.
- GLM 5.3: E4M3 MLA prefill, improved Spark EXL3 wave scheduling and TP6
  tiles, bounded decode graphs, automatic KV pool and FP8 DFlash2.
- GLM 5.3 Flash: default two-RTX head split, DFlash2 for all quants, FP8
  drafter and layout-dependent KDA/head precision (FP8 on one GPU; BF16 with
  the split).
- MiMo: Pro MOPD replaces RL, native A8 MXFP4 down projection, exact Spark
  slices, single-copy FP8 head/O/DFlash, pipelined head-split prefill; Flash
  defaults to two lanes while Pro retains three.
- Qwen: resident EXL3 placement with local MTP3 and one shared FP8 vocabulary
  head; Spark MTP and Spark FP8 remain unsupported. NVFP4 local experts are
  resident by default; paging requires an explicit expert window.

Open limits are recorded in `docs/models/`: numerical batch invariance and
byte-identical speculative output, V4 Pro fidelity reference, split FP8
GLM Flash promotion, V4.1 short FP8 replies and graph recapture, MiMo Flash
scaling/fidelity, native small-row NVFP4 W4A4, and Qwen Spark MTP/FP8. The
scope's MXFP4 32-row tails are outside this candidate: the implementation
on `work/mxfp4-tails` is unmerged and its strict distributed-oracle and
unchanged-NLL gates remain open. Existing exact Spark slices and A8 down
projection do not close that item.

### v1 candidate qualification (RC1, 2026-10-04 UTC)

`release/v1` prepares local `cuteafd-coordinator:v1.0.0-rc1` and
`cuteafd-spark-expert:v1.0.0-rc1` images from runtime source
`57d51b0fecfb9e501bf0e658c4a7a47ab325037b`, based on `work/p0`
`371d3a713f1686bd028766a344beb5f8aa81e700`. SparkInfer remains pinned at
`f6bb38bc56fdcd695791c1ebf91d0dbf133cd599` with its verified tree lock.
Coordinator image ID: `80086e9035e87d9eebda88582e5930d99d0039b52230a80254726f444fd31969`.
Spark image ID (all six ranks): `3ae622568eddeb4e6fe9aa3fa807465cd32f619e9e617ab43ab8e4482793c627`.
The first Spark export exhausted CUDA memory under unbounded Ninja concurrency;
RC1 rebuilt successfully with `CUTEAFD_RELEASE_NATIVE_BUILD_JOBS=1`.

Cargo check passed; workspace tests: 1424 passed, 170 ignored, no failures.
Scripts: 927 passed, 2 skipped, 193 subtests passed, no failing IDs. CPU
checks cover the source/build-script changes; subsequent changes publish docs
and reports only. V4.1 parity and `bench-ab` were skipped at TJ's direction.
No tag, main merge or registry image publication is part of this preparation.

The Release smoke matrix attempted all 28 cards: **27/28 pass**. Each
entry had a 600 s readiness limit and a 300 s benchmark limit, with at most
two entries on disjoint hardware. Each successful card used one warm launch.
Three initial setup failures were corrected: GLM Flash NVFP4 requires the
official FP8 companion, and its two-Spark entry must explicitly map rhea/moa
in both the launcher configuration and scheduling declaration. The initial
host mismatch collided with V4 Flash minimum workers. Only those three
invalid attempts were repeated; their original logs are retained.

| Model · quant | Minimum gate | Maximum gate |
| --- | --- | --- |
| V4.1 Flash · MXFP4 | PASS | PASS |
| V4.1 Flash · NVFP4 | PASS | PASS |
| V4 Flash · MXFP4 | PASS | PASS |
| V4 Pro · EXL3 K2 | PASS | PASS |
| GLM 5.3 · EXL3 K4 | PASS | PASS |
| GLM 5.3 · NVFP4 | PASS | PASS |
| GLM Flash · EXL3 K3.25 | PASS | PASS |
| GLM Flash · EXL3 tr3 4bpw | PASS | PASS |
| GLM Flash · FP8 | PASS | PASS |
| GLM Flash · NVFP4 | PASS | PASS |
| MiMo V2 Flash · FP8 | PASS | PASS |
| MiMo V2.6 Pro MOPD · MXFP4 | PASS | PASS |
| Qwen 3.8 · EXL3 K4.25 | PASS | PASS |
| Qwen 3.8 · NVFP4 | FAIL | PASS |

Qwen NVFP4 minimum (one RTX, local MTP3) fails launch: its MTP experts are
E4M3 `[640, 2560]`, while the NVFP4 backbone loader expects packed E2M1
`[640, 1280]`. It needs a separate FP8 MTP expert load/execution path.
The Spark maximum passes with speculation off; Qwen has no two-RTX head split,
so its maximum actually serves on one RTX plus four Sparks. The failed
minimum has no benchmark export; the generated README shows the available
NVFP4 Spark card, and this table records the required minimum's failure.

GLM 5.3 official FP8 remains outside PLAN's release scope (six-Spark serving
budget); Qwen official FP8 has no Spark package and is omitted as authorized.
V4 Flash minimum is native TP2 on two Sparks. GLM Flash FP8 minimum needs
four Sparks; its NVFP4 minimum uses two. MiMo Pro uses MOPD, replacing RL.
Both local and Spark NVFP4 packages, exact slices, MiMo A8 down projection,
GLM Flash KDA/head programs and Qwen MTP programs are included in RC1.

**Release blockers:** the failed Qwen NVFP4 minimum violates the all-green
matrix criterion, and scoped MXFP4 32-row tails remain absent from this
candidate with correctness gates open. Claude must resolve these or obtain
an explicit scope decision before TJ approves tagging/publishing.

Historical v0-to-RC smoke measurements are in the publication commit body.
They are descriptive, not matched A/B: checkpoint, placement and speculation
changes are identified there. V4.1 maximum and Qwen EXL3 minimum show sizeable
C1 drops in these single sessions. GLM Flash NVFP4 maximum also shows a large
prefill drop while moving from one RTX with copy-window drafts to the two-RTX
head split with DFlash2 and FP8 companion projections. These need review;
this preparation makes no performance-parity claim. V4.1 parity remains
deferred as directed.
Logs, matrix, placement snapshots and the full measurement table are retained
under `~/.cache/cuteafd/builds/release-v1/kit/`; gate logs are under
`~/.cache/cuteafd/builds/release-v1-gates/`.

### v1 regression follow-up (RC2, 2026-10-05)

The RC1 qualification above is historical. RC2 adds mixed-format local Qwen
experts: routed layers remain in the resident NVFP4 TP1 package, while the
MTP layer owns a separate resident FP8 TP1 package and prefill scratch. The
loader detects the MTP format from tensor headers, admits each package before
allocation, and keeps one copy of each layer. Golden NLL is unchanged with
MTP3 enabled; the lossless check matches all 128 greedy tokens and prefix
restores remain byte-exact. Spark MTP still requires a worker protocol/package
extension and is explicitly unsupported; the Spark card keeps MTP off.

The four requested Release smoke refreshes pass on clean runtime source
`01e7a8c5f9eb31fe35ce5726e3b719677c95ced5`: V4.1 Flash maximum, GLM Flash
NVFP4 maximum, Qwen EXL3 minimum and Qwen NVFP4 minimum. The published cohort
is **28/28 passing cards**, comprising these four RC2 exports and 24 retained
RC1 exports. This is not a rerun of the full matrix. Local untagged images:
coordinator `f6206446caa8f97b2d19b8e0b72996361b522849bf737e114f5a49a63890b9bd`;
Spark `261af7fa3ee0862d998e0c48e2d8a2b6d05034d0666e29406f6686c2d7f72c9d`
on ostrich, dodo, emu and kiwi. Both architectures rebuild the matching Rust
binary; the SparkInfer pin and tree lock remain unchanged at `f6bb38b`.
Cargo check and workspace tests pass (1425 passed, 170 ignored); script
tests pass (927 plus 193 subtests, two skipped), with no failing IDs.

All four new cards pass golden fidelity and byte-exact snapshot restores.
Qwen EXL3 and NVFP4 each match all 128 greedy tokens with MTP3 on/off.
The V4.1 smoke speculation verdict is informational: drafted and plain output
diverge at token 49 through verify rounding, while plain decode repeats
exactly. GLM's verdict passes its near-tie tolerance, but is not byte-identical
after token 27. Existing batch/verify numerical-invariance limitations remain;
the smoke pass does not claim byte-identical output across those paths.

V4.1 maximum uses the existing shared FP8 vocabulary head. Matched v0/RC1
ABAB controls do not reproduce the historical long-code C1 regression.
Qwen EXL3 matched BF16 ABAB also passes parity. Its historical default-speed
difference includes v0's decode-only FP8 projections with dual residency;
RC1 switched to BF16 projections to meet the single-copy precision gate.
Single-copy FP8 projections still miss that gate and remain opt-in. Restoring
the old default speed is not claimed by the same-precision comparison.

GLM Flash's old smoke warm-up could leave a measured prefill lane or its
expert accesses cold. Matched comparisons now record fixed prompt hashes and
zero cached tokens; smoke primes complete chunks plus the exact measured
prompt before timing an ordinary request, including normal cache retention.
The split loader also accepts BF16 block projections from the selected
checkpoint, quantizing before aligned slicing exactly as the unsplit loader
does. This makes `GLM5_FLASH_FP8_MODEL_ID=off` valid with NVFP4 head split;
the companion and precision defaults remain unchanged.

Measurements and conditions belong in the follow-up commit messages. Task
logs, reproducible configs and exports are under
`~/.cache/cuteafd/builds/release-rcfix/`. No release tag, image tag or registry
push is authorized by this follow-up. Scoped MXFP4 32-row tails remain
unmerged and unqualified; this work does not close that release blocker.

## Release v1 — priority plan (2026-10-02)

Everything after v0 lands as v1. Helpers: read AGENTS.md, then pick the top
open item; each names its branch (pushed WIP) and the next step. Merge green
steps into `work/p0`; tag `v1.0.0` when the list's top half is done.
The codex/v1 line (merged 2026-10-03, `work/codex-merge`) closed several
item-4 bugs and started items 7 and 10; commit messages carry its evidence.

1. **Device-driven Spark exchange, shared by every family** — branch
   [`work/v41-device`](https://github.com/tpurtell/cuteafd/tree/work/v41-device). Today every family does 2–3
   blocking host round trips per MoE layer (router ids D2H + sync, host-built
   request and RDMA post, host CQ poll before the reduce); no device-initiated
   networking exists. Design: GPU kernels write requests and set a ready flag;
   a host proxy thread posts pre-built WQEs (spins only while a step is in
   flight, parks on a futex when idle); replies land GPU-direct with a
   NIC-written completion flag and the reduce waits on the device; whole step
   in one CUDA graph. Order: DeepSeek V4 → V4.1 (adopt SparkIntake/GPU landing,
   device-side replica ownership) → GLM 5.3, GLM Flash, MiMo, Qwen. Then the
   Spark worker loop, then IBGDA (GPU rings the NIC doorbell) to drop the
   proxy. Projection (V4.1, `ae91c6a`): exchange alone C1 +5–7% / C4 +3–6%;
   with whole-step graphs C1 +10–15% / C4 +8–12%.
   State (2026-10-04, merged into work/p0 at db9ed89; all opt-in, default unchanged
   except the single-RTX head keeping one graph per verify width): `SparkDeviceLane`
   proxy, V4 `CUTEAFD_SPARK_DEVICE=1`, V4.1 `CUTEAFD_V41_DEVICE=1` (device-ordered
   verify passes: chained SM handoff between RTX, chained split head, engram uploads
   in the chain, per-layer staging, 60 s watchdog that logs lane sequences and exits).
   Both lanes device-ordered are consistent now (codex's attention-staging fence).
   Spark worker idle loop that waited on one connection (aa507bd) stalled device
   waves: reverted. Code case, warm, 325 W, one launch per arm: 2 RTX + 4 Sparks
   C1 190.4/192.4 → 194.6/194.1 (+1.5%), C4 flat; 1 RTX C1 −1%, C4 +0.5%: the
   exchange alone does not pay enough to be default. MiMo V2.6 Pro and GLM 5.3 on
   the exchange (+ whole-step decode graphs, `work/device-mimo-glm`): no gain
   (MiMo host C1 80.4/81.5 vs device step 77.0/79.8; GLM ~65 both) — their
   segments are GPU-bound. Write mode (`CUTEAFD_SPARK_WRITE=1`) launches but the
   pass sticks on the written flags (open). The remaining V4.1 lever is whole-step
   capture; blockers in order: index-selection shape vs context length, per-request
   pointer fingerprints (sparse/index), host-built per-layer metadata (one per-pass
   arena would remove it), 1-RTX GPU landing at 3.6 GB/s after weights load.
   FP8 draft head (`CUTEAFD_V41_FP8_HEAD=draft`, E4M3 copy, W8A16 GEMV 882 → 416
   µs per 129280-row head): parity, 3 interleaved sessions per arm, base → fp8d:
   2 RTX C1 183.6 → 186.4 (1.016), C4 489.0 → 512.0 (1.047), C16 1413.7 → 1442.2
   (1.020), weighted decode 0.997; 1 RTX C1 1.014, C4 1.039, C16 1.087, decode
   1.012. Remains opt-in: the BF16 target head stays resident beside the FP8
   draft copy, so this mode violates single residency. Claude accepted the
   `all` quality result on `work/v41-fp8head`; its new single-copy residency
   missed the dual-RTX C1 performance gate, so BF16 stays default.
   Kit:
   `~/.cache/cuteafd/builds/v41-device` (STATUS.md, build-coord.sh/build-spark.sh,
   v41-ab3.sh, v41-c4.sh, run-parity.sh).
2. **Whole-step graphs** — MiMo's per-layer segments are merged and opt-in
   (`DECODE_GRAPHS=on`, [`work/mimo-graphs`](https://github.com/tpurtell/cuteafd/tree/work/mimo-graphs)); flat today,
   they pay once item 1 removes the host hops. Same for every family.
3. **V4.1 step wins** (from the critical-path note): device-side draft
   acceptance (~0.8 ms host gap per round, up to +3%); FP8 target head
   (draft head done, see item 1; `all` quality accepted by Claude,
   single-copy opt-in, dual-RTX C1 speedup below promotion bar); one host
   thread serves both lanes
   (26–43% of wall time in CUDA calls) — item 1 removes most of it.
4. **Model-specific issues found** (fix in v1, not essential for v0):
   - GLM 5.3 Flash (likely GLM 5.3): a JSON-schema request whose grammar
     accepts the stop token keeps decoding; xgrammar `fill_bitmask` then fails
     and the whole batch fails. Fixed (`f3c7505`): every stop token ends a
     speculative grammar proposal.
   - MiMo V2.6 Pro: two-lane prefill runs only without the head split, so 8K
     prefill is slower on 2 RTX (4.79 s) than on 1 RTX (3.18 s). Fixed
     (`bf1a061`): both head-split GPUs pipeline the lanes; short independent
     requests also pair on the two lanes (`5b191f2`).
   - V4.1 on 1 RTX: startup is serial (Sparks load all 40 layers at ~0.4 GB/s
     each, ~205 s, including 5 the RTX holds; then the coordinator). Start the
     coordinator first, skip RTX-held layers on the Sparks, speed up the Spark
     layer load. 2 RTX: 108 s. Coordinator-first auto placement on 1 RTX
     landed (`be7049f`); Spark layer read speed is still open.
   - DeepSeek V4 Flash: the v0 native expert format refused 2 Sparks. Native
     TP2 now uses legacy expert requests (`e4a8055`) and is ledger-qualified
     (`443dc7a`); the v1 minimum uses two Sparks. V4.1 TP3 fits per
     `cuteafd plan` but is unqualified.
   - Benchmarks: reasoning-effort panel re-run after the pool back-off fix;
     turn-end cache check gates restores against their snapshot (4i); the code
     sandbox requires user/net/PID namespaces (`fd74aaf`; coordinators run
     with `docker/seccomp-code-bench.json`); tool-eval-bench reaches images
     with the next `./build.sh`.
   - From the v0 Release smoke matrix (10 of 22 cards fail the gate;
     logs in `~/.cache/cuteafd/builds/v0/kit/smoke-state/`):
     a. Forced tool calls: GLM 5.3 EXL3 (min, max) crashes the coordinator
        ("matcher terminated after accepting the stop token"); GLM 5.3 Flash
        EXL3 max, tr3 4bpw min/max and MiMo V2.6 Pro min/max abort the stream
        mid-response. Same grammar/matcher path: stop when the grammar accepts
        the stop token, never fail the batch. Fixed in `f3c7505`; rerun the
        smoke cards.
     b. A worker failure mid-stream drops the SSE connection with no error
        event (all families). Fixed in `f3c7505` (one structured error event);
        MiMo reports a fatal cause to every accepted request (`50b2608`) and
        retains native owners until queued work provably drains.
     c. Speculation not lossless: V4 Pro EXL3 K2 dSpark diverges at token 4
        (1.95 nat), C4 ≠ C1 at token 15; GLM 5.3 Flash tr3 DFlash2 0.84 nat;
        GLM 5.3 EXL3 0.57 nat. Suspect multi-row verify numerics/state.
        GLM Flash: FP32 MLA decode partials + K64 EXL3 for 2–16 rows
        (`262cf29`) cut serial-vs-verify KL 0.0060 → 0.0041; rejected-suffix
        causality checks pass. Byte equality and C1/C4 invariance open.
     d. Batch invariance: C4 ≠ C1 greedy on V4 Pro, GLM 5.3, GLM 5.3 Flash.
     e. NVFP4 local experts on one RTX: was implicit expert paging, not a
        slow kernel. Fixed (`4c02f2f`): local experts are resident by
        default (Qwen NVFP4 decode 21 s → 8.7 ms/step); paging needs an
        explicit `--expert-window`.
     f. GLM 5.3 Flash has a two-GPU head split (`work/glmf-split`, `glmf2`
        programs: half the KDA/MLA heads and their state, half the dense /
        shared-expert intermediate; default with RTX_GPUS=auto/2). 2 vs 1 RTX +
        4 Sparks: EXL3+DFlash2 C1 code 159 -> 168 tok/s, NVFP4 C1 76 -> 83,
        8K prefill equal; golden NLL 2.4073 -> 2.4054. Qwen still has none
        (two-GPU requests serve from the first GPU).
     g. Qwen 3.8 EXL3: 84 tok/s with 4 Sparks vs 261 on one RTX alone.
        Resolved placement (`21201b6`, `120f4e7`, `5ff0602`): qualify the
        supported EXL3 K4.25 package for resident local experts when weights,
        serving reservations, MTP and the requested KV pool fit the selected
        GPU. The planner includes resident experts and leaves unused GPUs
        empty; the launcher checks live available memory before choosing.
        `EXPERT_BACKEND=spark/local` preserves explicit placement. The old
        comparison also changed native MTP depth; matched backend-only runs
        still favor local experts. Real CPU launcher admission also selects
        local with explicit MTP. Qualified local EXL3 now defaults to native
        MTP3 with one FP8 head shared by target and drafts; explicit settings
        retain precedence. Spark MTP is unsupported (workers serve backbone
        experts only). C1/C4 and low-margin verify rounding still differ;
        the current lossless gate permits proven rounding, never a state bug.
     h. Prefill gets worse with more hardware: V4 Pro min 879 tok/s (9.2 s
        TTFT) vs 2,438 max; MiMo Flash max 2,899 vs min 5,877; MiMo Pro max
        1,754 vs min 2,741 (two-lane prefill off under the head split).
        Refreshed on `adfd821`: V4 Pro's large historical gap no longer
        reproduces; maximum remains faster. The launcher now honors explicit
        `RTX_EXPERT_LAYERS` (`a181a6a`). Remote-only backbone experts miss the
        declared TTFT improvement bar and give mixed decode results, so keep
        automatic placement. Golden fidelity is still unavailable; existing
        speculation and C1/C4 divergence remain open.
        MiMo Flash already uses the shared multi-lane head-split fix. Its
        inherited three-lane default was qualified for Pro; two lanes improve
        Flash prefill on both reference layouts and pass matched C1 with
        byte-identical output (`75bfb91`). Keep Pro at three lanes and preserve
        explicit overrides. Flash's maximum still trails its improved minimum:
        head split overhead remains open. Measurements and conditions are in
        the scale-anomalies commits; no shared native or exchange kernel edits.
     i. V4 / V4.1 turn-end prefix-cache restores are byte-exact (fixed in the
        check, `0f65c9b`): the old check compared a restored turn with a cold
        recompute, and V4 Flash / V4.1 prefill does not repeat bit for bit
        (Spark FP32 atomic expert reduction at 256+ rows); its turns also
        ended at EOS with one row to compare. The check now judges each
        restore against its own snapshot (turn rows, prompt-snapshot
        reference, decode step after the turn restore) and reports the cold
        recompute only. Smoke V4 Flash min and V4.1 min: prompt and turn end
        2 rows byte-identical, cold recompute differs. Deterministic prefill
        stays open: an ordered serial-slice reducer passed component gates
        on a private codex branch (Flash TP4 only); not merged. V4.1's unchanged
        base also produces different long-context reasoning across launches
        with fixed dSpark drafts (34,745-token prompt, 2,048-token output);
        cold/warm restores match within each launch. Long text comparisons
        cannot qualify decode graph changes until cold prefill is deterministic.
     j. MiMo V2 Flash fidelity is the weakest that passes (KL 0.10, top-1 82%).
        Opt-in BF16 expert-input Spark packages (`EXPERT_INPUT=bf16`,
        `CUTEAFD_*_FP8_MOE_BF16_FAMILIES=mimo`) improve it; default stays FP8.
     k. Qwen 3.8 FP8 has no Spark expert package (173 GB, no one-RTX fit).
     l. Fixed on codex/v1: generic families reject contexts beyond the
        compiled index extent before allocation; `HOST_CACHE_BYTES=auto`
        is bounded by live host memory; cancelled stage chains drain before
        staging is reused; generic KV admission waits (FIFO) under
        transient pool pressure instead of rejecting.
     m. V4.1 cold prefill: large index-selection / attention-query graph
        entries are evicted between encoder and replay shapes, so warm
        requests still capture graphs. Open.
5. **Spark expert kernels**: MiMo V2.6 Pro TP6 prefill is Spark-bound (~35 of
   ~42 ms per layer); GLM 5.3 verify is bound by distinct expert reads; NVFP4
   W4A16 GB10 prefill (14.3 vs EXL3 9.1 ms/layer TP4).
   GLM 5.3 (2026-10-03, `work/glm-perf`): 8K prefill is ~3.0 s on min and max
   alike because both are Spark-bound — worker kernel time per 2752-row wave
   (3 lanes) is 11.7 ms at TP4 width 512, 11.0 at TP6 width 384, 7.6 at width
   256, so six Sparks save only ~5% Spark time (the width-384 package ran
   128-wide tiles). Fork f6bb38bc (dynamic tile claims, FP8 wire input,
   192-wide TP6 tiles; bit-identical) cuts live waves to 11.24 / 9.87 / 7.05
   ms and Spark busy per 8K to 2.62 s (TP4) / 2.30 s (TP6); still Spark-bound.
   Served 8K TTFT with E4M3 MLA + these packages vs v0.1.0 (2026-10-04, one
   launch per arm): max 2.96-2.99 -> 2.49-2.64 s, min 3.08-3.11 -> 2.82-2.91 s;
   C1 code flat (max ~68, min 58-61; text changes with the MLA numerics); C4 is
   dominated by within-batch greedy divergence (item 4d) in every arm. The
   head split only moves the wait from the GPU to the Sparks. GB10's
   wave is bound by bytes (FC1 BF16 input gathers per N tile, FC2 partial
   round trip ~2 ms, top-k sum 1.6 ms) and the FC1 rotation, not MMA rate.
   Coordinator GPU-only 8K prefill is 2.7 s, half of it the sparse MLA prefill
   kernel. E4M3 MLA prefill (`work/glm-mla-fp8`, merged, default e4m3-p2;
   real-expert TP4 gate: KL vs golden 0.0364 -> 0.0379, NLL 2.4680 -> 2.4717,
   deterministic) cuts the GPU-only 8K prefill 2.89 -> 2.57 s, but 8K with
   Sparks stays 2.95 s (GPU wait 2.11 -> 1.45 s, Spark wait up): the GB10
   wave is the bound. Lanes 2 or 4 lose to 3.
   Verify layouts (busiest-rank expert reads, uniform routes): TP6 beats
   TP2xEP3 up to 16 rows (1.50 vs 2.02 expert-equivalents at 1 row, 10.8 vs
   11.2 at 8) and loses by 1-8% only at 32-64 rows; keep TP6. DFlash2 at max
   (TP6 + split, code C1): adaptive 66-68 tok/s vs fixed 7/5/3 at 65/62/61;
   offline trace scoring with the measured TP6 table: adaptive 72.9 vs best
   fixed 62.7 (oracle 83.3) — the policy is not the limit.
6. **RTX 5090 audit and claim**: hard-coded `4*188` grid clamps and the
   per-tensor FP8 GEMM grid sized for 188 SMs; one SM120 build must serve both.
   Expert quantizer grids now come from each engine's GPU (`6d4ea7a`).
7. **Phase 6 placement planner** (incl. cold components such as the vision
   encoder on a Spark) and **multimodal input** (official encoders only).
   **Joint serving capacity** (TJ): default C16 with 20 front-state slots and
   a common 2,097,152-token GPU KV pool; reserve KV, workspaces, graphs,
   transport and drafters per physical GPU (97% of total minus existing use)
   before onboarding expert layers; report shortfall instead of silently
   shrinking. Pure resolver: `cuteafd-core`/`cuteafd-loader`
   `serving_capacity`; `cuteafd plan` describes cache storage. MiMo admits
   its runtime reservations before loading (`mimo_v2/admission.rs`). Next:
   startup consumes the same resolved plan for every family.
8. **NVFP4 follow-ups**: native per-tensor FP8 decode with static scales.
   **V4.1 NVFP4 fidelity and wire (deferred to v3, 2026-10-09):** the current
   V4.1 path already uses the 44-slot W4A4 family for every row count; older
   W4A8/opt-in descriptions were stale. First preserve each expert's static
   FC1 input_scale instead of the layer maximum (FC2 is already per-expert).
   Then evaluate BF16 versus FP8 wire plus per-expert FP4 quantization; a
   shared-scale NVFP4 wire is not checkpoint-exact. The cooperative kernel's
   low occupancy is the stronger C1 hypothesis, since route packing already
   quantizes FC1 once per route, not once per FC1 slice. Gate: quick fidelity
   and golden/KL vs the official reference, 8K prefill and C1/C4/C16 on the
   natural-min and max layouts. Keep today's default until hardware gates.
   **W4A4 decode/verify rows** (TJ): decode rows (1–16, incl. speculative
   verify) run W4A16 even on W4A4 checkpoints. Measure W4A4 decode with the
   activation quant fused into the GEMV/MMA prologue on one NVFP4 model (C1
   step, KL); bandwidth-bound either way, so expect parity — if so, make W4A4
   decode the default for checkpoints that declare it (one numerics path from
   prefill through verify).
9. **Activation precision policy** (TJ, 2026-10-02): converge on **A8
   wherever quality holds** (FP8/MXFP8 activations on tensor cores, the speed
   lever for prefill and wide verify) and **A4 only where the checkpoint
   declares it** (NVIDIA ModelOpt NVFP4). Every A8 switch is gated on golden
   NLL/KL (≤0.005 nat) plus a tool-eval/agentic check, per model.
   - EXL3 × A8: EXL3 trellis experts (V4 Pro, GLM 5.3, GLM Flash, Qwen) run
     A16 today. Checked 2026-10-03: nothing upstream runs standard (MCG) EXL3
     with A8 — master's W4A8 trellis decodes only QSRT; brandonmusic's PR #342
     (MCG->E4M3, SM120 TP4, source-available licence, expert-shared suh) does
     not fit our checkpoints. GB10 measured full-rate F16 MMA (124.8 TFLOPS,
     E4M3/INT8 248), and an INT8 A8 prototype (fork `cuteafd/exl3-a8`, local;
     INT8 weights 0.9-1.6% rel error vs 3.7% for E4M3) saves only ~7% of a
     GB10 wave: not built. The wave is byte- and rotation-bound (item 5).
   - MXFP4 experts (V4.1 already W4A8; MiMo V2.6 Pro W4A16): A8 prefill for
     MiMo Pro (Spark-bound prefill), and MXFP4 × MXFP8 MMAs for both.
   - FP8 experts: extend W8A8 (MiMo GB10 gate/up) to the down projection and
     to RTX-local experts where KL allows (Qwen FP8 local was +0.024: needs
     finer activation scales).
10. **Resident weight representations** (TJ, 2026-10-03): one resident
    BF16 or FP8 representation per weight set; preserve each checkpoint
    tensor's precision by default; calibration-free conversion only as an
    explicit option; drafter precision is chosen by emitted tok/s and memory.
    Landed: MiMo resolves head/O/drafter formats from headers; MiMo V2.6
    Pro defaults to single-copy FP8 head/O/DFlash (TJ-approved exception;
    `MIMO_WEIGHT_POLICY=checkpoint` opts out). Qwen and GLM Flash now have
    compact single-copy FP8 consumers; GLM/GLM Flash DFlash defaults to FP8
    (`SPECULATOR_FP8=off` retains checkpoint BF16).
    **GLM Flash precision recheck (2026-10-04, Claude decision):** resolve
    launcher defaults after the actual coordinator split is selected.
    One serving GPU uses row128 FP8 KDA and an FP8 head; two-GPU head split
    uses BF16 KDA/head. Explicit current or deprecated precision keys win
    independently. The matched EXL3 K3.25 recheck clears the single-GPU
    quality/C4 bars; FP8 under the split misses both, so remains opt-in.
    Conditions and both tables: `docs/models/glm5_flash.md` and
    `benchmarks/glm5_flash/2026-10-04-fp8-recheck/comparison.json`.
    **Split regression audit (same recheck, no new hardware run):** the
    FP8 arm's warm and two timed 8K requests have essentially constant
    coordinator GPU wait while expert wait rises on each request. Its warm
    prefill is faster than BF16; the timed aggregate loss is dominated by
    Spark expert wait, not evidence of an equally large KDA compute loss.
    Both arms use the same prefill shape. Split KDA directly consumes FP8
    weights (`glmf2_kda_w8_m4096`); the generated library loads on every CUDA
    device. No BF16 re-conversion or replicated full-head KDA work found.
    All 238 KDA projection tensors in the primary EXL3 and official FP8
    companion snapshots are byte-identical before conversion. Column
    slicing of KDA O preserves the 128-K scale-block boundaries. Split
    attention partials still round to BF16 before the peer sum: interaction
    with FP8 quantization is a plausible, unproven contributor to top-1
    loss. Worker logs show no explanatory timeout/stall; per-request route
    distributions and Spark clock/thermal samples were not recorded.
    No clear fix qualified: keep split FP8 opt-in. Next discriminating
    measurement is one matched D/F launch with identical 8K token IDs,
    worker timing/route distributions and Spark clocks, plus KDA/head
    precision ablation if quality remains below the split promotion bar.
    **Fresh split profiling:** a warmed matched launch per arm did not
    reproduce the large prefill/C4 losses, while the golden quality delta
    reproduced exactly. Nsight timelines confirm half-head FP8 KDA on both
    GPUs, with unchanged peer traffic and synchronization counts. W8A16
    KDA projection compute is locally slower at wide prefill and verification
    shapes. Row128 head slices preserve whole-weight payloads and scales
    byte for byte. KDA-only/head-only ablations localize most extra KL to
    KDA. Golden scoring uses a one-token prefix and 64-row verification
    chunks, so the half-head GEMV tuning and chunked-recurrence window do
    not explain that delta. KDA partial rounding is being tested independently;
    MLA and dense/shared FFN partials also round before the peer sum.
    Measurements and conditions are recorded in the profiling audit commit.
    **Split investigation handoff (2026-10-05):** three interleaved BF16
    versus full-K KDA token-row FP8 launches qualify the measured EXL3 K3.25
    two-RTX/TP4 opt-in under TJ's paired quality/C1 bar. Recommend promotion
    to Hugh/TJ, but leave defaults unchanged. C1 consistently improves;
    C4/prefill are parity to modest gains, agentic was not measured. Original
    large prefill loss does not reproduce in FP8; synchronized ~440 ms
    worker-response gaps instead occur in a BF16 launch with identical routes
    and near-stable expert execution. No overlapping serving/build container
    found in retained lifecycle logs; incomplete host/fabric history means
    contention is not completely excluded. Coordinator/transport stalls
    remain a separate open issue, not a precision verdict. Full-MLA/full-FFN
    row prototypes are retired. Conditions, medians/spreads, paired quality,
    opt-ins and limits: `docs/models/glm5_flash.md`. Hugh owns further GLM
    Flash work (issue #1); no new broad experiments in this task.
    V4.1 `all` now releases BF16 and shares a single FP8 vocabulary head
    across target and dSpark. Claude accepted its target-head quality;
    dual-RTX C1 missed the promotion bar, so BF16 stays default. `draft`
    retains dual residency
    and its earlier parity does not qualify a target-head conversion.
11. **Parked**: Spark-side reduce-scatter ([`work/spark-reduce`](https://github.com/tpurtell/cuteafd/tree/work/spark-reduce),
   +3% one rail, +9–12% two rails at 200G); split intake
   ([`work/split-intake`](https://github.com/tpurtell/cuteafd/tree/work/split-intake), slower). Revisit only on new evidence.
12. **Housekeeping**: prune agent test images on raptor; delete
    `~/.cache/cuteafd/builds/{n10-rel,bisect-rel}` on ostrich (root).

## Release v3 scope (decided 2026-10-09)

TJ: two key items, both urgent right after v2.0.0.
1. **Multi-GPU coordinator placement and memory management for every family.**
   One shared design replaces today's per-family patchwork:
   - **TP2 RTX expert layers.** With two RTX, each RTX-resident routed-expert
     layer runs as two halves, one per GPU (split on the intermediate
     dimension). Each half is added into the per-layer all-reduce the head
     split already does, so it costs no extra peer hops. Today only V4.1 has
     TP2 RTX experts, in its own code; MiMo, GLM 5.3, GLM Flash and V4 keep
     routed experts on GPU0 or the Sparks, leaving GPU1 mostly empty
     (MiMo Pro 2 RTX + 6 uses 15 of 95 GiB on GPU1).
   - **Attention handling.** Use the head split where it pays (measured: V4
     decode -10/-12%, MiMo Pro -41% coordinator-only, GLM 5.3 -9.5%). Add
     Qwen's head split. V4.1's choice is settled by item 2.
   - **Per-layer GPU ownership (TJ, 2026-10-09).** The layer-range split
     (an RTX owns a contiguous layer range: attention, KV and that layer's
     work, one hidden-state hop at the boundary) is a general capability for
     every family, not V4.1-only; TJ has used it productively for GLM Flash.
     It composes with the head split per layer: the solver assigns each layer
     `Whole(gpu)` or `HeadSplit` by attention type and measurement (e.g.
     head-split full MLA/GQA layers, whole KDA/SWA layers on one GPU). KV
     follows ownership; layer ownership is a memory-balancing lever alongside
     TP2 experts.
   - **Memory management.** One admission solver for every family:
     - reserve the KV pool first (2M PRO / 1M <=32 GB);
     - then graphs and workspaces;
     - then RTX expert layers, TP2 across both GPUs, with non-split items
       (drafter, encoders) moved to balance the two GPUs;
     - the planner equals the runtime admission, with a test per family.
       Measured gaps to close (rc2 logs, `builds/qwen-vision-floor/`,
       `builds/v4-small-extent/hardware/admission-comparison.json`): Qwen
       local over-admits 5.35 GB (EXL3) / 6.72 GB (NVFP4), since the planner
       charges a 512 MiB graph constant against a 5.0 GB runtime graph set
       and omits 1.06 GB of NVFP4 package scratch; V4 Flash over-charges
       ~84 MB (CUDA context/module estimate); MiMo graphs-on bound 664 MiB
       vs 512 MiB planned; GLM Flash's startup graph set can exceed its
       1.5 GiB allowance. The planner must share the runtime's graph-set
       definition and package-scratch formulas instead of constants.
     - Spark ring admission exact per family: endpoint count (GLM Flash
       can open more than 2 prefill transports) and ingress dtype (MiMo's
       BF16 expert-input override, V4.1 NVFP4's BF16 rows) sized from the
       real geometry, enforced on every worker path. rc3 enforces it for
       V4.1 only (work/spark-vision-ring) and logs the rest.
     - Shared scratch sized from the programs the family launches, not every
       program name in the image (V4 `engine.rs` ~378-388 scans all names;
       dispatch uses only self.family/split_family ~295-309). On the release
       image a V4 Flash server reserves 2 x 782 MB for GLM Flash's
       `glmf_kda_w8_m4096`; selecting its own programs frees 1.07 GB on a
       32 GB card. Change the runtime allocator and the loader helper
       (`v4_workspace.rs` ~99-120) together, keeping the unsplit target
       family for dSpark in split mode; planner-only filtering is unsafe.
       Audit the other families for the same union.
   - **Design:** "v3 placement: design" below.
   - **Starting point:** work/v4-placement b2f26af9 (pool-first solver,
     planner/runtime equality) and `builds/v4-placement/SUMMARY.md` (V4 TP2
     design and kernel/loader audit). V4 Flash/Pro measured in rc1:
     KV pool 581K-1.39M tokens because experts are placed first, GPU1
     13-24 of 90 GiB used.
4. **Spark-free layouts as first-class options (TJ, 2026-10-09).** Any layout
   with no Sparks that fits is supported: GLM 5.3 Flash and V4 Flash on 2 RTX
   at a reduced KV pool, using item 1's TP2 RTX experts and attention
   placement (today's head split leaves every routed expert on GPU0). Qwen on
   1 RTX already serves (525,568-token plan for EXL3 K4.25). Input:
   `builds/no-spark-layouts/` (per-GPU expert bytes, pools at 2M/1M/262K).
6. **Shared resource-priced draft policy from V4.1 (TJ, 2026-10-09, priority).**
   AFD is an MoE technique and Spark expert traffic dominates step time, so
   adopt V4.1's dSpark cost model (`cuteafd_core::dspark_policy`: per-layer
   `alpha + beta*rows + bytes/bandwidth` from route-history traffic
   forecasts, separate RTX/Spark resource classes, Huber fits with
   forgetting, online per-position calibration) as the shared policy for
   every speculative family, with per-layer timing and committed-route
   plumbing added to each engine. Replaces `CycleCost`'s row table; the
   per-concurrency buckets on `work/draft-policy-v2` are interim. Design:
   "v3 draft policy: resource-priced, shared" below; order: "v3 roadmap".
7. **API gateway and sessions (TJ, 2026-10-09; building now on
   `work/api-gateway`).** Complete, spec-faithful Anthropic Messages and
   OpenAI Responses so Claude Code and Codex CLI use cuteafd directly (no
   LiteLLM), model aliasing and listing, server-side web search (Exa plus a
   no-key option), and the OpenAI Realtime API (crucial; text first, audio in
   through an audio-capable backend, TTS seam), all on one session layer with
   an upstream backend for GPU-free testing. The APIs are the payload; later
   session operations (steer/compact/history mutation/fork/splice) build on
   the same session layer.
5. **Qwen FP8 KV cache (TJ, 2026-10-09).** Qwen stores BF16 K/V records
   (2,048 B per row) and BF16 index keys, the most KV bytes per token of any
   family; Qwen NVFP4 min plans 217,856 tokens, below the 256K agentic
   floor. Add an FP8 K/V record option (E4M3 with per-group scales, as GLM's
   latents and MiMo's int8 records already do), planner charge included,
   gated on golden/fidelity and the quick A/B on 1 RTX local and min/max.
   FP8 or NVFP4 options for the MLA/DSA families (GLM, GLM Flash, V4) are
   skipped for now (TJ).
3. **V4.1 NVFP4 numerics and wire** (deferred by TJ, 2026-10-09). The Spark
   kernels already run W4A4 (mxf4nvf4), but FC1 uses one layer-max input
   scale instead of each expert's calibrated `input_scale` (~6x spread), and
   the coordinator sends BF16 rows (10,240 B) the Spark re-quantizes per
   route. Per-expert exact scales first, then the wire (BF16 / FP8 / FP4
   shared-max), gated on real activation captures and fidelity vs the
   official V4.1 reference. C1's 16% gap to MXFP4 is likely the cooperative
   kernel's occupancy, a separate item. Input: work/v41-nvfp4-wire,
   `builds/v41-nvfp4-wire/SUMMARY.md`.
2. **Retire ds41rt: V4.1 moves onto shared infrastructure** (below, "First
   after v2"). V4.1's TP2 expert layers and memory placement are inputs to
   item 1; build the shared versions once, not twice. Design: "v3: retiring
   ds41rt" below.
8. **`cuteafd plan --files` follows the effective serving config** (found
   2026-10-10 in the MiMo sliced-checkpoint check).
   - The bug: for the serving config, `plan --files` leaves out the MTP files
     unless `--include-speculator` is given. It always leaves out a bundled
     DFlash drafter (`dflash/`). A user who slices by the docs gets a
     checkpoint that can't start the default speculator.
   - The fix: by default, the file list follows the launcher's effective
     speculator, drafter snapshot, vision and audio settings.
   - Gate: for every family's default config, a slice from `plan --files`
     starts and serves with the default speculator.
9. **Ship a starting config per family** (found 2026-10-10). The repo
   tracks `cuteafd.config` (V4.1) and `examples/configs/` (V4.1 TP/EP
   layouts) only.
   - The problem: the GLM 5.3, GLM Flash, MiMo, Qwen and V4 configs the
     release cards use exist only in the release kit. A user starting from
     the README gets no drafter for GLM 5.3, because the launcher's bare
     default is off while the release configs set
     `SPECULATOR=dflash2` / `incoai/GLM-5.3-DFlash2`.
   - The fix: add `examples/configs/<family>-<quant>.config` for each
     release card, with no host details. Make the launcher's bare defaults
     match them, and link each one from the family doc.
10. **NVFP4 W4A4 guards and wire for GLM, GLM Flash and Qwen** (found
    2026-10-10). These families already use each expert's own
    `input_scale` and `weight_scale_2`, unlike V4.1 (item 3).
    - **Latent bug:** FC1 quantizes activations with the gate projection's
      `input_scale` but dequantizes the up half with the up projection's
      (`_nvfp4_moe_a4.py` `StreamNvfp4GateUpA4`). Every expert in the four
      NVIDIA NVFP4 checkpoints has bit-identical gate/up scales, so output
      is unaffected today.
    - **Fix:** a load-time error when they differ.
    - **Wire:** Spark W4A4 prefill quantizes twice (BF16 → FP8 K32 wire →
      FP4). Measure fidelity and 8K prefill with BF16 wire rows for the
      W4A4 steps.
    - **Fidelity blind spot:** quick fidelity scores decode-shaped steps
      (≤ 8 rows), which run W4A16, so W4A4 prefill (> 1,024 rows on GB10)
      is only seen by the full tier's prefill-shaped pass. Give NVFP4
      release cards a prefill-shaped check.
11. **Replicated-latent KV under the head split (TJ, 2026-10-10).** Under the
    head split, MLA/DSA/CSA latents are stored in full on both GPUs, because
    every head reads the whole latent.
    - **Cost:** GLM 5.3 max spends 65 GiB of GPU1 on the copy and caps at a
      1.30M pool. V4 Flash/Pro lose 4-5.6 GiB per GPU, about 1-2 TP2 layers.
      GLM Flash's DSA layers are affected too.
    - **Two attention placements, both offered, chosen per layout:**
      - **Token-split latent** (`attention=context`; **default**). Each GPU
        holds half the tokens, page-interleaved. Projections stay head-split
        and the query is replicated, which is tiny at decode. Each GPU runs
        all heads over its token shard, and the partials merge by
        log-sum-exp.
        - KV is halved, and each GPU reads half the latent per decode step,
          so it is optimized for C1.
        - DSA: indexer scores are token-sharded, with a global top-k by
          exchanging candidate (score, index) pairs; ties break on token
          index, for determinism.
        - The merge changes summation order, so it needs its own fidelity
          gate.
        - Prefill merge traffic needs overlap, or the head split for short
          contexts.
      - **Layer ownership** (`attention=layers`). Each attention layer's KV
        lives on one GPU, alternating. Two lanes run one layer apart, so
        both GPUs stay busy. There is no merge and no duplicate, so it is
        optimized for C > 1.
        - It uses P3's `Whole{gpu}` and hops, and P7's per-layer executor.
    - **Launcher:** `ATTENTION_PLACEMENT=context|layers|heads` (`heads` is
      today's replicated head split, kept for A/B). The solver charges each
      exactly, and a card records the choice.
    - **Per-family default (TJ, 2026-10-10):** `context` wherever its
      measured C1 is at least `layers`' C1 at the 2M pool on the family's max.
      Otherwise that family defaults to `layers`. The quick A/B decides, and
      the family's doc records the choice and its numbers.
    - **Order:** design first; then GLM 5.3 (the biggest win, and P9's
      ranges-vs-split choice becomes these options), then V4 Flash/Pro, then
      GLM Flash's DSA layers. V4.1 keeps its 20/20 ranges unless measured.
    - **Design:** "v3 attention placement without latent replication"
      below (PRs `K0`-`K7`, decision gates `D-GLM`, `D-V4`, `D-GLMF`).
    - **Gate:** the family golden and quick fidelity (merge numerics); the 2M
      pool at max; C1 for `context` at least head-split C1; C8/C16 measured
      for both options.
Gate per family: golden/fidelity, then the quick A/B at the 2M operating
point on the min and max reference configs. Requalify each family's cards
as it moves.

## v3 roadmap (2026-10-09)

One ordered plan across the three v3 designs below: placement (item 1),
retiring ds41rt (item 2) and the shared draft policy (item 6). Steps are
named by their design (`P` placement PRs, `S` ds41rt stages, `D` draft-policy
steps) so the sections below stay the reference for each step's content.
Sizes: S under two days, M under a week, L more. Three lanes run in
parallel; hardware gates are serialized on the cluster and ordered here by
value (the measured GLM Flash regression and V4's 603K pool first).

**Merged, so built once:**
- `S4a` + `S4b` (V4.1 experts and admission onto shared) **is** `P12`. One
  branch, one gate.
- `S1a`'s `Speculator` trait and `VerifyCost` are gone; V4.1's policy *is*
  the shared one (`D0`, `D1`), and `S1a` consumes the shared binding.
- `S0`'s expert type moves are `P4`'s prerequisite (`NativeTp2` is V4.1's
  `RankWave` over shared `ExpertLayer`/`ExpertShard`).
- `P2`'s `GraphSet` (shapes and bytes, planner-visible) is the warm list
  `S3`'s `GraphBank` consumes; define it once in `P2`.
- The ds41rt design's `ShardedVocabulary` (its "what `shared/` must grow"
  item 9) is the placement design's head shard ("shard the vocabulary,
  solver-chosen ratio"); it lands with `P5`.
- `P13` (delete `family_costs` rows) is folded into each family's solver port.
- `P3`'s hop primitive is `S4c`'s; `S4c` follows `P7` (the first generic
  per-layer executor) rather than inventing a second one.
- `Fp8Layer::bytes_for` and the EXL3 manifest residency are the one source of
  per-rank expert bytes for both `ExpertCost` (`P1`) and `slice_bytes` (`D2`).
- v3 item 5 (Qwen FP8 KV) is `P10`'s `KvDemand.format` input. Item 3 (V4.1
  NVFP4) stays deferred and off this roadmap.

| # | step | lane | size | after | gate |
|---|---|---|---|---|---|
| 1 | `D0` geometry-parameterized core (`dspark_policy` -> `draft_policy`), V4.1 binding passes its geometry | D | S | - | identical decisions on recorded observations; V4.1 golden; one A/B pair V4.1 min |
| 2 | `S0` invert dependencies: `CudaCopyEngine`, expert types, `ScoreRows`, V4.1 golden entry | S | M | - | tests; byte-exact golden V4.1 / V4 / GLM Flash EXL3; one A/B pair V4.1 min |
| 3 | `P1` placement module, `PoolPolicy`, `solve`, V4 port, `planner_equals_runtime_deepseek_v4` | P | M | - | equality test; V4 Flash/Pro A/B at 2M (603K -> 2M pool must not cost C1) |
| 4 | `D1` shared plumbing: `LayerClock`, `RoundRoutes`, `RoundClock`, `Evidence`/`DraftSource`, observation builder; V4.1 binding onto them | D | M | 1 | tests; V4.1 golden; A/B min |
| 5 | `P3` `ResidualHome`, hop primitive, `LayerMode` in `Placement` | P | S | 3 | unit tests, no behavior change |
| 6 | `D2` GLM Flash onto the shared policy (events, routes, geometry, selector prior, Platt) | D | L | 4 | matched-prompts card: fresh C1, **C1 after a C16 sweep**, C4, C16, emitted tok/s, min and max, EXL3 and FP8 |
| 7 | `P2` exact items: `RuntimeInventory::measure`, context table, `GraphSet`, `ProgramSet`, package scratch | P | M | 3 | ledger within 64 MiB at ready: V4 small, Qwen local, MiMo, GLM Flash |
| 8 | `S1a` V4.1 serve loop on shared parts at share 0 (`PrefillQueue`, lanes, shared draft binding, `ScoreRows`, admission hooks) | S | L | 2, 4 | standard V4.1 stage gates + 3-session C16 recheck at share 0 |
| 9 | `S2` V4.1 prefix cache onto the engine (restore plan, snapshot meta, pending capture) | S | L | 2 | byte-exact exact-prefix text/image/odd frontier; approximate replay NLL; host tier; agentic hit tokens |
| 10 | `S3` `GraphBank` (`FixedStartup`), GLM Flash `GraphCache` as `Budgeted`, V4.1 banks onto it | S | M | 7 | `CaptureWatch` = 0 steady-state captures over C16 |
| 11 | `P4` `shared/experts/rtx` (`NativeTp2` from V4.1 `RankWave`, `Combine`, `RouteIdentity`), dsv4 `rtx_tp2` CMake, V4 wiring | P | L | 2, 3 | V4 Pro EXL3 K2 KL no worse than rc1; route check; A/B min/max; V4.1 byte-exact through `NativeTp2` |
| 12 | `D3` global row budget and copy competition on GLM Flash | D | M | 6 | same card; C16 heterogeneous prompts is the metric; C1 unchanged |
| 13 | `S1b` decode share on V4.1 (`--prefill-chunk-s`, `--decode-share`) | S | M | 8 | stream KL <= 2x batch envelope; long-prompt greedy-16 exact; decode gaps |
| 14 | `P5` V4 drafter as a `Movable` on GPU1, terminal close, marks, `ShardedVocabulary` | P | M | 11 | lossless spec; C1 at max |
| 15 | `P6` GLM Flash solver port, coordinator tp2 packages, `Fp8MoeTp2`/`Exl3Tp2`, router replica, DFlash2 on GPU1 | P | M | 11 | golden NLL 2.4054; A/B min/max at 2M |
| 16 | `D4` MiMo, GLM 5.3, Qwen (chain depth as the pre-draft action), V4 (fixed-width binding) onto the shared policy | D | M, S, M, S | 6 | the D2 card per family on its min/max |
| 17 | `P8` MiMo solver port, bidirectional MoE exchange, `Fp8MoeTp2`, drop the unused wire buffer | P | M | 11 | golden; A/B min/max (max sheds 12 Spark layers) |
| 18 | `P7` GLM Flash per-layer executor: Whole DSA layers, compact index on owners, Spark-free 2 RTX at 2M, tr3 at 1M | P | L | 5, 15 | fidelity; Spark-free card; A/B min/max |
| 19 | `P11` exact Spark admission for every family (`ModelAdmission`, endpoints, ingress, finite `RingBudget`) | P | M | 7 | every family's min/max: ring peak == charge |
| 20 | `S4` (= `P12`) V4.1 experts, admission and handoff onto shared: coordinator exchange on `SparkLink`, local/TP2 on `shared/experts/{local,rtx}` with `OwnerReduce`, `FamilyPlacement`, `PlacementHandoff`; delete `memory*.rs`, `placement.rs`, `v41_experts/tp2.rs` staging | S/P | L | 8, 11 | `planner_equals_runtime_deepseek_v41`; KV pool at 2M on min/max never smaller; MXFP4/NVFP4/EXL3 launches; V4.1 golden byte-exact |
| 21 | `S4c` fold V4.1 single-RTX and distributed paths onto the per-layer ownership map (`P3` hop) | S | L | 18, 20 | standard V4.1 gates; C1 single-lane fast path survives |
| 22 | `P9` GLM 5.3 solver port, ranges vs head split at max; `P10` Qwen solver port, ranges on 2 RTX, FP8 KV option | P | M, M | 18 | A/B both modes at max (GLM 5.3); golden + 1 RTX local and 2 RTX ranges A/B (Qwen) |
| 23 | `S5` launcher: `cuteafd plan --deployment` -> `DeploymentSpec`, `run-family.sh` serves V4.1, `run.sh` dispatches only | S | M | 19, 20 | `--dry-run` equality with `run.sh` for release configs; script tests; one launch per arm |
| 24 | `D5` delete `CycleCost`, `Calibration`, `allocate`, v2 buckets, `glm5/dflash_policy.rs` planning, per-family copy loops | D | S | 16 | tests; failing ids unchanged |
| 25 | `S6` delete and rename: legacy non-topology path, `v41_` module prefix, dated compatibility readers | S | M | 21, 23 | full V4.1 parity (3 interleaved sessions) against v2.0.0 |

**Release cuts.** Full V4.1 parity and the agentic bench run at cuts, not per
step: a cut after step 11 (placement for V4, draft policy for GLM Flash,
V4.1 byte-exact on shared types), one after step 20, and v3.0.0 after 25.

**v3 reference cards (TJ, 2026-10-10).** Besides each family's min/max, these
cards are in the regular test set for the unification steps and are v3
release cards:
- **GLM 5.3 Flash EXL3 K3.25, 2× RTX, no Sparks.** Every routed expert is on
  the RTX cards. This is the far edge case for the adaptive draft policy (no
  Spark traffic to price) and for TP2 placement (`P6`, `P7`), and the
  Spark-free card from v3 item 4.
- **GLM 5.3 Flash, no experts onboarded.** Every routed expert is on the
  Sparks. The opposite edge case for the draft policy's resource classes.
- **GLM 5.3 Flash, about half the expert layers onboarded,** bought with a
  larger KV pool and concurrency limit. The mixed case where the solver
  trades experts against KV, and the draft policy prices both resource
  classes at once.
- **V4.1 Flash, 1× RTX + 4 Sparks, and 2× RTX + 6 Sparks.** The unification
  reference for every ds41rt stage (`S0`–`S6`) and for `P12`, beside the
  shipped min (1× RTX + 3) and max (2× RTX + 4) cards.

The draft-policy steps (`D2`, `D3`, `D4`) gate on the GLM Flash edge cards
as well as min/max.
- GLM Flash today runs all-local or all-Spark experts per process
  (`--peers` conflicts with `--local-experts`).
  - The **all-Spark** card: min/max as shipped.
  - The all-Spark card runs now and gates `D2`. There is no one-RTX
    all-local card: the K3.25 routed experts are 115.6 GiB (K3 106.7 GiB),
    over one 95.6 GiB RTX, so all-local GLM Flash is the 2× RTX no-Spark
    card below.
- The **mixed, about half onboarded** card needs a per-layer expert backend
  (local fp8moe/EXL3 layers plus Spark layers in one process). That is
  `P6`/`P7` placement work, and it gates `D3`/`D4` once it lands.
- The **2× RTX no-Spark** card needs TP2 RTX experts (`P6`). It joins the
  set then. The placement steps (`P1`–`P12`) and ds41rt stages
gate on the cards their family touches.

**Dropped or flagged (review 2026-10-09):**
- Dropped: the `Speculator` lifecycle trait and `VerifyCost` (one
  implementor each under decision 2); `P13` as its own PR; `CycleCost`
  concurrency buckets as a gated step (interim only, see the draft design).
- Under-specified, to fix in the step that owns it: hop buffers
  (`[4096,4,H]` BF16 = 128 MiB per lane per boundary) are not in the
  placement tables and must be `fixed` demands in `P3`; `GraphSet`'s
  bytes-per-executable is a per-arch measured constant that `P2` must record
  with its driver version; `S4c` is unaudited (~1.5-3K lines) and sized from
  file counts; `group_rows` for fp8moe/EXL3/NVFP4 kernels is assumed 16 until
  the AOT manifests record the tile (`D2`); `S2`'s distributed `prefill_hold`
  is either implemented or logged, never claimed.
- Questions for TJ, not changed: decision 7 (keep V4.1's 20/20 split) is
  consistent with `P7`'s per-layer mix being measured later; decision 2 (no
  unified loop in v3) is what makes dropping `Speculator` safe, and it means
  three family loops carry the shared draft binding by hand (one `D1` helper,
  three call sites).

**What this changes in the sections below** (edited in place, marked
"revised 2026-10-09"): ds41rt map item 3 and stage 1a drop `Speculator`/
`VerifyCost`; stage 3 consumes `P2`'s `GraphSet`; stage 4a+4b is `P12`; stage
4c follows `P7`; placement `P13` is folded; the placement design's head shard
is the ds41rt design's `ShardedVocabulary`.

## v3: retiring ds41rt (design, 2026-10-09)

Design for v3 item 2. Inputs: four read-only code audits
(`builds/v3-ds41rt-design/final-{A-serve,B-prefix,C-experts,D-exec}.md`, with
file-by-file classifications, line counts and interface sketches), the
work/v41-decode-share closeout, and the parallel placement design (item 1).
Paths are under `rust/crates/cuteafd-daemon/src/` unless stated; line counts
are `wc -l` including inline tests.

**Corrections to the outline above, from the code:**
- "HC-lagged replay" is two separate things, and neither is a generic lagged
  state.
  - **mHC lag.** Each sublayer collapses its residual with the *incoming*
    `pre` and produces the next sublayer's `pre` (`v41_hc.rs`, "the newly
    generated pre belongs to the NEXT sublayer"). This is model arithmetic
    and stays in the family.
  - **CED replay.** Encoder layers 0-19 prefill the whole prompt
    (`CacheStage::Encoder`, windows 0..20). Then decoder layers 20-39 replay
    only the last `min(end, 128)` rows from the retained layer-19
    residual/pre (`EncoderSuffix`, 40,976 B/row, `begin_decoder_replay`).
    Replay happens once per prompt, not once per encoder wave.
  - **What is generic.** Only the prefix-cache consequence is generic: a
    partial hit reuses compressed sources through the even-aligned common
    prefix (`source_end`), but rebuilds windows from `source_end - 128`
    (`replay_start`). MiMo's partial SWA replay has the same two-frontier
    shape.
- V4.1's two-RTX layout is a **20/20 layer-range split**: layers 0-19 own
  GPU0 and layers 20-39 own GPU1 (`v41_backbone_cache/placement.rs:16`).
  - It is not a head split (the head split measured -5..-8%).
  - Its TP2 experts broadcast canonical routes to the peer and reduce
    routed and shared outputs separately, rank 0 then rank 1, then add in
    BF16 (`v41_experts/tp2_ffn.rs:332,364`). They do not join any
    all-reduce.
  - In item 1's per-layer ownership terms this is `Whole(0)` x 20 +
    `Whole(1)` x 20, with TP2 experts combining by their own peer reduce.
- There is no shared serve loop and no shared speculator trait to move onto.
  GLM Flash and MiMo each run their own scheduler over shared parts
  (`PrefillQueue`, sampler, token I/O, draft policy, console, prefix
  engine). "Ordinary family" here means the same: a family scheduler over
  shared parts, no private engine machinery. `round_groups` does not exist;
  the queue has `round` and `round_pairs`.
- **Shared code already imports V4.1:**
  - `CudaCopyEngine`: `shared/prefix.rs` and every family serve loop;
  - `BatchScores`/`VOCAB`: `shared/constraints.rs:1`;
  - `ExpertLayer`/`ExpertWeights`/`HostExpertExchange` and the EXL3 types:
    `shared/experts/service*.rs`, `deepseek_v4/local.rs`, GLM Flash's
    streamed EXL3.

  Retiring ds41rt starts by inverting these dependencies.

### Map: V4.1 subsystems against the shared layer

Class: **same** (a shared counterpart exists; V4.1's copy is a vestige),
**extend** (generic, but `shared/` lacks it: grow `shared/`, then V4.1 uses
it), **model** (stays in the family).

| V4.1 subsystem | Files | Lines (tests) | Shared counterpart | Class |
|---|---|---:|---|---|
| Serve loop, admission, independent lanes, sampling plan, scores, console, copy drafts, scoring | `v41_native_serve.rs`, `v41_native_serve/{scheduler*,speculative*,copy_drafts,scores,console,distributed,prefill_target,placement}.rs` | 7,938 (2,860) | `shared/prefill_share` (`PrefillQueue`), `shared/sampler` (V4.1's head already re-exports it), `shared/token_io`, `shared/console` (V4.1's console is already a feed into it), `shared/draft_policy`, `shared/probe`, engine `DeferredAdmission`, engine `MediaAdmission` | extend: resumable prefill, lanes, speculator, retained scores. same: console/sampler/media glue. model: CED prefill phases |
| Prefix cache + host tier | `v41_native_serve/prefix*.rs` | 1,809 (803) | `cuteafd-engine::prefix` (`PrefixCache`, `PrefixFamily`, `RefPagePool`, `MarkStore`, `ReuseRule::V41`), `cuteafd-hostcache`, engine `MediaKeys`, `shared/prefix` | same: radix, banks, image keys, host orchestration. extend: restore plan, snapshot metadata, pending capture |
| Memory / admission | `v41_native_serve/memory*.rs` | 1,159 (767) | item 1 `cuteafd_loader::placement::solve`, `shared/placement.rs`, `shared/memory_report` | same: replaced by the solver. model: source-pool cost provider |
| Request leases, Engram history, CED cache phases | `v41_requests*`, `v41_backbone_cache*` | 4,326 (1,942) | engine `RefPagePool` generations (pages only) | model, plus extend: atomic cache+Engram reservation |
| Compressed KV (CSA2 sources + index keys) | `v41_compressor*` | 3,546 (958) | `RefPagePool` (basic refcounts in `source_cache/ownership.rs` duplicate it) | model; the ownership basics are a vestige |
| FP8 window rings, dSpark rings | `v41_window*`, `v41_dspark_cache*` | 2,466 (555) | `MarkArena` (positional marks) | model; captured as marks |
| Expert path: layers, execution, exchange, EXL3, local, TP2, NVFP4, assignment | `v41_experts.rs`, `v41_experts/{coordinator,execution*,exl3*,local*,nvfp4,paired,tp2*}.rs` | 9,091 (2,900) | `shared/experts/service*` (already uses these types), `shared/spark_intake` (`SparkLink`), `shared/peer_split`, `shared/memory/device`; item 1 `shared/experts/rtx.rs` | extend: move out as shared types. same: coordinator receive/reduce. model: catalog/role/format adapters |
| dSpark drafter | `v41_experts/dspark*` | 5,685 (1,078) | none (GLM Flash/MiMo drafters are family code too) | model; its expert backend moves to shared executors |
| dSpark policy binding, copy drafts | `v41_native_serve/speculative/policy.rs`, `copy_drafts.rs` | 633 (240) | `cuteafd-core::dspark_policy` is already shared code; it becomes the shared policy (v3 item 6) | same: the binding becomes the shared `shared/draft` binding; the copy *search* moves to `shared/speculation/copy.rs` (revised 2026-10-09) |
| Shared FFN, router, projection TP2 | `v41_backbone_shared*`, `v41_shared_ffn`, `v41_backbone_router`, `v41_projection_tp2` | 2,135 (954) | `shared/peer_split` | model |
| Target pass, lanes, execution, block, layer graphs | `v41_target_pass*`, `v41_backbone_lane*`, `v41_backbone_execution*`, `v41_block*`, `v41_layer_graphs` | 10,598 (4,198) | `shared/memory/chain` (`StageChain`, already used), `shared/decode_graph` (bucket policy only, no graph owner) | model: layer program. extend: lane set, ordered two-lane pipeline, graph bank. vestige: single-vs-distributed duplication |
| Target head, embedding | `v41_target_head*`, `v41_target_embedding*` | 4,075 (2,573) | `shared/sampler`, `shared/token_io::TokenEmbedding` | model: mHC collapse. extend: sharded vocabulary. same: embedding |
| mHC | `v41_hc`, `v41_backbone_hc` | 425 (0) | none | model |
| Attention, indexer, sparse attention | `v41_attention_*`, `v41_index_*`, `v41_sparse_attention*` | 4,880 (1,166) | none | model |
| Engram tables and gates | `v41_engram*` (+ loader `engram_*`) | 981 (112) | loader `MappedTable`, `shared/mapped_table::MappedTableDevice` | model; extend: non-blocking row upload |
| Vision tower | `v41_vision*` | 1,107 (280) | `shared/vision` (remote path already shared) | model; extend: local encoder job adapter (BF16 patches, merge 3) |
| Tensors / vocab shard | `v41_tensors*` | 344 (57) | loader catalog | model |
| **Family total** | 141 files | **60,565 (~21,200)** | | |

Outside the daemon family: native `native/families/deepseek_v41/` (42 files,
4,234 lines, all kernels: model); loader `families/deepseek_v41/` (~3.3K Rust);
FFI bindings (~3.4K); `run.sh` V4.1 branch (lines 144-832, 689 lines);
`NativeServeArgs` (`cli.rs:743-889`).

### What `shared/` must grow

Every item is opt-in for the families that don't use it. Signatures are
sketches; typed errors inside crates.

1. **Resumable, time-sized prefill** (`shared/prefill_share.rs`). This
   generalises MiMo's `--prefill-chunk-s`.

   ```rust
   pub struct ChunkBudget { pub target: Duration, pub row_cap: usize }
   pub struct PrefillUnit<U> { pub work: U, pub rows: usize, pub estimate: Duration, pub finalizes: bool }
   pub trait PrefillDriver {
       type Cursor;            // family prefill state; survives decode steps between units
       type Unit;              // a legal shape for the family
       fn plan(&self, cursor: &Self::Cursor, budget: ChunkBudget) -> Result<PrefillUnit<Self::Unit>>;
       fn execute(&mut self, cursor: &mut Self::Cursor, unit: &PrefillUnit<Self::Unit>) -> Result<Chunk>; // returns drained, committed
       fn observe(&mut self, unit: &PrefillUnit<Self::Unit>, took: Duration);
       fn abort_and_drain(&mut self, cursor: &mut Self::Cursor) -> Result<()>;
   }
   ```

   - **Shared arguments.** `prefill_chunk_s: Option<f64>` moves into
     `DecodeShareArgs`. MiMo's `timed_chunk_rows`/`split_timed_chunk`
     become its `plan`.
   - **Queue changes.** Add `one_wave_rounds()` and `decode_seconds()` from
     work/v41-decode-share (cb2aea55).
   - **V4.1 driver.** `V41Prefill` is a cursor ported from that branch's
     `PrefillProgress`: Start / Encoder{suffix, next} / Replay{suffix} /
     Continuation. Its units are:
     - an encoder group: one or two ordered waves, keeping the two-lane
       overlap;
     - the final replay, indivisible and ≤128 rows;
     - a cached continuation.
   - **Planning.** `plan` estimates critical-path time, not the sum of the
     lane times. It keeps legal capacities (`prefill_capacity`: 80, 256,
     1024, 4096).
   - **Share 0 adds no per-round work.** Pick the serve-loop variant once
     at startup (`serve::<F, const SHARED_PREFILL: bool>`). Share 0 calls
     today's whole-prompt path directly.
2. **Lanes and ordered two-lane pipelining.**
   - `shared/serve/lanes.rs`: `LaneSet<L>` / `LaneLease`, with
     independent mutable lanes over borrowed immutable weights. Lanes are
     polled with `join!` (never cancel a peer future that owns queued
     work). `LaneCommit` covers prepare, then target+draft commit, poll
     both, publish, or `abort_and_drain`.
   - `shared/prefill_pipeline.rs`: `PipelineOrder` / `ChunkPermit`
     (`reserve`, `wait_predecessor(stage)`, `publish(stage)`, `commit`,
     `cancel`). This is V4.1's `encoder_stream.rs` dependency order with the
     family's stage hooks. The C1 `single_lane_round` fast path stays.
3. **Draft policy and copy drafts** (revised 2026-10-09; design: "v3 draft
   policy: resource-priced, shared").
   - The shared policy *is* V4.1's: `cuteafd-core::dspark_policy` becomes
     `cuteafd-core::draft_policy` with the geometry (layers, experts, top-k,
     group rows, slice bytes, resource class per layer) as an input, and
     V4.1's `speculative/policy.rs` is its first binding (roadmap step 1,
     decision-identical). Nothing in this stage designs a `VerifyCost` trait
     or a second cost model.
   - **No `Speculator` lifecycle trait in v3.** With decision 2 (family
     loops over shared parts, no unified loop) it would have one
     implementor. `DraftRuntime`/`DraftChain`, the three windows, taps, RNG
     and widths 5/7 stay family code. What is shared is the policy's
     observation contract: proposals and observed rows carry `DraftSource`
     (neural / copy), so copied rows never train neural evidence (the
     binding already does this). Converge drafter lifecycles after v3 with
     the loops.
   - `copy_drafts.rs`'s indexed 8-gram match becomes the shared
     `shared/speculation/copy.rs` helper (MiMo's longest-backward match is
     its second policy); whether a copy span is *used* is the shared policy's
     acceptance-gated decision (decision 9), not a per-family override.
4. **Retained scores** (`shared/token_io.rs`).
   - `ScoreRows { selected, packed, row_to_pack, vocab }` and
     `RetainedScores { vocab, raw }` replace V4.1's `BatchScores`/
     `TokenScores` (fixed 129,280 vocab). `constraints.rs` stops importing
     the family.
   - Draws carry an explicit `draw_position`: V4.1 draws at `generated +
     row`, while the `SelectBatch` callers draw at `start + 1`.
5. **Prefix engine** (`cuteafd-engine::prefix`).
   - **Restore plan.**

     ```rust
     pub struct LaggedState { pub source_end: usize, pub replay_start: usize }
     pub enum RestoreFidelity { Exact, ApproximateReplay }
     pub struct RestorePlan { pub snapshot_end: usize, pub target_end: usize, pub lag: LaggedState, pub fidelity: RestoreFidelity }
     ```

     - `PrefixFamily::plan_restore(&self, RestoreCandidate) ->
       RestorePlan`.
     - `restore(mark, placement, &RestorePlan, &RestoreContext { native_tokens, media })`.
     - Pages fork through `source_end`, not through `replay_start`. Today's
       single `len` conflates them.
     - Exact hits stay exact. An approximate hit never replaces an exact
       frontier.
     - MiMo partial reuse moves to the same plan.
   - **Snapshot metadata.**
     - Add `type SnapshotMeta: Clone`, with `capture_meta`/`restore_meta`
       and `HostPayload<M> { after, media, family: M }`.
     - V4.1 keeps window/carry descriptors and Engram history there.
       Partial restores rebuild Engram history from native token ids.
     - dSpark rings go into V4.1's combined mark (MiMo precedent). No new
       engine draft lifecycle; `has_draft: false` is never claimed with
       draft bytes.
   - **Pending capture.**
     - `queue_capture`/`poll_capture`/`abort_capture`, with a typed
       `CaptureTicket` from the family. Synchronous families complete
       immediately.
     - The pending owner keeps pages, marks and source owners until both the
       target and drafter copies land. On failure, quarantine; never free.
   - **Other.** `PrefixCache::prefill_hold()` passes through to the host
     tier. Lazy-tail COW, work-aware eviction and multi-class pages are
     deferred (open question 4).
6. **Graph bank** (`shared/decode_graph.rs`).
   - `GraphBank<K>` over a `GraphOwner` that pins device, weights,
     buffers, geometry and workspace. It offers `warm`, `enqueue` (missing:
     a typed eager decision), `retire` and `drain_retired`.
   - `GraphPolicy::FixedStartup` is V4.1's default: shapes are warmed at
     readiness, anything else runs eagerly, never captured.
     `GraphPolicy::Budgeted{bytes}` is GLM Flash's `GraphCache`, moved here.
   - `CaptureWatch` replaces `graph_capture_watch` and covers every verify
     path; the sampled-terminal return skips it today.
   - Exact keys first. V4.1 on `ROW_BUCKETS` waits for a sentinel and
     route-crossover audit: its `u64` positions and Engram/index/tap masks
     don't take `MaskedRow`'s -1 sentinels as is.
7. **Expert types out of the family** (`shared/experts/{layer,execution,exchange,exl3,local,assignment}.rs`).
   - **Types.**
     - `ExpertLayer { Backbone { layer, shard }, Draft { stage, shard } }`
       with `ExpertShard { rank, world, intermediate }`.
     - `ExpertLoadBudget`, `ExpertExecution`, `HostExpertExchange`,
       `Exl3Weights`/`Exl3Execution`/`Exl3Worker`.
     - `ExpertOutputLayout { Fp32Routes, Fp32Tokens, Bf16Routes, Bf16Tokens }`.
     - `RouteAssignment` (replicated / paired).
   - **What stays in the family.** Catalog, role ids, the 44-slot native
     binding and NVFP4 scales.
   - **Item 1's TP2 layer.** `shared/experts/rtx.rs::RtxExpertLayer` takes
     `BackboneTp2` as its first implementation. It needs a combine mode:
     `Tp2Combine::{HeadSplitAllReduce, PeerReduce}`. V4.1 keeps
     `PeerReduce` with today's rounding order (routed reduce, shared
     reduce, BF16 add) until a fused path passes fidelity.
   - **Spark side.** The coordinator's Spark receive/reduce moves onto
     `SparkLink`/`SparkIntake`.
8. **Admission through item 1's solver.**
   - V4.1 implements `cuteafd_loader::placement::FamilyPlacement` in
     `placement/families/deepseek_v41.rs`, ported from
     `plan/layout/v41.rs` and `v41_native_serve/memory.rs`.
   - **KV cost is a per-rank unit cost, not a per-token scalar.** A
     512-token unit is 5 source pages of 91,136 B (~890 B/token). On top
     come FP8 window marks (2.72 MB per retained sequence), source
     replicas and COW tails.
   - **The dual-RTX minimum routed prefix becomes an explicit
     `ExpertDemand` floor.** Today it is 20 layers, 1 with an explicit
     topology.
   - **Engram as a family-owned resource.** It needs no shared trait: the
     family keeps its typed owner and declares device rows plus pinned
     staging as `fixed` demands.
   - **Handoff.** `v41_native_serve/placement.rs` (nonce, plan, ack,
     ready) becomes `shared/placement/handoff.rs::PlacementHandoff<P>`.
   - **Test.** `planner_equals_runtime_deepseek_v41`.
9. **Smaller extensions.**
   - `shared/vocabulary.rs::ShardedVocabulary`: the two-GPU head with
     checked argmax and global offsets.
   - `shared/mapped_table`: `PendingRows::{poll,cancel}` and
     `MappedTableDevice::{enqueue_upload,reclaim}` for Engram.
   - `shared/vision::EncoderJobAdapter`: BF16 patch input and merge 3.
10. **Launcher.**
    - `cuteafd plan --deployment` emits a `DeploymentSpec` JSON that
      `run-family.sh` consumes. It covers:
      - role/manifest proof (`V41_EXPERT_TP_AOT.json`, `symbols_verified`,
        native-library hash, the `io.cuteafd.spark_tp_roles` label);
      - artifact identity and the resolved-config fingerprint;
      - the topology (`SPARK_TP`/`SPARK_EP`) and the placement handoff;
      - `KV_POOL_SIZE` and `MEMORY_RESERVATION` as `AdmissionArgs`.
    - **CLI.** V4.1's arguments are composed from shared clap groups
      (`RuntimeServeArgs`, `AdmissionArgs`, `TopologyArgs`, `PrefixArgs`,
      `ConsoleArgs`, `ApiArgs`) plus a family `V41ModelArgs` (TP2_*,
      dSpark, paired EXL3).

### Staged migration

Each stage is its own branch off work/p0, merges on its own gates, and
deletes the V4.1 code it replaces. **Order:**
- Stage 0 first (with the draft policy's `D0`/`D1`, which stage 1a consumes).
- Stages 1, 2 and 3 touch disjoint files and can run in parallel.
  Hardware gates are serialized.
- Stage 4 (= placement PR 12) follows item 1's solver landing for V4 (P1,
  P4) and stage 1a; 4c follows P7.
- Stage 5 needs stage 4's `DeploymentSpec` inputs and P11's Spark admission.
- Stage 6 is last.
The cross-design order with sizes and gates is the "v3 roadmap" above.

**Gates for every stage (V4.1 MXFP4, plus NVFP4 where the stage touches
experts or prefill):**
1. **CPU tests.** Cargo/script tests, reported as failing ids.
2. **Fidelity.**
   - Quick fidelity tier against the official reference: KL and top-1
     within the cold-run noise envelope of work/p0 measured the same day;
     tools 30/30.
   - "Exact cache" is byte-exact: decode after an exact hit equals cold
     decode for the same restored state.
   - "Lossless spec" holds.
3. **Quick A/B at the 2M pool on min (1 RTX + 4) and max (2 RTX + 4).**
   - The candidate WIP is measured against work/p0 the same day, and also
     against the v2.0.0 release images, so stage slack cannot accumulate.
   - It covers C1 and C16 code decode, 8K prefill, readiness time and
     memory headroom.

**C1 bar (V4.1 is the speed reference):**
- **Per stage.** C1 must be ≥0.98 against both baselines on both layouts.
  - Between 0.95 and 0.98: 3 interleaved sessions, pass if the median is
    ≥0.99.
  - Below 0.95: fail.
- **C16 and 8K prefill.** ≥0.97 (decode-share measured ±5% C16 session
  noise); the same 3-session rule applies at 0.97.
- **Readiness.** At most +5%.
- **Stage 6.** It runs the full V4.1 parity gate (3 interleaved sessions)
  against v2.0.0. The bar is C1 ≥0.99 on min and max, C16 and prefill
  ≥0.98, and no fidelity regression. A miss blocks the release cut, not
  earlier merges.

**Stage 0: invert dependencies (M).** Moves only, no behaviour change.
- `CudaCopyEngine` + `Regions` and their tests move to
  `shared/prefix/cuda_copy.rs`.
- Expert layer, execution, exchange and EXL3 types move to
  `shared/experts/` (map item 7).
- `ScoreRows`/`RetainedScores` move to `shared/token_io`.
- `deepseek_v41` gets a golden entry in `commands/family.rs` (it has
  `None` today), built on the existing scoring probe.
- **Gate.** Tests; byte-exact golden for V4.1, V4 and GLM Flash streamed
  EXL3; one quick A/B pair on V4.1 min (expert exchange is a shared hot
  path).
- **Leaves the family:** ~3.5-4.5K (relocated).

**Stage 1a: serve loop on shared parts, share 0 (L).** V4.1's scheduler
keeps its admission and CED phases and adopts:
- `PrefillQueue` + `V41Prefill`;
- `LaneSet`/`LaneCommit`;
- `PipelineOrder`;
- the shared draft binding (`shared/draft/*`, roadmap `D1`): `LayerClock`,
  `RoundRoutes`, `RoundClock` and the `Evidence`/`DraftSource` observation
  builder replace `speculative/policy.rs`'s hand-built observation; the
  dSpark drafter itself stays family code (revised 2026-10-09, no
  `Speculator` trait);
- `ScoreRows`;
- engine `DeferredAdmission` and `MediaAdmission`, including the lifetime
  budget fit as a policy hook;
- the console `Ticket` with an id and a lane.

The default stays share 0 through the static bypass.
- **Gate.** The standard gates plus an explicit C16 3-session recheck at
  share 0 (decode-share saw a -5.3% median it could not explain).
- **Leaves the family:** ~3.5-5K. Net deletion ~1-1.5K.

**Stage 1b: decode share on V4.1 (M).** Time-sized units via
`--prefill-chunk-s` and `--decode-share 0.2`.
- **Gate.**
  - The fidelity envelope: stream KL ≤2x the batch-mix envelope.
    Decode-share passed at 0.2 and failed at 0.4.
  - Long-prompt greedy-16 exact.
  - Decode gaps measured with
    `scripts/bench/deepseek_v41/bench-prefill-interference.py`, ported from
    the closed branch.
- **Default.** Set by TJ from the measured gap/TTFT trade (open question 3).

**Stage 2: prefix cache (L).**
- The engine grows restore plans, snapshot metadata, pending capture and
  `prefill_hold` (map item 5).
- `V41Prefix: PrefixFamily` bundles the five source pages into one 512-token
  unit. Window/carry/dSpark rings form an arena mark; Engram history goes in
  `SnapshotMeta`.
- **Deletes:**
  - `v41_native_serve/prefix.rs` and `prefix/images.rs` (engine
    `MediaKeys` replaces them);
  - the host-cache binding;
  - `source_cache/ownership.rs` basics;
  - the duplicate `HostBudget`.
- **Gate.** The standard gates, plus:
  - byte-exact exact-prefix for text, image and odd-frontier prompts;
  - approximate-replay NLL no worse than today;
  - host-tier restore;
  - 1-2 short agentic sessions with hit tokens ≥ today's.
- **Leaves the family:** ~1.5-2.3K. Net ~1-1.6K.

**Stage 3: graphs (M).**
- `GraphBank` with `FixedStartup` replaces `v41_layer_graphs.rs`
  (`LayerGraphs`/`RowGraphs`) and the banks in the head, query/output,
  index and embedding.
- GLM Flash's `GraphCache` moves in as `Budgeted`.
- The warm list is placement `P2`'s `GraphSet::startup(shape)` (revised
  2026-10-09): one definition of the startup shapes and their bytes, read by
  the planner for admission and by the bank for warm-up, so admission cannot
  under-count GPU1's new router/TP2 executables.
- **Gate.** The standard gates plus `CaptureWatch` = 0 steady-state
  captures over the C16 battery.
- **Leaves the family:** ~0.6-1.1K. Net ~0.2-0.4K.

**Stage 4: experts, admission, layer ownership (L; two PRs; revised
2026-10-09).**
- **4a+4b is placement PR 12**, one branch and one gate: coordinator Spark
  exchange onto `SparkLink`; local/TP2 onto `shared/experts/{local,rtx}`
  with `Combine::OwnerReduce`; the dSpark expert backend onto shared
  executors; admission via `FamilyPlacement` + `RuntimeInventory`;
  `v41_native_serve/placement.rs` becomes `PlacementHandoff`. Deletes
  `memory.rs`, `memory/distributed.rs`, `placement.rs` and the
  `RankWeights`/`RankWave` staging in `v41_experts/tp2.rs` once `NativeTp2`
  serves V4.1 byte-exact (P4 proves that first on V4). Splitting experts from
  admission bought nothing: both change the same startup path and both need
  the same V4.1 launch matrix.
- **4c.** Fold the single-RTX and distributed paths into one path driven
  by the per-layer ownership map, where 1 RTX means every layer is
  `Whole(0)`. Today `v41_target_pass.rs` vs
  `v41_target_pass/distributed.rs`, `v41_backbone_execution.rs` vs
  `v41_backbone_execution/distributed.rs`, and so on duplicate
  orchestration. C1's single-lane fast path must survive. It follows
  placement `P7` (the first generic per-layer executor, on GLM Flash) and
  uses `P3`'s hop primitive, which is `BlockTransfer` moved; V4.1 does not
  grow a second one.
- **Gate.** The standard gates, plus:
  - `planner_equals_runtime_deepseek_v41`;
  - KV pool at 2M on min and max, never smaller than today;
  - MXFP4, NVFP4 and EXL3 (compact TP4) launches.
- **Leaves the family:** ~5-8K. Net ~2-4K (4c is unaudited, ~1.5-3K of
  that).

**Stage 5: launcher (M).**
- V4.1 launches through `run-family.sh` from the `DeploymentSpec`.
- `run.sh` becomes the dispatcher only.
- `serve-native` becomes a hidden alias of `serve --family deepseek_v41`
  for one release.
- Old keys warn for one release through `key()`. `KV_POOL_TOKENS` aliases
  `POOL_TOKENS`, `DSPARK` maps to `SPECULATOR=dspark`, and
  `SPARK_REDUCTION_MIN_ROWS`, `COORDINATOR_GPU_HEADROOM_GIB`,
  `SPARKINFER_EXL3` and `MODEL_VARIANT` are dropped (accepted but never
  forwarded today).
- **Gate.** `--dry-run` resolves the same fingerprint, roles and topology
  as `run.sh` for the release configs. Script tests. One launch per arm on
  min and max.
- **Net:** ~250-400 lines of launcher and ~50-120 of CLI. The 689-line
  branch is the gross figure.

**Stage 6: delete and rename (M).**
- **Remaining vestiges go:** the legacy non-topology path if TJ agrees
  (open question 5) and leftover duplicate glue.
- **Module names drop the `v41_` prefix inside the family**, as GLM
  Flash's do.
- **`v41` stays as the model's tag** in C symbols (`cuteafd_v41_*`, 207
  identifiers), AOT prefixes and package directories, as `glmf_`/`mimo_`
  do.
- **No ds41/ds41rt names remain** in Rust, scripts, configs or docs,
  except the compatibility readers below. Each is dated and dropped at
  v4:
  - the checkpoint `meta.ds41rt`/`ds41rt_*` keys;
  - the `ds41_json_schema`/`ds41_tool_schema` grammar envelope: rename
    emitter and reader together and accept the old name;
  - the `DS41RF01` debug-frame magic: version it;
  - the `io.cuteafd.v41.spark_tp_roles` fallback read.
- **Gate.** Full V4.1 parity (above).

### What is left, and how much goes

| | Lines |
|---|---:|
| Family today (incl. ~21.2K tests) | 60,565 |
| Leaves the family, stages 0-6 | ~15-22K |
| of which net repository deletion | ~5-10K |
| Family at the end (incl. ~12-14K tests) | **~38-45K** |

- **The family holds:**
  - attention, indexer and sparse attention (3.7K production);
  - CSA2 compressed KV and windows (~3.7K);
  - CED cache phases (~1.3K);
  - the dSpark drafter model (~4.3K);
  - router, shared FFN and projection TP2 (1.2K);
  - mHC, Engram, vision and tensors (~2.2K);
  - the layer program (~3.5K) and head collapse (~0.9K);
  - adapters: `PrefillDriver`, `Speculator`, `PrefixFamily`,
    `FamilyPlacement`, the expert catalog/format (~4K);
  - a GLM Flash-sized serve loop (~1.5K).
- **Production code is about 26-30K.** That is about 3x GLM Flash's
  9.1K, because the model has more distinct mechanisms (CED, CSA2, indexer,
  mHC, Engram, a 5.7K neural drafter), not because of engine code.
- **Goal state:** V4.1 is one of six families. Its directory holds no
  scheduler, prefix cache, graph bank, expert service, admission solver or
  launcher. It runs through `serve`/`golden`/`run-family.sh` like the
  others. Features land once in `shared/`. The "GLM Flash-sized" target
  above is not reachable without deleting model code (open question 1).

### Risks and mitigations

1. **Encoder waves of 400-700 ms and a ~250 ms replay.**
   - Wave-boundary sharing alone can't bring decode gaps to tens of ms.
   - **Mitigations:**
     - Size units by time through the 80/256/1024/4096 capacities.
       512-row waves are projected at ~120-171 ms and 256-row at ~60-85
       ms; both are projections, to measure.
     - Treat a smaller wave as a numerics change: chunk boundaries change
       row counts and kernel tiles, so fidelity-gate it.
     - Leave the replay indivisible, so the floor is ~250 ms. Splitting it
       needs mid-replay decoder-window state and its own fidelity gate;
       defer that.
     - Measure separately whether decode can keep one execution lane while
       a prefill unit holds the other (V4.1 already has independent lanes;
       SM contention will slow decode).
2. **Cold prefill is not bit-reproducible.**
   - Spark slices accumulate prefill with unordered FP32 atomics at
     capacities ≥256 (`atomic_min_capacity=256` in
     `python/tools/aot/export_b12x_v41_experts_aot.py`), so byte-exact A/B
     is impossible for any stage that touches prefill.
   - **Mitigations:**
     - Gate prefill stages on KL/top-1/greedy-16 against a cold-alone
       noise envelope measured the same day (decode-share: alone max KL
       6.97e-5).
     - Gate decode byte-exactly after an exact cache restore (small decode
       rows don't use atomics).
     - Use the ordered packages (risk 3) for strict proofs.
3. **The deterministic Spark reduction option.**
   - Export without `atomic_min_capacity` (ordered FP32 route planes). A
     private "ordered ABI2" package already built on work/v41-decode-share
     but never ran its proof.
   - **Mitigation and plan:**
     - In stage 1a, make it a selectable package variant
       (`SPARK_REDUCTION=ordered`).
     - Measure 8K prefill and C16 on min/max.
     - Use it for every byte-exact gate in stages 1-2.
     - Make it the default if 8K prefill costs ≤2% (open question 6).
4. **Unexplained share-0 C16 -5.3%** (decode-share, 3 pairs).
   - **Mitigations:** the static share-0 bypass (no queue, clocks or debt
     calls), and a mandatory 3-session C16 recheck at stage 1a.
5. **TP2 combine order.**
   - Fusing routed and shared partials into one exchange reorders BF16/FP32
     rounding.
   - **Mitigation:** V4.1 stays on `PeerReduce` with today's order. The
     fused path is opt-in behind fidelity, in coordination with item 1.
6. **Async snapshot lifetimes.**
   - Pending target/draft captures and host restores own device memory
     across scheduler yields.
   - **Mitigations:**
     - fault-injection tests: cancel during capture, partial enqueue, host
       timeout;
     - quarantine on terminal CUDA errors;
     - drain before release (AGENTS lifetime rules).
7. **Planner/runtime mismatch.**
   - Today the planner auto-picks V4.1 RTX layers from average layer bytes
     / 2, a 512 MiB margin and 256 MiB per rank (loader `plan/layout.rs`
     ~1008). Runtime uses per-layer budgets and AOT scratch.
   - **Mitigation:** item 1's equality test is the stage 4b gate; no
     estimate stays labelled Exact.
8. **Launch safety.**
   - Moving to `run-family.sh` could drop role/manifest proof or the
     fingerprint.
   - **Mitigation:** dry-run equality against `run.sh` for every release
     config before `run.sh`'s branch is deleted.
9. **Distributed host-cache pacing.**
   - Distributed `PrefillTarget` ignores the per-chunk `prefill_hold`
     (warns once).
   - **Mitigation:** stage 2 either implements it or keeps it explicit and
     logged; don't claim it is solved.
10. **Readiness.**
    - Moves must keep bounded parallel reads, owner-thread packing and
      released staging.
    - **Mitigation:** readiness at most +5% per stage.

### Decisions (TJ, 2026-10-09)

1. **End-state size.** Judge "ordinary" by structure: no engine machinery in
   the family directory. Model code stays.
2. **Serve loop.** Not unified in v3. V4.1 gets a family loop over the
   shared parts, like GLM Flash and MiMo; converge loops after v3.
3. **Decode/prefill sharing is a cross-model feature.** V4.1 joins the
   shared machinery in stage 1b like every family. Its default is decided
   by measurement there (on if within the stage gate), not held off by
   policy.
4. **Prefix pages.** Bundle the five source pages into one 512-token unit
   (455 KB, ~890 B/token; no padded unified pool). The only overhead is one
   eager tail copy per cached prefix (~455 KB). Add page classes with lazy
   COW tails only if the agentic bench shows lost hits or pool.
5. **Legacy non-topology Spark path.** Express TP4 as `SPARK_TP=4`, one
   quick A/B, delete the legacy path in stage 6 unless a compact layout
   still has users.
6. **Deterministic Spark prefill reduction.** Build now as a package option;
   default if 8K prefill costs <= 2%.
7. **Two-RTX attention.** Keep V4.1's 20/20 layer split as the default.
   Per-layer ownership is a general tool for every family (item 1). Never
   measured on any model: head-splitting only V4.1's compressed (CSA/HCA)
   layers while keeping sliding-window layers whole. V4.1's earlier head
   split lost (C1 187 -> 170) because each split projection ended in a
   host wait, so measure the per-layer mix once item 1's shared exchange
   primitives exist.
8. **Naming.** Keep `v41` as the model tag, drop the `v41_` module prefix,
   dated compatibility readers for the ds41rt strings until v4.
9. **Copy drafting.** Rejected as a default on V4.1 (C1 0.982, though +34%
   on a literal-table edit) and MiMo (C1 0.957; copied tokens accepted ~45%
   vs DFlash ~99.5%). Becomes shared machinery as an acceptance-gated copy
   inside the shared draft policy (copy only when its span beats the neural
   draft), with the per-drafter calibration work; opt-in until it wins.

## v3 placement: design (2026-10-09)

Design for v3 item 1 (and item 4), written before implementation. Nothing here
ran on hardware. Numbers are CPU `cuteafd plan --layout` runs on origin/work/p0
3af226db (debug build, sparknest snapshots, default flags, no PROGRAMS.json),
plus arithmetic over the planner's items. Scripts and JSON:
`builds/v3-placement-design/` (`sweep.sh`, `project.py`, `sweep/`). The code
audits behind the tables are Sol reports relayed in this design's branch
report; file:line citations are at 3af226db..2a758f74.

### What the audits and planner runs found
- **Head-split engines carry a replicated hidden state.** V4, GLM 5.3, GLM
  Flash and MiMo sum the two attention partials in the same order on both
  GPUs, so the post-attention normalized input matches bitwise. Each layer
  has one FFN exchange slot (`4*lane + 2*(L%2) + 1`, BF16 `[T,H]`). V4, GLM 5.3
  and GLM Flash already exchange rank 1's shared/dense half there: a rank-1
  routed half fits into `peer_front` (V4), `peer_attention`/`peer_segment`
  (GLM 5.3) or `w1.shared` before the push (GLM Flash) with no new hop.
  **MiMo MoE layers are one-way today** (rank 0 broadcasts the finished
  routed delta; MiMo has no shared expert), so TP2 adds the rank1 -> rank0
  direction on the same slot.
- **Routers are lead-only everywhere** (V4 gate + bias + hash `tid2eid`, GLM
  `ops[0]`, MiMo BF16 or hi/lo FP32). TP2 must replicate them on GPU1.
- **V4.1 is a layer-range engine.** It owns layers 0-19 / 20-39 per GPU,
  with one boundary hop of `[T,4,5120]` BF16 + `[T,4]` FP32 (40,976 B/row).
  Its TP2 experts broadcast inputs and routes (15,568 B/row), then reduce
  routed and shared outputs separately in rank order, then add in BF16 on
  the owner (`tp2_ffn.rs` ~258-364). Measured: its head-split modes lose
  (C1 187 -> 170).
- **Kernels.** The half-width math exists almost everywhere; packaging and
  executors are missing (table under "TP2 RTX experts").
- **Admission differs by family.**
  - KV pool targets: the runtime helpers target 2M even on 32 GB cards,
    while the planner targets 1M there.
  - Timing: GLM Flash's eager path measures free memory after allocating
    experts, drafter and workspaces, while MiMo preflights before allocating.
  - Planner == runtime equality is complete only on work/v4-placement for V4.
  - Every non-V4.1 Spark worker runs with `RingBudget::new(usize::MAX)` and a
    hard-coded two endpoints, though GLM 5.3 opens four by default and MiMo
    Pro three.

Today's planner at the max reference configs (GiB used / capacity):

| config | KV pool | GPU0 | GPU1 |
|---|---:|---|---|
| V4 Flash 2 RTX + 4 | 603,648 | 90.3 / 90.3 (20 layers) | 13.1 / 89.3 |
| V4 Pro EXL3 K2 2 RTX + 6 | 653,056 | 90.7 / 90.7 (7 layers) | 24.5 / 89.3 |
| GLM 5.3 EXL3 K4 2 RTX + 6 | 1,296,768 | 93.5 / 93.5 | 82.4 / 93.5 (65.1 replicated MLA KV) |
| MiMo Pro 2 RTX + 6 | 2,097,152 | 49.9 / 93.5 | 42.4 / 93.5 |
| GLM Flash K3.25 2 RTX + 4 | 2,097,152 | 44.0 / 93.5 | 38.3 / 93.5 |

**Pool first, then TP2 experts, both at a 2M pool:**

| model | RTX expert layers |
|---|---|
| V4 Flash | 36 (today 20, with a 603K pool) |
| V4 Pro | 11 (today 7, with a 653K pool) |
| MiMo Pro | 12 (today 0; its Spark-bound prefill sheds 12 of its 69 MoE layers from the Sparks) |

These are arithmetic: free bytes over half-layer bytes, at 1.594 / 2.964 /
3.59 GiB per half-layer for Flash / Pro / MiMo Pro.

### 1. One admission solver

**Module.** The module is `rust/crates/cuteafd-loader/src/placement/`:
- `mod.rs`: the types;
- `solve.rs`;
- `pool.rs`;
- `families/<family>.rs`;
- `tests.rs`.

It is CPU-only, with no CUDA. It absorbs `serving_capacity/v4_placement.rs`
from b2f26af9 (`deepseek_v4_placement`, `deepseek_v4_expert_cost`,
`deepseek_v4_native_workspace`). The planner (`plan/layout.rs` `layout()`)
and every family's runtime admission call one function:

```rust
pub fn solve(request: &PlacementRequest) -> Result<Placement, PlacementError>;

pub struct PlacementRequest {
    pub inventory: Inventory,
    pub pool: PoolPolicy,
    pub layers: Vec<LayerDemand>,   // one per backbone layer, in order
    pub fixed: Vec<Demand>,         // non-layer items per device or per mode
    pub movables: Vec<Movable>,     // drafter, vision, audio
    pub spark: Option<SparkDemand>,
    pub policy: LayerPolicy,
}
pub struct Inventory { pub gpus: Vec<GpuBudget>, pub sparks: Vec<SparkBudget>, pub peer_access: bool }
pub struct GpuBudget { pub capacity_bytes: u64, pub headroom_bytes: u64, pub baseline: Baseline }
/// Planned: nothing allocated yet; every demand is charged, runtime context
/// included. Measured: one sample after CUDA context + program modules and
/// before any weight; `context_bytes` is what that sample shows in use, and
/// every named demand is still future (no double counting).
pub enum Baseline { Planned { context_bytes: u64 }, Measured { free_bytes: u64, context_bytes: u64 } }
pub struct PoolPolicy { pub requested: Option<u64>, pub target: u64, pub floor: u64, pub unit_rows: u64 }
pub struct LayerDemand {
    pub kind: AttentionKind,                 // Mla, Dsa, Csa, Gqa { kv_heads }, Swa { kv_heads }, Kda, Gdn
    pub colocate: Option<u16>,               // layers sharing index/source state (GLM 5.3 shared indexer, V4.1 source groups)
    pub weights: ModeBytes,                  // attention + norms + router + shared/dense FFN, per mode and rank
    pub kv: KvDemand,
    pub experts: Option<ExpertCost>,         // None: dense layer
    pub modes: Vec<LayerMode>,               // what this build can execute for the layer
}
pub struct ModeBytes { pub whole: u64, pub split: [u64; 2] }
pub struct KvDemand {
    pub format: KvRecordFormat,              // the family's record format (Qwen Bf16 | Fp8 lands here)
    pub unit_bytes_whole: u64,               // per pool unit when one GPU owns the layer
    pub unit_bytes_split: [u64; 2],          // per pool unit under HeadSplit: halves (GQA/KDA) or full copies (MLA latent)
    pub sequence_bytes: ModeBytes,           // recurrent/active state per slot
    pub mark_bytes: ModeBytes,               // prefix mark per slot
}
pub struct ExpertCost { pub whole: Bytes3, pub half: [Bytes3; 2], pub tp2: bool, pub spark_ok: bool }
pub struct Bytes3 { pub resident: u64, pub staging: u64, pub workspace: u64 }
pub enum LayerMode {
    HeadSplit,                               // attention heads split, KV partitioned or replicated
    Whole { gpu: u8, ffn: FfnMode },         // one GPU owns attention and KV
}
pub enum FfnMode { Split, Owner }            // Split: TP2 halves + fused all-reduce; Owner: owner-reduce (V4.1)
pub struct Movable { pub id: MovableId, pub bytes: Bytes3, pub allowed: Vec<DeviceRef>, pub colocate_with: Option<MovableId> }
pub struct Demand { pub owner: OwnerId, pub category: Category, pub lifetime: Lifetime, pub bytes: PerDevice, pub basis: Basis }
pub enum Lifetime { Resident, LoadPeak, Growth }   // Growth: graphs, allocator slack
pub struct SparkDemand { pub ranks: u8, pub slices: Vec<SliceGeometry>, pub endpoints: Vec<Endpoint>, pub capacity_rows: u32, pub workspace: u64, pub host_bytes: u64 }
pub struct Endpoint { pub role: EndpointRole /* Decode, PrefillLane(n), DeviceExchange */, pub ingress: IngressDtype /* Fp8K32, Bf16 */ }

pub struct Placement {
    pub pool_tokens: u64,
    pub layers: Vec<LayerAssignment>,        // { mode: LayerMode, experts: ExpertHome }
    pub movables: Vec<(MovableId, DeviceRef)>,
    pub hops: Vec<Hop>,                      // { after_layer, from, to, bytes_per_row }
    pub sparks: Option<SparkAssignment>,     // slices per rank, endpoints, exact ring bytes
    pub layout: cuteafd_core::memory_layout::MemoryLayout,
}
pub enum ExpertHome { RtxTp2, RtxWhole { gpu: u8 }, Spark }
```

**Algorithm.** It is deterministic, in TJ's order.
1. **Layer modes.** Start from `policy` per attention kind (section 3). One
   GPU, or no peer access, means `Whole{0, Owner}` everywhere. Colocate groups
   take one mode.
2. **Fixed demands.** These depend on the modes:
   - context/modules;
   - per-layer weights for the chosen mode;
   - embedding/head;
   - the graph set;
   - step workspaces from the family's own program set;
   - peer exchange slots and hop buffers (`hops` derived from the modes, see
     section 3);
   - coordinator Spark intake and rings.
3. **KV pool.** Each GPU's cost per unit is the sum over layers of the
   KV it owns, plus state x slots and marks x mark slots. The pool is
   `min(target, fit)` in whole units across GPUs (today's `size_pool`). An
   explicit pool is strict: if it doesn't fit, admission fails.
   - **With Sparks**, the pool is reserved before experts (b2f26af9).
   - **Spark-free** (every expert must be on RTX), experts are mandatory. The
     pool takes what is left, down to `floor`; below the floor, the request is
     Unsupported, naming the shortfall per GPU.
4. **Movables.** Drafter, vision and audio, largest first, each go to the
   allowed device with the most free bytes, keeping `colocate_with` (MiMo
   image + audio share one owner). Because TP2 halves charge both GPUs equally,
   balancing movables before experts maximizes TP2 layers
   (`min(free)/half`). Spark encoders are placed after step 5, because RTX
   layers only shrink Spark slices.
5. **RTX expert layers.** In layer order from the first MoE layer:
   - `HeadSplit` and `Whole{ffn: Split}` layers take `RtxTp2` (each half's
     resident + staging peak);
   - `Whole{ffn: Owner}` layers take TP2 with the owner reduce (V4.1) or
     `RtxWhole` where TP2 isn't built.

   Filling stops at the first layer that doesn't fit; the rest are `Spark`.
   It is one contiguous RTX prefix, as today, so Spark slices stay one range.
6. **Sparks.** Each rank's slice uses exact bytes: the worker-reported
   geometry, section 5. Rings are `compact_ring_bytes` x the endpoints at
   their ingress dtype. Workspaces come from the package. The host footprint
   is a measured constant per family. Encoders go on the lightest rank (as
   `encoder_placement` does today).
7. **Memory lever.** If the pool misses `target`, or a Spark-free layout misses,
   flip layers to the policy's next mode, kind by kind, taking first the kind
   that saves the most bytes per layer (replicated MLA KV first). Then re-run
   from step 2. The candidates are at most one flip per attention kind (≤ 4
   runs). Keep the first that meets the target, else the largest pool.

**Pool policy.** `PoolPolicy::resolve(card_bytes, context, requested)` has
one definition:
- target is 2M above 32 GiB and 1M at 32 GiB or less, never below the
  compiled context;
- floor is `max(context, 262_144)` (the agentic floor) for automatic
  Spark-free layouts, and `context` otherwise.

This replaces the 2M constants in `memory_report.rs` (`measured_pool_tokens`,
`planned_pool_tokens`) and Qwen's admission.

**P1 measured (2026-10-10): pool first costs V4 C1 until GPU1 holds experts.**
Reserving the 2M pool first on one RTX moves expert layers to the Sparks.

| card | pool | RTX layers | C1 vs p0 |
|---|---|---|---|
| Flash min | 521K -> 2.10M | 19 -> 17 | 0.939, 0.970 |
| Pro min | 356K -> 2.10M | 5 -> 3 | 0.951 |
| Flash max | 1.18M -> 2.10M | — | C1 0.979, C8 0.873 |

So V4 keeps v2's experts-first default (`Onboard::ExpertsFirst`,
`RTX_EXPERT_LAYERS=max`), resolved by the shared solver. Pool first (`auto`)
and the GPU1 EXL3 ranges (`RTX_EXPERT_PEER=on`) are opt-in. The way to get
both the 2M pool and the RTX experts is TP2 on two RTX (`P4`), not a
different single-GPU split.

**Decision (TJ, 2026-10-11): pool first is the uniform default, with no
C1 gate.** `auto` (pool first) is the default on every layout and family.
A C1 gap against experts-first is a performance bug to fix, not a reason to
keep experts-first as the default. `max` (experts first) stays a supported
mode everywhere. This corrects P2, which had kept 1-RTX V4 Pro on
experts-first after one pair.

**Decision (TJ, 2026-10-10): KV planning is the default rule; TP2 is the only
dual-RTX expert mode.**
- **The default is the pool-first rule** (`Onboard::Auto`). The C1 cost P1
  measured comes from unfinished work, not from the rule:
  - on 2 RTX, TP2 is missing (`P4`);
  - on 1 RTX, the planner's constant over-reservations (`P2`) cost the
    layers.

  `ExpertsFirst` (`RTX_EXPERT_LAYERS=max`) **stays as a supported mode**
  (TJ, 2026-10-10: "a mode I want … but it is not the default"). Only the
  default moves: `auto` becomes the V4 default in the steps that recover the
  layers, `P4` for 2 RTX and `P2` for 1 RTX, both before the first v3 cut.
  Under TP2, `ExpertsFirst` places TP2 halves first and gives the pool what
  is left (subject to its floor).
- **In a 2-RTX configuration, routed experts on the RTX cards are always TP2
  halves**, folded into the head split's existing all-reduce. `P4` deletes
  the two placements TP2 replaces, because neither can beat it on PCIe:
  - TP1 routed experts on one GPU of a 2-RTX layout (today's V4 default
    leaves GPU1 nearly empty);
  - the GPU1 whole-layer expert ranges (`RTX_EXPERT_PEER`, a hidden-state
    hop each way per layer, and the path with the two-lane deadlock).

  Every onboard mode (`auto`, `max`/`ExpertsFirst`, `N`, `N%`, `all`) stays
  and resolves to TP2 halves on 2 RTX.
- **Single-RTX layouts are TP1 by nature.** Per-layer GPU ownership
  (`Whole{gpu}`: attention, KV and that layer's work on one GPU) stays a
  general capability, and its FFN uses TP2 split or V4.1's owner-reduce per
  `FfnMode`.

**P2 outcome (2026-10-11): planner == runtime at ready on measured inventory.**
Both sides now read one inventory (`placement::inventory`): `ArchContext` per
arch (SM120 PRO 188 SMs / 101,973,491,712 B CUDA total, SM120 5090-class
170, SM121 GB10 48; context, cuBLAS, driver), the family's `ProgramSet`,
`GraphSet` (startup sets counted at ready, lazy captures as growth), exact
package scratch, and `LOADED_CODE`: the native code each family holds at ready
(lazily loaded functions, cuBLAS, package modules), measured per family,
package, split and rank. The planner charges context + loaded code; serve
reserves what has not arrived at its admission sample. The ready ledger is one
tagged report serve logs at readiness, before any request.

| card (ready ledger, worst GPU) | before (p0 planner) | after |
|---|---|---|
| Qwen EXL3 / NVFP4 min | +247 / n/a MiB | -16 / +10 |
| GLM Flash EXL3 min / max | +5469 / n/a | -3 / +0 |
| MiMo Flash min / max | +346 / n/a | +3 / +3 |
| V4 Flash 32 GB / max | refused / -102 | -15 / +17 |
| V4 Pro EXL3 min | n/a | +0 |

P1's V4 Flash max gap (1.45M planned, 1.18M served) was the planner's 95.5 GiB
card against CUDA's 94.97 GiB plus the NVML sampler's 550 MiB per GPU; at the
true total both sides now say 1,323,264 tokens and 19 layers.

Default: pool first (`auto`) for V4 Flash on one RTX (17 vs 18 layers, 2.10M
vs 1.51M pool, C1 1.07x over two matched pairs). V4 Pro on one RTX keeps
experts first: one Pro layer is 12.4 GiB, and pool first cost C1 12%
(3 vs 4 layers). Two RTX stay experts first until `P4`.

Two latent bugs from b2f26af9's GPU1 ranges are inputs to `P3`/`P4`:
- native `rtx_backbone` expert variants bind to the first CUDA device
  (`cudaErrorInvalidDevice` on GPU1);
- under two-lane prefill, the Pro EXL3 GPU1 ranges deadlock on `peer_wait`.

**P3 landed (2026-10-10): per-layer ownership in the solver, no default
change.**
- **Types.** `placement::residual` has `ResidualHome`, the section-3
  transitions, `plan_hops` and `hop_buffer_bytes`. `Placement.residual` gives
  the home at every layer boundary, and `Placement.hops` the moves.
- **Hop buffers.** Their receive buffers are `residual hops` Transport fixed
  demands: lanes × min(hops in, 2) × rows × row bytes, so 128 MiB per lane
  slot for a `[4096,4,4096]` BF16 mHC hop.
- **Executor modes.** `ExecutorModes` per family: the solver only picks what
  the executor runs (`NoMode`, `UnsupportedHop` at plan time), and the
  engine re-checks. V4 runs all `HeadSplit` or all `Whole{0}`, with the
  entry hop only.
- **The hop primitive.** `shared::peer_split::hop::HopLink` splits send from
  land, so the executor places each wait.
- **Ordering checker.** `shared::peer_split::order::check` proves a recorded
  two-stream push/wait schedule deadlock-free on the CPU.
  - Its fixture is the bug above: rank 0 waits for GPU1's expert result
    inside the unit while rank 1 waits in the next lane's attention.
  - Moving the wait to the post alone does not drain it. It drains when the
    post also follows the next lane's attention, which is the order the
    Spark-pipelined path already uses.
- **What the next steps take from P3.**
  - `P7` and `S4c` record their schedules per lane interleaving and assert
    `order::check`.
  - `P4`'s TP2 halves ride the existing FFN slot, so they add no hop.
  - `P4` deletes the GPU1 ranges path, so the bug is not fixed there.

**Two ways to fix the plan (TJ, 2026-10-10).** By default the KV pool is the
fixed definition: reserve the pool target, then place expert layers in what
is left. The solver also takes the inverse, for comparison benchmarking and
the edge cards:
- `PlacementRequest.onboard: Onboard::{Auto, Layers(n), Fraction(f)}`
  (`RTX_EXPERT_LAYERS=auto|N|N%`, `--rtx-expert-layers`). `Auto` is today's
  pool-first rule.
- `Layers(n)`/`Fraction(f)` fixes how many routed-expert layers are
  resident on the RTX cards. TP2 halves count as one layer. The solver then
  fills every remaining byte with KV, so the pool is the output, not the
  input.
  - The pool is rounded to `unit_rows`. It is the total KV shared by all
    requests, so it can exceed the compiled extent; only each request's
    context is clamped to the extent. V4 Flash max at 21 layers gives an
    8.48M-token pool, and at 0 layers 16.1M.
  - Below the floor it is a refusal that names the shortfall, the same as
    any no-fit.
  - Concurrency follows the admitted pool unless pinned.
- Which layers go local is the solver's choice: by `ExpertCost` per byte of
  Spark traffic saved, ties going to the deepest layers. Pin them explicitly
  with `RTX_EXPERT_LAYER_LIST` when a benchmark needs the exact set.
  - Until the per-layer executor (`P3`, `P7`) lands, V4 runs only a
    contiguous prefix: GPU0 gets layers `0..k` and GPU1 `k..n`. The solver
    therefore only picks the GPU split, maximizing the pool, and the pin list
    waits for `P7`.
- `0` means every expert on the Sparks; `all` (or `100%`) means Spark-free
  where it fits.
- The planner (`cuteafd plan --layout --rtx-expert-layers N`) and the runtime
  resolve the same request. The equality test covers a fixed-onboard case for
  each family that has local experts.
- Cards record the resolved `onboard` and pool in their configuration panel,
  so two benchmarks at different onboard points compare like for like.
- One launcher key replaces today's V4-only `RTX_EXPERT_LAYERS`. It lands with
  `P1` for V4 and with each family's solver port (`P6` GLM Flash, `P8` MiMo,
  `P10` Qwen, `P12` V4.1). The GLM Flash edge cards need it at `D2`, so `P6`
  or a minimal GLM Flash `Layers(n)` path lands first.

**KV record format** is a per-family input: `KvDemand.format`, with bytes from
`FamilyModel::cache_geometry(CacheOptions { .. })`. Qwen FP8 KV (v3 item 5)
adds `CacheOptions.qwen_kv: Qwen4KvCache::{Bf16, Fp8}` and its bytes formula.
The solver doesn't change. `FamilyCacheGeometry` gains
`layers: Vec<LayerCacheGeometry>` (unit bytes and split kind per layer), so
KV can follow per-layer ownership.

**Where today's code plugs in** (`trait FamilyPlacement`, one impl per
family in `placement/families/`, building `PlacementRequest` from existing
code):

| family | weights per mode | KV/state | workspaces/graphs | experts | moves from |
|---|---|---|---|---|---|
| deepseek_v4 | `layout/deepseek_v4.rs` `resident_weights` | `serving_capacity/deepseek.rs` | `v4_workspace.rs` (family-scratch 2a758f74) | `deepseek_v4_expert_cost` | `families/deepseek_v4/admission.rs` (b2f26af9) |
| deepseek_v41 | `layout/v41.rs` `resident_weights` | V4.1 source pages (5 x 91,136 B per 512 tokens, ~890 B/token), 2.7 MB FP8 window marks per sequence, source replicas | native plan APIs | `ExpertLoadBudget` per layer (not the planner's average/2 + 512 MiB) | `v41_native_serve/memory.rs` (ds41rt design's memory stage) |
| glm5 | `share_of` + `GlmDsaConfig` | `glm5` cache geometry | `glm_decode_graph_allowance` | Spark only | `glm5/mod.rs` planned path |
| glm5_flash | `load_conversions` + split weights | `glmf` cache geometry (`GlmfIndexCache`) | `glmf_step_workspaces`, `partial_exchange_reserve` | `Fp8Experts::bytes_for`, EXL3 residency | `glm5_flash/mod.rs` `kv_admission`/`measured_admission` |
| mimo_v2 | `MimoResidentLayout` | `MimoKvCache` geometry | MiMo native contracts (`admission.rs` ~1043), decode-graph count | `Fp8Experts::bytes_for` | `mimo_v2/admission.rs` preflight (CPU parts move to the loader) |
| qwen4 | `checkpoint_resident_bytes` | `qwen4` geometry | graph-set iteration (`admission.rs` ~42) | `qwen_exl3_arenas`, FP8/NVFP4 package scratch | `qwen4/admission.rs` |

**Runtime side.** `cuteafd-daemon/src/shared/placement.rs` has
`admit(library, &dyn FamilyPlacement, args) -> Result<Placement>`.
1. It creates CUDA contexts and loads the family's program modules.
2. `RuntimeInventory::measure(library, gpus)` takes one sample per GPU
   (`Baseline::Measured`).
3. It calls `solve`.
4. It logs `Placement` once (the same line `cuteafd plan --layout` prints).

Families allocate in any order after that, but never re-sample to size the
pool. This replaces GLM Flash's two paths and V4.1's synthetic dual sample,
and makes MiMo's module-delta measurement the context term.

**Encoder-before-experts fix.** `resolve_encoder` runs at layout.rs ~977,
before V4.1's automatic local layers (~1008), whose Spark slices are then
rewritten (~1021). V4 instead places its local layers before encoders (~775).
Steps 4-6 give every family one order:
1. RTX encoder and drafter reservations, before experts;
2. Spark encoders, after expert slices are final.

**Estimates replaced by measured or exact items:**

| item today | replacement |
|---|---|
| `family_costs.runtime_bytes` (V4 +95.7 MB vs measured) | Runtime: measured `context_bytes`. Planner: per-arch context constant (measured once per driver major and SM, a table in `placement/pool.rs`) + module bytes from PROGRAMS.json (`programs[].module_bytes`, added by the exporter) |
| `graph_bytes` (Qwen: 512 MiB planned vs a 5.0 GB set; MiMo 664 vs 512 MiB; GLM Flash > 1.5 GiB) | One `GraphSet::startup(shape)`, shared by runtime warm-up and planner, x a measured per-arch bytes-per-executable (Qwen 149,712 B, MiMo 192 KiB) + driver margin. `Lifetime::Growth` items cover lazy captures |
| `workspace_bytes` allowances | Program-manifest formulas, which V4 and GLM Flash already have; add MiMo, GLM 5.3 and Qwen from their engine's workspace functions |
| local-expert allowance (V4 -13.6 MB; Qwen NVFP4 omits 1.06 GB package scratch) | `ExpertCost` from the catalog (resident, staging, package scratch from `Fp8MoeInfo`/EXL3 manifests) |
| external drafter = safetensors bytes + 1300 MiB | The drafter's resident representation (FP8/BF16) + its workspace formula |
| Spark workspace/ring allowances | Worker-reported admission (section 5) |

**Planner == runtime.** One function, so the test compares inputs:
- Each family has `planner_equals_runtime_<family>` in
  `cuteafd-daemon/src/shared/placement/tests.rs`. It builds the planner
  request from a fixture snapshot (`plan/testing.rs` writers, e.g.
  `write_v4_snapshot` from b2f26af9) and the runtime request through the
  family's daemon path, with a `FakeProbe { total, context }`.
- It asserts equal `Placement`, including identical layout items per
  (category, group).
- Configs: 1 RTX, 2 RTX, 2 RTX asymmetric, 32 GB, Spark-free.
- The hardware ledger check (`scripts/bench/memory-audit.py`) stays the gate:
  planner vs measured per category within 64 MiB at ready.

### 2. TP2 RTX experts for every family

**Module:** `cuteafd-daemon/src/shared/experts/rtx.rs` plus `rtx/{native,
fp8moe,exl3,combine,routes}.rs`.

```rust
pub enum RtxShard { Whole, Tp2 { rank: u8 } }
pub trait RtxExpertLayer {
    fn shard(&self) -> RtxShard;
    fn partial(&self) -> PartialDtype;                 // F32 (native, EXL3) or Bf16 (fp8moe today)
    fn workspace_bytes(&self, rows: u32) -> u64;       // exact; feeds ExpertCost
    /// This rank's routed sum for `rows` rows into `out`: no shared add, no
    /// inter-rank reduce, no finish/copy. Input BF16 x (fp8moe, EXL3) or
    /// FP8-K32 rows (native), routes = ids + weights [rows, topk].
    unsafe fn enqueue(&self, stream: &Stream, input: ExpertInput, routes: Routes, out: Partial) -> Result<()>;
}
pub enum Combine {
    /// HeadSplit / Whole{ffn: Split} layers: each rank adds its routed partial
    /// to its shared/dense half in FP32, then the existing FFN slot exchange
    /// sums the two in rank order on both GPUs.
    FusedAllReduce { exchange: ExchangeDtype },        // Bf16 | F32
    /// Whole{ffn: Owner} layers (V4.1 today): broadcast inputs/routes, reduce
    /// routed then shared in rank order onto the owner, add in BF16.
    OwnerReduce { owner: u8 },
}
pub enum RouteSource { Replicated, Broadcast }         // Broadcast: rank0 pushes ids+weights (rows*topk*8 B)
```

Impls, each wrapping existing code:

| impl | wraps | geometries |
|---|---|---|
| `NativeTp2` | V4.1 `RankWeights`/`RankWave`/`BackboneTp2` (`v41_experts/tp2.rs`), generalized over `ExpertGeometry` | v41, dsv4f, dsv4p |
| `V41Nvfp4Tp2` | `v41_nvfp4_tp2_expert_*` | v41 NVFP4 |
| `Fp8MoeTp2` | `Fp8Experts::load(tp = 2, rank)`, `exact_layout` | FP8, MXFP4 (mimof/mimop), NVFP4 (glm, glmf, qwen4) |
| `Exl3Tp2` | `residency(sel, 2, rank)` + `launch_layer_into` (raw FP32, no `reducer.finish`) | dsv4f, dsv4p, glm, glmf |

**Combine numerics.** Fused combine computes
`out = BF16(FP32(p0) + FP32(p1))` with `p_r = round(routed_r + shared_r)`.
Spark reduction instead sums routed rank planes first and adds shared after,
so this is a numerics change: it needs its own fidelity gate per format and
top-k (top-6/8/10).
- `ExchangeDtype::F32` (4·T·H bytes, as GLM Flash's FP32 KDA partials)
  rounds once.
- `Bf16` keeps today's payload.

Measure F32 decode first (see open questions).

**Routes.** GPU1 gets a replica of the router weights: the `Router`
component becomes `Share::Replicated` under TP2, and V4 also takes `gate.bias`
or `tid2eid` plus token ids for hash layers. Each rank runs the router on its
bitwise-identical input. `RouteIdentity::check(engine)` runs at startup
after graph warm-up, on a fixed 512-row prefill plus one decode step:
1. Copy ids + weights from both ranks for every MoE layer.
2. Compare bytes.
3. On a mismatch, log the first layer and row, then switch the engine to
   `RouteSource::Broadcast` (V4.1's current method, one DIRECT push per MoE
   layer).

`CUTEAFD_ROUTE_CHECK=N` re-checks every N steps (off by default).

**TP2 expert loader (P4, 2026-10-11).** Each TP2 layer pair is read from
storage once, into pinned banks, and both GPU halves are uploaded from it
(32 readers). Storage bytes per layer pair: Flash 4.56 → 3.42 GB, Pro
10.61 → 6.37 GB. Cold expert load against the duplicate-read loader: Flash
22.1 → 17.3 s; Pro 16.7 → 17.0 s, 1.7% slower. Pro's load is CPU-read bound
(16.4 s reading, 0.26 s submitting, ~0 bank wait). 48 EXL3 readers made it
worse (+3.7%). Follow-ups, measured on their own: GPU-direct reads (cuFile /
O_DIRECT into pinned banks) and pinned prefetch of the next layers.

**TP2 lane payload sizing (P4 follow-up, 2026-10-10).** The persistent
per-lane TP2 payload is sized for FP32 (4 B per element), so the BF16 and
FP32 exchange share one buffer and both dtypes can be warmed. Once BF16
becomes the default, that over-reserves 64.5 MiB per rank on V4 Flash and
112.9 MiB per rank on Pro (at 4096 prefill / 64 decode rows). Size the
payload to the exchange dtype actually in use, through the geometry,
layout and admission APIs and their tests, as its own small PR after P4. On
Pro that is about 225 MiB across both GPUs, part of the margin toward its
11th TP2 layer.

**EXL3 TP2 input (P4, 2026-10-10).** `Exl3Tp2` takes FP8 K32 wire rows, as
V4.1's rank sequence does, so V4.1 stays byte-exact through the shared impl.
Each rank quantizes its own copy of the post-split hidden state. Follow-up,
measured on its own: native BF16 input for EXL3 TP2, which drops the FP8
quantize and the per-capacity wire-decode workspace.

**Route source follows the layer mode (TJ, 2026-10-09).**
- `HeadSplit` layers: both GPUs already hold the post-attention hidden state,
  so replicate the router (0.1-0.7 GiB total; per MoE layer 2-11 MB); a
  route broadcast would add a 3-5 us hop per MoE layer (~0.5-1.5% of a C1
  step) that nothing else needs.
- `Whole{gpu}` layers with TP2 experts: the owner must push the hidden rows
  (8-14 KB/row) to the peer anyway, so broadcast the routes (48-80 B/row) in
  that same push; no router replica, no extra hop (V4.1 today).

**Kernels, exports, loaders.** The table extends the v4-placement audit;
paths are under `python/tools/aot/`, `native/cmake/shared/` and
`rust/crates/`.

| family / format | SM120 half export | still needed |
|---|---|---|
| V4.1 MXFP4 / NVFP4 / EXL3 | exist (`rtx_tp2`, `v41_tp2_experts.cmake`, `v41_nvfp4_experts.cmake`, `rtx-tp2`) | none: reference impls |
| V4 Flash/Pro official MXFP4 | `export_b12x_slices_aot.py` `role_geometry` gives I/2 = 1024/1536; FFI role-3 geometry and `family_symbol` already work | **S**: `expert_families.cmake` ~44 admits `rtx_tp2` as coordinator and maps it to `cuteafd_{dsv4f,dsv4p}_tp2_expert_*` |
| V4 Flash/Pro EXL3 | `shard_profiles` emits `rtx-tp2` (Pro k23 package built) | **M**: `deepseek_v4/local.rs` ~332 loads `BackboneFull` and calls `reducer.finish`; switch to `Exl3Tp2` raw partials |
| GLM Flash EXL3 K3/K3.25/K4 (incl. tr3) | `rtx-tp2` emitted (I = 2048, 8+8 blocks) | **M**: GLM Flash local EXL3 path (`engine.rs` ~3708) drops wire quantization, finish and D2D copy |
| GLM Flash / GLM 5.3 / MiMo FP8, MXFP4, NVFP4 | coordinator layouts are `tp1` only (`package_fp8_moe_aot.py` ~75 `ROLE_LAYOUTS`, `MXFP4_ROLE_LAYOUTS`, `NVFP4_ROLE_LAYOUTS`); the compiler builds tp2 with `--layouts` | **S/M**: add coordinator `tp2` (exact H128) in those tables and `fp8_moe.cmake` ~64, BF16 input. No new kernel math |
| GLM 5.3 any format | none (no coordinator local expert backend, `glm.rs` ~563) | **L**: a local backend. Not needed for Spark-free (338-380 GiB of experts never fit); needed only for RTX layers at max. Deferred |
| Qwen EXL3 / FP8 (I = 640, 5 blocks) | `exl3.cmake` ~158 drops `rtx-tp2` | **M**: unequal 384/256 profiles. Not needed if Qwen uses layer ranges (section 3) |
| Qwen NVFP4 | padded 320+64 / 320+64 via existing loader | **S**: coordinator `tp2` layout. Same caveat |
| MiMo V2.6 true FP8 weights | no `GEOMETRIES` entry (`mimof`/`mimop` are MXFP4) | only if an FP8-weight V2.6 checkpoint becomes a target |

Cleanup that pays with TP2: MiMo `moe_front` (`engine.rs` ~2419) builds an
FP8-K32 wire buffer the BF16 local package never reads (S). Local EXL3 in
GLM Flash, Qwen and V4 quantizes wire rows, finishes to BF16 and copies
(M; it changes numerics, so it carries its own gate).

**Drafter under TP2.** The whole drafter is one `Movable` (default GPU1).
V4.1 already runs dSpark there.
- **Taps.** Rank 1 holds the replicated residual for every non-terminal
  layer. Terminal taps need rank 1's final FFN close, which every
  head-split engine skips today. Enable it when the drafter is on GPU1: one
  extra exchange per step at the last layer.
- **Head and embedding are solver-placed `Movable`s (TJ, 2026-10-09).**
  Sizes (vocab x hidden, untied everywhere): head FP8 0.49-0.89 GiB,
  embedding BF16 0.99-1.77 GiB (V4 Flash 129,280 x 4,096 ... GLM 5.3
  154,880 x 6,144). Rules:
  - Layer ranges, or GPU1 owning the last layers: the head lives on the
    last-layer GPU with the drafter, the embedding on the first-layer GPU (or
    host-mapped). No copy; at most one 4-6 KB hidden-row hop per step (~3 us)
    when the final layer is on the other GPU.
  - Head split on the final layers: shard the vocabulary across both GPUs,
    ratio chosen by the solver (V4.1's uneven split for a cache target is
    the precedent), one small argmax exchange.
  - Never replicate the head unless both of the above lose on measured C1.
- **Embedding.** Draft rows read the host-mapped embedding (one pinned copy),
  so no second device copy.
- **Prefix state.** DFlash restores cold (`valid_from`), so nothing moves.
  The arenas that do move to the drafter's GPU: V4 dSpark window marks, MiMo
  `--mimo-prefix-draft` rings, and Qwen MTP stash (`[T,4,2560]`).
- **Draft experts.** V4 dSpark stage experts (9.56 GiB Flash, 17.78 GiB Pro)
  and Qwen MTP experts stay full width with the drafter. DsparkTp2 EXL3 stays
  unsupported.

### 3. Attention placement: per-layer ownership

**`ResidualHome`.** Ownership composes per layer through one state machine
in `shared/peer_split.rs`: `ResidualHome::{Replicated, Owned(gpu)}` and
`fn transition(home, next: LayerMode) -> (Option<Hop>, ResidualHome)`. The
executor follows `Placement.layers`, and the solver charges the hop buffers.

| from \ next layer | HeadSplit | Whole{g, Split} | Whole{g, Owner} |
|---|---|---|---|
| Replicated | attention all-reduce | none in; broadcast g -> peer after attention | none in |
| Owned(g) | broadcast g -> peer | none | none |
| Owned(other) | broadcast | boundary hop | boundary hop |

Notes:
- Hops carry the residual: `[T,H]` BF16, or `[T,4,H]` + FP32 pre-coefficients
  for mHC families (V4, GLM Flash, Qwen, V4.1).
- After a Split FFN the residual is Replicated: the FFN all-reduce,
  bidirectional.
- An `Owner` FFN leaves it Owned(g).
- **Layer ranges** are runs of `Whole{g, Owner}`: one hop per boundary.
- **A single whole layer between split layers** (`Whole{g, Split}`) costs
  one broadcast plus the FFN all-reduce: the same two exchanges as a
  head-split layer. The peer idles through the owner's attention.

**KV follows ownership.** Whole layers keep KV, index keys and recurrent
state on the owner only. HeadSplit layers partition KV heads where the
format allows: GQA (MiMo 4+4 KV heads), KDA heads (GLM Flash), GDN (Qwen 8/24
heads, if split). They replicate MLA/DSA/CSA latents (GLM 5.3, GLM Flash
MLA, V4). Prefix marks and graphs are owned per GPU, as head-split marks are
today.

**Executor work** (none of the generic engines has it: `attach_peer`
requires every layer split):
- a per-layer plan in each engine loop;
- unsplit programs and weights for Whole layers on both GPUs;
- per-owner caches;
- the hop primitive.

V4.1's `BlockTransfer` (`v41_block/transfer.rs`) is the reference for the
boundary hop: an SM copy in the deferred device chain, otherwise peer DMA plus
a cooperative wait.

**Policy** (`LayerPolicy`, defaults per family and attention kind, preference
order; `--layer-modes auto|split|ranges|<kind>=whole,...` and
`COORDINATOR_SPLIT=auto|heads|ranges|off` in run-family):

| family | kinds | default with Sparks | memory fallback (lever) | evidence |
|---|---|---|---|---|
| V4 Flash/Pro | CSA (replicated) | HeadSplit | ranges | decode -10% / -12% (5560ed00); ranges halve the 7.9-11.2 GiB per-GPU KV at 2M |
| V4.1 | CSA + source groups | ranges (20/20) | none | head split / TP2 attention C1 187 -> 170 (100122a6); ds41rt design keeps it |
| GLM 5.3 | MLA/DSA (replicated, 53,940 B/token) | HeadSplit | ranges | verify 27.9 -> 25.2 ms, coordinator prefill -28% (7e27a769); at 2M ranges plan 81.0 / 69.9 GiB vs head split capped at a 1.30M pool |
| GLM Flash | KDA (partitioned) + DSA (replicated) | HeadSplit all | KDA HeadSplit, DSA `Whole{alternating, Split}` | C1 159 -> 168 (9380e8b8) |
| MiMo | GQA full + SWA (partitioned) | HeadSplit all | ranges | -41% coordinator decode (3a1d428c) |
| Qwen | GDN + GQA (2 KV heads) | Whole (1 GPU) | ranges on 2 RTX | no split exists; hidden 2560 makes hops relatively costly |

**Rule.** Split attention by default only where a measured C1 A/B on the
family's min/max shows a win. Whole layers are the memory lever the solver
uses only when the pool or Spark-free fit requires them. A family adopts a
non-default mode only through the quick A/B at 2M on min/max.

**GLM Flash compact index cache.** It is single-GPU only today
(`serving_capacity.rs:395`, `engine.rs:1645`). Under the mixed policy the
DSA layers are Whole, so each MLA layer's compact index lives on its owner
with no split variant to build. That saves 11.0 GiB at 2M in total (5.5 per
GPU). Shared-indexer layers (GLM 5.3) use `colocate`.

### 4. Spark-free layouts as solver outputs

No special cases: `Inventory.sparks` is empty, experts are mandatory, and
the pool floats down to the floor (section 1, step 3).

The table projects GiB GPU0 / GPU1 at fixed pools from the planner's one-
and two-GPU items:
- vision off (it adds 1.79 GiB to the lighter GPU);
- capacity 93.5 / 93.5, except V4 at 90.0 / 89.3;
- head split moves the drafter to GPU1;
- ranges use the best split point (22-24);
- mixed is GLM Flash's KDA split with its 11 DSA layers Whole and alternating.

| model | pool | head split + TP2 | ranges | mixed |
|---|---:|---|---|---|
| GLM Flash K3 | 2M | 94.1 / 94.9 no | 80.4 / 80.4 fits | 85.1 / 81.0 fits |
| GLM Flash K3 | 1M | 82.6 / 83.4 fits | 74.5 / 74.8 fits | 78.8 / 75.7 fits |
| GLM Flash K3.25 | 2M | 98.6 / 99.4 no | 85.0 / 84.8 fits | 89.6 / 85.3 fits |
| GLM Flash K3.25 | 1M | 87.0 / 87.8 fits | 79.1 / 79.1 fits | 83.3 / 80.0 fits |
| GLM Flash tr3 K4 | 1M | 100.3 / 101.1 no | 92.7 / 92.1 fits | 96.9 / 93.0 no |
| GLM Flash tr3 K4 | 512K | 94.6 / 95.4 no | 89.7 / 89.3 fits | 93.8 / 90.4 no |
| GLM Flash NVFP4 | 512K | 103.2 / 103.9 no | 98.5 / 97.7 no | 102.6 / 98.8 no |
| V4 Flash, dSpark | 2M | 90.1 / 97.6 no | 89.2 / 88.7 fits (tight) | - |
| V4 Flash, dSpark | 1M | 86.2 / 93.6 no | 87.1 / 86.9 fits | - |
| V4 Flash, no dSpark | 2M | 90.1 / 87.8 no (GPU0 +0.06) | 85.6 / 82.6 fits | - |
| V4 Flash, no dSpark | 1M | 86.2 / 83.9 fits | 83.6 / 80.6 fits | - |
| MiMo Flash | 512K | 89.2 / 92.1 fits | 87.7 / 90.9 fits | - |

Rows that fit nowhere at a higher pool are omitted: tr3 K4 at 2M, NVFP4 at
1M and 2M, and MiMo Flash at 1M and 2M. MiMo Flash at 1M fits with ranges once
audio is off or on a Spark (-3.5 GiB on GPU1).

Qwen:
- 1 RTX: the solver output equals today: 525,568 tokens (EXL3 K4.25) and
  436,480 (NVFP4).
- 2 RTX with ranges: ~70.7 GiB per GPU at 2M (from the one-GPU 137.6 GiB plan
  plus duplicated context/graphs/workspaces), so it fits 2M with no head
  split.
- FP8 KV (item 5) raises the 1-RTX pool.

**Reading of the table.**
- **Layer ranges, not head-split TP2, are what fit Spark-free at 2M.** TP2
  halves the experts, but the head split replicates MLA/CSA KV and duplicates
  per-GPU fixed costs. Ranges own KV once.
- **GLM Flash K3/K3.25 at 2M:** mixed keeps the measured KDA head split.
- **tr3 K4 needs ranges (1M).** Official NVFP4 and FP8 stay unsupported.
- **V4 Flash Spark-free:** ranges at 2M, or head split + TP2 at 1M without
  dSpark.

Caveats:
- These are arithmetic, not admission.
- Hop buffers (`[4096,4,H]` BF16 = 128 MiB per lane at H = 4096) are not in
  them.
- The rc3 launcher still rejects V4 with zero Sparks, and the planner rejects
  zero-Spark DeepSeek (`deepseek.rs:432`). The solver's `spark_ok`/`modes`
  replace both checks.

### 5. Exact Spark admission

**Endpoints and ingress per family.** The coordinator states them; the worker
enforces them.

| family | endpoints (default) | ingress |
|---|---|---|
| deepseek_v4 | 2 (lanes) + optional device exchange | FP8 K32 |
| deepseek_v41 | 2 + optional device exchange | BF16 (NVFP4), else FP8 K32 |
| glm5 | lanes + 1 = 4 (1 with one lane) | FP8 K32 |
| glm5_flash | lanes = 2 (configurable) | FP8 K32 |
| mimo_v2 | main + lanes - 1 = 2 (Flash) / 3 (Pro) | `--expert-input`: FP8 K32, BF16, or BF16 decode / FP8 prefill per endpoint |
| qwen4 | 1 | FP8 K32 |

**Protocol.** `FamilyPlacement::spark(ctx)` returns `SparkDemand.endpoints`.
The coordinator sends a `ModelAdmission` request in the protocol_v2 bootstrap
carrying:
- the checkpoint identity;
- the rank and world;
- `Vec<Endpoint>`;
- capacity rows.

The worker computes rings with `spark_ring_bytes_for_ingress` per endpoint
and builds `RingBudget::new(exact)` for every family. This replaces
`usize::MAX` (`service.rs:380`) and `parse_rdma_endpoints`' fixed 2
(`service.rs:331`), and the ingress flag moves from `service/local.rs:41` to
the endpoint list.

The worker replies with `ModelAdmissionResponse`:
- resident, workspace, ring and host bytes;
- slice widths and the EXL3 per-projection tier and geometry it actually
  loaded (the sliced-checkpoints v3 design).

The coordinator checks that all ranks agree and equal the solver's
`SparkAssignment` before allocating. Any mismatch fails startup, naming the
rank and field.

**Shared scratch.** `ProgramSet::for_serving(family, split_family,
draft_family)` lives in `cuteafd-loader/src/serving_capacity/program_set.rs`.
- **Users:** the runtime scratch allocators and the planner workspace
  formulas. V4 already moved to this on work/family-scratch 2a758f74; GLM 5.3,
  GLM Flash, MiMo and Qwen already select their own programs.
- **Test:** for each family, planner set == runtime set.

### 6. Migration (ordered PRs, each gateable)

Gates on every PR:
- cargo/script tests by failing id;
- golden NLL/greedy on one GPU or loopback;
- the PR's own measurement;
- the quick A/B at the 2M operating point on min and max where serving
  changes.

Full V4.1 parity applies when a shared hot path changes (marked †).

| # | PR | size | gate | notes |
|---|---|---|---|---|
| 1 | `placement` module, `PoolPolicy`, `solve` (HeadSplit/Whole modes, no TP2), V4 port of b2f26af9 (pool first, GPU0/GPU1 whole-layer ranges), `planner_equals_runtime_deepseek_v4` | M | equality test; V4 Flash/Pro A/B at 2M: max pool 603K -> 2M must not cost C1 | V4 first: worst pool, empty GPU1, an equality test exists, native TP2 is CMake-only |
| 2 | Exact items: `RuntimeInventory::measure`, per-arch context table, `GraphSet`, `ProgramSet`, exact expert package scratch; planner uses them for all families | M | ledger compare at ready within 64 MiB: V4 small, Qwen local EXL3/NVFP4, MiMo, GLM Flash | closes the PLAN-listed gaps (Qwen +5.35/6.72 GB, V4 84 MB) |
| 3 | `ResidualHome`, hop primitive, `LayerMode` in `Placement`; engines assert their mode set | S | unit tests | no behavior change |
| 4 † | `shared/experts/rtx` (`NativeTp2` from V4.1 `RankWave`, `Combine`, `RouteIdentity`), dsv4 `rtx_tp2` CMake, V4 engine wiring (router replica, `post_split`/`peer_front`) | L | V4 Pro EXL3 K2 quick KL no worse than rc1 (0.0605 min / 0.0596 max) and Flash golden; route check passes; A/B min/max | V4.1 byte-exact through `NativeTp2` before its private copy is retired (PR 12) |
| 5 | Drafter as a movable: V4 dSpark (with stage experts) on GPU1, terminal close, marks; `shared/vocabulary::ShardedVocabulary` (the ds41rt design's item 9, V4.1's two-GPU head moved) for the head shard | M | lossless spec check; C1 at max | one sharded head for both designs (revised 2026-10-09) |
| 6 | GLM Flash: solver port, coordinator tp2 packages (fp8moe), `Exl3Tp2`/`Fp8MoeTp2` wiring, router replica, DFlash2 on GPU1 | M | golden NLL (2.4054 split baseline); A/B min/max at 2M | |
| 7 | Per-layer executor for GLM Flash: Whole DSA layers, compact index on owners; Spark-free 2 RTX K3/K3.25 at 2M; tr3 with ranges at 1M | L | fidelity; Spark-free card correctness; A/B min/max | proves the mixed mode |
| 8 † | MiMo: solver port, bidirectional MoE exchange, `Fp8MoeTp2`, drop unused wire buffer | M | golden; A/B min/max (max should shed 12 Spark layers) | |
| 9 | GLM 5.3: solver port, ranges vs head split at max (2M vs 1.30M pool) | M | A/B C1/C4 both modes at max; pick by section 3's rule | RTX experts need the local backend (L, deferred) |
| 10 | Qwen: solver port, layer ranges on 2 RTX (Spark-free 2M), FP8 KV option plugs in (item 5) | M | golden; 1 RTX local + 2 RTX ranges A/B | head split only if later measured |
| 11 † | Exact Spark admission: `ModelAdmission` request/response, endpoints and ingress, finite `RingBudget` for all, EXL3 worker geometry | M | every family's min/max: ring peak == charge (as work/spark-vision-ring) | |
| 12 † | **= ds41rt stage 4a+4b.** V4.1 onto the solver (`FamilyPlacement` impl, `NativeTp2`/`V41Nvfp4Tp2`/`Exl3Tp2` with `OwnerReduce`, `PlacementHandoff`, coordinator exchange on `SparkLink`, `planner_equals_runtime_deepseek_v41`); delete `memory*.rs`, `placement.rs` and `v41_experts/tp2.rs` `RankWave`/`RankWeights` once shared impls serve it byte-exact | L | V4.1 golden byte-exact; the stage's standard gates; full 3-session parity at the release cut | one branch for both designs (revised 2026-10-09) |
| 13 | Delete `family_costs` rows and `layout.rs` family branches | S | planner fixture tests | folded into each family's solver port (PRs 1, 6, 8, 9, 10, 12), not a PR of its own (revised 2026-10-09) |

**V4.1 reuse.** The `NativeTp2`, `V41Nvfp4Tp2` and `Exl3Tp2` impls are
V4.1's code moved, not rewritten:
- `RankWeights`/`RankWave`/`BackboneTp2` staging;
- `load_exl3_pair`;
- `PeerReduction` as `Combine::OwnerReduce`.

V4.1 keeps 20/20 ranges and its owner reduce. The fused combine is offered to
it only through its own gate, because fusing changes rounding.

### Risks
- **Fused-combine numerics.** Changing to BF16 per-rank rounding can move KL.
  F32 exchange costs 2x payload. Gate per format.
- **Graphs.** The replicated router and rank-1 routed work add GPU1 graph
  executables. They belong in `GraphSet`, or admission under-counts again.
- **Hop bandwidth.** Mixed/ranges prefill hops are 128 MiB per lane per
  boundary at 4096 rows. Peer bandwidth is ~1.1 ms per 48 MiB (2026-10-01),
  and saturated host -> GPU0 ingress quadruples small hops. Measure 8K
  prefill on max.
- **Measured baseline.** If another process holds GPU memory, the measured
  sample sees it. That is correct behavior but makes planner != runtime; the
  test covers only the fake probe.
- **Readiness.** The route check adds a 512-row prefill per start (~0.1-0.5
  s), and TP2 loading reads half-slices on both GPUs in parallel. Load speed
  must not regress.
- **Scope.** GLM 5.3 RTX experts need a new local backend. Qwen TP2 needs
  unequal EXL3 profiles. Both are deferred unless their gates call for them.

### Open questions for TJ (with recommendations)
1. **Fused combine dtype.** Recommend F32 partials for decode/verify (≤128
   rows, ≤ 6 MiB) and BF16 for prefill, keeping BF16 only where KL matches.
2. **GLM 5.3 at max.** Head split with a 1.30M pool, or layer ranges with 2M?
   Recommend measuring both (PR 9). Default to ranges unless the head split
   wins C1 by more than 2%, since 2M is the agentic operating point.
3. **Spark-free pool floor.** Recommend auto-shrinking to
   `max(context, 256K)`, with cards advertised only at ≥ 1M.
4. **Qwen on 2 RTX.** Recommend layer ranges only (2M fits); no Qwen head
   split or unequal TP2 unless measured.
5. **Draft head on GPU1.** Decided (TJ): no replica. The head goes with the
   last layer and the drafter, the embedding with the first layer; shard the
   vocabulary (solver-chosen ratio) under the head split.
6. **Route mismatch.** Recommend falling back to broadcast routes with a
   warning, not refusing to start.
7. **V4 Spark-free.** Recommend ranges at 2M with dSpark, rather than head
   split + TP2 at 1M without it. Confirm by A/B once PR 7's executor exists for
   V4.
8. **GLM Flash with Sparks.** Recommend keeping the head split (measured +5%
   C1). Ranges only as the memory lever, unless TJ's earlier GLM Flash range
   numbers show a C1 win (please point to them; the audits found none in
   git).

## v3 attention placement without latent replication (design, 2026-10-10)

Design for v3 item 11. Nothing here ran on hardware. Inputs:
- memory: CPU `cuteafd plan --layout` from the P2 branch (work/v3-p2 0fcd554e,
  94.97 GiB CUDA total, 92.97 GiB capacity after headroom), and V4 TP2
  numbers from P4's 461e7db1 commit table (95.5 GiB card, 0.53 GiB per card
  taken off);
- cost: the measured head-split gains below, the 2026-10-01 peer probe
  (two-way 3.4 / 4.8 / 25 us for 12 KiB / 96 KiB / 1 MiB, so about
  3.4 us + bytes / 42 GB/s), and byte arithmetic;
- schedules: a Python port of P3's `order::check`.

Scripts, plans and output are in `builds/v3-kv-context-design/`
(`plans.sh`, `model.py` -> `model.out`, `order_check.py`).

Three placements for MLA/DSA/CSA layers, chosen per layout
(`ATTENTION_PLACEMENT=context|layers|heads`):
- `heads`: today's head split, with the latent replicated on both GPUs.
- `context`: the latent is token-split. A new `LayerMode::ContextSplit`:
  projections stay head-split, KV is partitioned by page, and the residual is
  replicated after the layer, exactly as under `HeadSplit`.
- `layers`: P3's `Whole{gpu, Split}`, alternating by colocate group. KV
  lives on the owner, and the FFN stays split as today.

### 1. Token-split decode and verify

**Reuse.** Every decode kernel involved already splits KV and merges the
pieces by LSE. GLM's `glm_sparse_mla` decode route writes "normalized BF16
partials + LSE" per split, then runs `SparseMLASplitDecodeMergeKernel`.
V4's decode route does the same, ending in the sink merge. `context` makes
the peer's token shard one more split. The prefill MG kernels already write
a base-2 LSE.

**Dataflow, GLM 5.3 (78 MLA layers, 21 full-indexer and 57 shared-indexer).**
Per layer, on both GPUs, in queue order:
1. **Input norm.** Replicated, as today.
2. **`glm2_producer`.**
   - `w_qkv_a` and the latent are replicated.
   - `w_q_b` and `w_uk` stay head-split, so each GPU produces the absorbed
     576-wide query of its own 32 heads.
   - Each row's 656-byte record is written only by the owner of the row's
     page. On the other GPU the row's slot points at a per-GPU scratch row,
     the mechanism `StepTables::pad_rows` already uses, so the producer is
     unchanged.
3. **Push q** (own heads) to the peer: 36,864 B per row.
4. **Indexer (full-indexer layers only).**
   - The index producer stays replicated. Only the page owner writes the
     index keys.
   - Each GPU runs the index top-k over its own pages only, through a
     compacted half-width page table.
   - The scored variant emits up to 2,048 candidates per row as
     (FP32 score, global logical index): 16,384 B per row. Push them.
5. **Merge candidates** (full-indexer layers only).
   - Wait for the peer's list. Both GPUs merge the two lists into the global
     top-2048 by (score desc, logical index asc).
   - That is the deterministic kernel's own tie key (`_tie_key = ~gidx`:
     the lower logical index wins).
   - Each GPU keeps the picks that fall on its own pages, as physical slots
     with a length (an ascending compacted list).
   - Shared-indexer layers reuse that local list with no exchange: all
     layers share page ids, so ownership is the same in every layer of a
     colocate group.
6. **Attention.** Wait for the peer's q. Sparse MLA then runs over all 64
   heads on the local selection.
   - It does the peer's 32 heads first and pushes that normalized partial
     plus its LSE: 32 × 512 × 2 + 32 × 4 = 32,896 B per row, with `has_sink`
     off.
   - Then it computes its own heads, which hides the push.
7. **Combine.** Wait for the peer's partial of this GPU's heads, then run
   `lse_combine2` (GPU0's partial first, then GPU1's) into the BF16
   `[T,32,512]` attention output.
8. **`glm_o`.** `W_UV` and `o_proj` stay head-split, followed by today's
   attention all-reduce (12,288 B per row) and post-norm.

**The scores are exact.** A token's index score is computed by the same
per-token arithmetic wherever its page lives, and the tie key is a total
order. So the merged selection equals today's selection bit for bit. A
check mode (`CUTEAFD_CONTEXT_CHECK=N`) compares it against a full
single-GPU top-k. Only the attention summation changes.

**V4 Flash/Pro (CSA).** `ContextSplit` applies to the C4 layers only:
21 / 30 layers, holding 96% of V4's record bytes.
- **What stays head-split and replicated:** C128 layers (1,728 B per
  256 tokens; 0.26 / 0.41 GiB per GPU at 2M for Flash / Pro), window-only layers, the window rings
  and the compressor carry (per-sequence state).
- **Records:** the compressor runs replicated. Only the owner of a
  256-token unit writes that unit's C4 compressed rows and index keys.
- **Top-k:** the global top-k merges `index_topk` (512 / 1,024) candidates of
  (score, global compressed-row index): 4,096 / 8,192 B per row.
- **Window:** in a context layer, the window part of the union is attended by
  one GPU, alternating by layer parity. The other passes
  `swa_lengths = 0`.
- **Query:** 32,768 B per row (Flash) / 65,536 (Pro).
- **Output:** `wo_a`/`wo_b` stay group-split.

**GLM Flash DSA (11 MLA layers; the 34 KDA layers stay `HeadSplit` with
their partitioned state).**
- **Interleave:** by 256-token unit, so a 4-token pool never straddles
  owners.
- **Top-k:** over pooled keys, 512 candidates per row (4,096 B), then
  `index_expand` on each GPU over its own pools.
- **Open tail pool:** `index_kpool_always_select_tail` selects it; it is
  attended by the owner of its unit.
- **Index cache:** `context` requires `--index-cache compact`. Each
  sequence's tails (at most 3 rows per MLA layer) are per-sequence state,
  computed replicated, so pooled keys never need a peer's token keys. This
  also makes the compact cache available on two GPUs, which the head split
  refuses today.
- **Query:** 32,768 B per row (absorbed 512).
- **Page 0:** each GPU keeps its reserved zero page, because masked
  candidates read slot 0.

**Merge payload per layer per row, each way** (new exchanges, on a second
`PeerExchange` with q / candidate / partial flags per parity and lane, sized
at decode rows):

| family | q | partial + LSE | candidates (indexer layers) | today's all-reduce (kept) | context layers |
|---|---:|---:|---:|---:|---|
| GLM 5.3 | 36,864 | 32,896 | 16,384 (21) | 12,288 | 78 |
| V4 Flash | 32,768 | 32,896 | 4,096 (21) | 8,192 | 21 C4 |
| V4 Pro | 65,536 | 65,792 | 8,192 (30) | 14,336 | 30 C4 |
| GLM Flash | 32,768 | 32,896 | 4,096 (11) | 8,192 | 11 MLA |

**Why the merge doesn't fuse into the existing all-reduce.** The merge can
fold into the o_proj all-reduce by exchanging only the LSEs (256 B per row).
Each GPU would then scale its own all-head partial and run `W_UV + o_proj`
for all 64 heads. That replicates `o_proj` and `W_UV`: GLM 5.3 reads about
55 MiB more per layer per GPU, roughly 2.7 ms per step and +4.2 GiB per GPU.
The 33 KB partial push costs about 1 us. **Rejected.**

**Sink and normalisation.**
- **Partials:** each GPU's partial is normalized over its own tokens, with its
  base-2 LSE, as the split kernels already produce.
- **Combine:** `out = (2^{l0-m} o0 + 2^{l1-m} o1) / (2^{l0-m} + 2^{l1-m}
  [+ 2^{sink-m}])`.
- **V4's per-head sink** enters only in the final combine, so it counts
  exactly once. Both partials run with `has_sink` false. GLM 5.3 and GLM
  Flash have no sink.
- **Empty shard:** a GPU with no selected token for a row (the first page
  sits on GPU0) must emit `lse = -inf` and `out = 0`, never NaN. This is a
  kernel test.

**Ordering.** The decode schedule keeps rank 1 queued one unit ahead and adds
three matched push/wait pairs per indexer layer and two per shared layer. It
drains under the `order::check` port for all 78 GLM 5.3 layers, both with
rank 1 one unit ahead and in lockstep. The engine PR records it as a P3
fixture.

### 2. Prefill

**Exchanging is the wrong shape for prefill.** The decode route's traffic for
a 4,096-row chunk:

| family | q + partials + candidates per chunk | at 42 GB/s |
|---|---:|---:|
| GLM 5.3 | 23.7 GB | 0.56 s |
| V4 Flash (C4 layers) | 6.0 GB | 0.14 s |
| V4 Pro | 17.1 GB | 0.41 s |
| GLM Flash | 3.1 GB | 0.07 s |

That is a third of GLM 5.3's whole chunk time. It also competes with Spark
ingress on GPU0's PCIe root, which the probe showed quadruples small hops.

**Gather route (prefill default).** Each GPU assembles a contiguous
full-context view of the layer and runs today's head-split prefill kernels on
it unchanged. The arithmetic is identical to `heads`.
1. The producer and index producer write all chunk rows into the staging view
   at their logical positions, since the latent projection is replicated.
   A local copy then commits the owned rows to pool pages (2,048 × 788 B).
2. History: the own shard is a local D2D copy. The peer's shard is peer DMA,
   prefetched one layer ahead into the other parity slot.
3. Staging is two parity slots × the compiled extent × that layer's bytes per
   token. It is a fixed solver demand:

| family | bytes per token per layer | staging |
|---|---:|---:|
| GLM 5.3 (latent + index keys) | 788 | 1.54 GiB |
| V4 (C4 records + keys) | 179 | 0.35 GiB |
| GLM Flash (latent + pool keys, compact) | 561 | 1.10 GiB |

**Gather traffic per chunk** grows with history (all layers, peer half):

| family | at 128K | at 1M |
|---|---:|---:|
| GLM 5.3 | 4.0 GB (96 ms) | 32 GB (0.77 s) |
| V4 Flash | 0.25 GB | 2.0 GB |
| V4 Pro | 0.35 GB | 2.8 GB |
| GLM Flash | 0.4 GB | 3.2 GB |

It overlaps compute: a 1M-history GLM 5.3 chunk computes for several seconds.

**Crossover with the exchange route:** GLM 5.3 771K, GLM Flash 1.02M,
V4 Flash 3.2M, V4 Pro 6.4M tokens of history. Only GLM 5.3 past 771K would
prefer exchanging, at about 0.2 s per chunk.

**Recommendation: gather only.**
- It keeps prefill byte-exact with `heads`: goldens and exact-cache restores
  don't depend on where a chunk boundary falls.
- Prefill needs no new kernel.

**Short contexts.** There is no separate fallback: the gather route *is* the
head split on a gathered view. Decode/verify always exchange. A 6-row
verify would gather more cheaply only below about 1.1K tokens (GLM 5.3),
which is not worth a second decode route.

### 3. Paging and the prefix cache

**Interleave, not blocks.** The owner of a page is a function of its logical
page index alone: page j of every sequence lives on GPU (j mod 2).
- **Pages per family:** GLM 5.3 uses 64-token pages, V4 and GLM Flash
  256-token units.
- **Why interleave:**
  - Every row's selection spreads over both GPUs, so both work in every
    decode step at any context length.
  - Blocks would leave short sequences on GPU0 and make C1 single-GPU.
  - A prefill chunk splits 32/32 pages.
- **Why by logical index:** prefix sharing keeps positions, so a forked page
  or a tail copy stays at its index and therefore on its owner. That only
  works if ownership follows the logical index, not the sequence.

**Pool.**
- **Pages:** `RefPagePool` becomes two half pools under one id space: owner =
  `id & 1`, local index = `id >> 1`. `alloc_for(positions)` hands page j a
  free id of parity j. Admission checks both free counts.
- **Balance:** a sequence of n pages takes ⌈n/2⌉ pages on GPU0, so imbalance
  is at most one page per live sequence or retained snapshot. At 2M, GLM 5.3
  has 32,768 pages and V4 8,192 units; 16 sequences cost 16 pages.
- **Per-step work balance:** this depends on where DSA's picks land. Each
  step logs the split of selected counts, and the gate reports p50/p99.

**Prefix marks and snapshots.**
- **GLM 5.3** has no marks: pages are the whole state.
- **V4 and GLM Flash** keep marks per GPU, as under the head split today.
  V4's window rings and carry stay replicated. GLM Flash's KDA state is
  head-partitioned, and its compact tails are replicated.
- **Fork and restore** are unchanged:
  - full pages are shared by reference;
  - `copy_rows` runs on the owner's stream (the tail page keeps its index);
  - the length is set on both GPUs.

**Host tier (currently off under the head split) can come on.**
- **What changes:** each page lives once, so a snapshot is stored once.
- **Interface:** `PrefixFamily` gains `page_device(page)` and per-device
  segments. Replicated segments (V4's C128 pages and marks) are stored from
  GPU0 and restored to both GPUs, and the copy engine batches per device.

**Exact-cache byte-exactness.**
- **Prefill:** restored and fresh prefill both use the gather route, so they
  match byte for byte, as `heads` does today.
- **Decode:** after a restore, decode uses the same pages, the same selection
  and a fixed combine order (GPU0's partial, then GPU1's). Per-GPU split
  counts depend only on the row bucket.
- **The gate:** compares `context` against itself (fresh vs restored, odd
  frontier, host tier), as the existing check does. `context` is not
  byte-equal to `heads` in decode.

### 4. Numerics

What changes, and what doesn't:
- **The DSA selection doesn't change** (section 1).
- **Prefill doesn't change** (gather route).
- **The decode/verify attention output changes.** Each head was a merge of
  the kernel's contiguous splits of the selected list. It becomes a merge of
  two owner-partitioned shards, each merged from its own splits, with one
  more BF16 rounding of the partial on the wire.
  - This is the same rounding class as today's in-kernel BF16 split partials.
  - FP32 partials (`fp32_partials` exists in the AOT) double the payload to
    65,664 B per row.
- **V4's sink** moves from the split merge into the final combine.

**Fidelity gate per family:**

| family | golden (prefill-scored) | decode-shaped quick tier at max | other |
|---|---|---|---|
| GLM 5.3 EXL3 K4 | NLL byte-exact vs `heads` (split golden 2.4686) | KL / top-1 within `heads`' run-to-run envelope | `CUTEAFD_CONTEXT_CHECK` selection equal on the golden's decode steps; verify-by-replay exact |
| V4 Flash, Pro EXL3 K2 | Flash golden byte-exact vs `heads` | Pro quick KL no worse than `heads` (0.0596 max) | as GLM 5.3 |
| GLM Flash EXL3 K3.25 | NLL vs `heads` + compact: byte-exact when `heads` also runs compact on one GPU | teacher-forced decode KL ≤ `heads` + 0.002 | as GLM 5.3 |

Measure BF16 partials first, and FP32 only if the quick tier misses.

### 5. Layer ownership (`attention=layers`, for C > 1)

**Modes.**
- **Mode:** P3's `Whole{gpu, Split}` on every MLA/DSA/CSA layer. Ownership
  alternates by colocate group:
  - GLM 5.3: groups `{0}`, `{1}`, then the four-layer indexer groups,
    21 groups in all;
  - V4: per layer;
  - GLM Flash: its 11 DSA layers (P7's "mixed" plan).
- **Transition:** each layer's transition is section 3's single whole layer
  between split layers:
  - the owner's attention (unsplit `glm_*` / `dsv4f_*` programs);
  - an `AfterAttention` broadcast of the residual (12,288 B per row for
    GLM 5.3, `[4,H]` + FP32 for mHC families);
  - then today's split FFN all-reduce.

  Spark exchange, the router and the TP2 halves stay where they are (GPU0,
  P4's fused combine).
- **Why not `Owner` FFN:** it would move the Spark exchange to GPU1 on odd
  groups (a second NIC intake). Not planned.
- **KV and weights:** KV lives on the owner. The replicated attention
  operands and indexer weights exist once.

**Lanes.**
- At C1 there is one lane, and the peer idles through each owned attention.
  So `layers` gives up the attention share of the measured head-split gain.
- At C > 1 the decode batch splits into two lanes offset by one group: lane
  A's attention on GPU(k mod 2) runs while lane B's group k−1 runs on the
  other GPU, then both lanes' FFN all-reduces.
- This is a two-lane decode executor (P7-style, beside the prefill lanes).
  It also pipelines Spark waves between lanes.

**Ordering, checked.** Owner-FFN groups with two lanes drain under the
`order::check` port. With Split FFN, the schedule drains only when both GPUs
queue the two lanes' FFN all-reduces in the same lane order per time slot.
Swapping them on GPU1 deadlocks (gpu0 at lane A group 1, gpu1 at lane B
group 0). The executor records every lane interleaving and asserts
`order::check`.

**When `layers` beats `context`.**
- `layers` adds no per-row peer bytes; `context` adds about 70 KB per row
  per layer (GLM 5.3).
- By arithmetic they cost the same at about 15-19 verify rows, roughly C3.
  Above that, `layers` wins **if** the two-lane executor exists.
- `layers` also frees slightly more memory: no staging, and V4's window state
  is not replicated.
- `context` wins C1 and long contexts: it halves the replicated indexer scan
  per GPU, which `layers` doesn't.

### 6. Memory per family at max

**Basis.**
- P2 planner at 94.97 GiB (92.97 capacity), 2M pool unless noted.
- `context` charges staging and the context exchange.
- `layers` charges P3 hop slots (4 lanes × 2 × 4,096 rows) and drops the
  replicated operands.
- The caveats are P2's:
  - graphs and workspaces are partly calibrated, and the 64 MiB ledger gate
    has not run on these items;
  - staging and exchange are new demands, so no runtime ledger covers them
    yet;
  - the V4 rows come from P4's 95.5 GiB table, adjusted.

| family (max) | `heads` (today) | `context` | `layers` | extra RTX TP2 layers (`heads` → `context` / `layers`) |
|---|---|---|---|---|
| GLM 5.3 EXL3 K4, 2 RTX + 6 | pool 1.29M (2M needs 105.4 GiB latent per GPU); 93.0 / 82.0 GiB, 64.9 GiB latent per GPU | 2M: 82.3 / 71.3 GiB (52.7 latent + 1.5 staging per GPU; frees 52.7 GiB per GPU against `heads` at 2M); max pool 2.52M | 2M: 77.8 / 72.1 GiB (25,592 / 28,348 B per token; the latent exists once); max 2.73M | none: no local GLM 5.3 expert backend (with one: 4 / 6 half-layer pairs) |
| GLM 5.3 NVFP4, 2 RTX + 6 | pool 1.13M; 93.0 / 82.0 | 2M: 90.4 / 79.4; max 2.20M | 2M: 85.9 / 80.2; max 2.40M | none (1 / 2) |
| V4 Flash, 2 RTX + 4, P4 TP2 | 2M, 36 TP2 pairs, 7.87 GiB records per GPU, GPU0 slack 0.07 | frees 3.68 (C4) − 0.35 staging | frees 4.96 − 0.50 hop slots | 36 → 38 / 38 |
| V4 Pro EXL3 K2, 2 RTX + 6, P4 TP2 | 2M, 10 pairs, 11.16 GiB per GPU | frees 5.25 − 0.35 | frees 6.83 − 0.88 | 10 → 11 / 12 |
| GLM Flash K3.25, 2 RTX + 4 | 2M: 41.8 / 37.8, 23.05 GiB MLA records + keys per GPU | compact, half the units: frees 15.9 per GPU | compact on owners: frees 17.1 / 16.0 | P6 TP2: 37 of 42 → 42 / 42 |
| GLM Flash K3.25, 2 RTX, Spark-free | 2M with TP2: 98.6 / 99.4, no fit | 82.7 / 83.5, fits | 89.6 / 85.3 (section 4's mixed), fits | all experts on RTX |

### 7. Cost against today's head split

**C1 (verify rounds of 4-6 rows, short context ≤ 32K).**
- **`context`'s exposed cost:**
  - the q push on non-indexer layers (on indexer layers it hides under the
    replicated indexer);
  - per-layer exchange latency and the combine (about 3 us);
  - the candidate merge;
  - minus the extra TP2 pairs, at V4.1's measured remote 789 us vs local
    TP2 450 us per layer.
- **`layers`' cost:**
  - the attention share (about 80%) of each family's measured head-split
    gain;
  - plus the broadcasts;
  - minus the extra pairs.

| family | C1 round under `heads` | Δ round (Δ C1), `context` | Δ round (Δ C1), `layers` | measured head-split gain |
|---|---|---:|---:|---|
| GLM 5.3 | ~50 ms (4-row verify 50.6) | +1.2 ms (−2.4%) | +3.0 ms (−5.9%) | verify 1 / 4 / 16 rows −2.6 / −3.2 / −3.2 ms (7e27a769) |
| V4 Flash | 13.2 ms | −0.4 ms (+3.0%) | +1.1 ms (−8.1%) | 14.6 → 13.2 ms (5560ed00) |
| V4 Pro | 28.7 ms | −0.3 ms (+0.9%) | +3.6 ms (−12.6%) | 32.5 → 28.7 ms |
| GLM Flash | ~13 ms | +0.15 ms (−1.1%) | +0.25 ms (−1.9%) | C1 159 → 168 (9380e8b8) |

**C8 (about 40 verify rows).**
- **`context`:**
  - added cost: GLM 5.3 +3.7 ms (hidden) to +6.4 ms (unhidden), V4 Flash
    +0.3 to +1.7, Pro +0.6 to +4.4, GLM Flash +0.2 to +0.9;
  - against per-stream rounds of roughly 150 / 40 / 90 / 70 ms (rc3 C8
    per-stream rates × ~2.5 tokens per round), that is about −1 to −5%;
  - before V4's +2 pairs.
- **`layers` on one lane:** gives up about the same milliseconds as at C1,
  since the head-split gain barely grows with rows (GLM 5.3: 2.6 / 3.2 /
  3.2 ms at 1 / 4 / 16 rows). That is about −2 to −4%.
- **`layers` on two lanes:** about equal to `heads` or better: no
  per-row exchange, and both GPUs busy.

**Long context.**
- **Indexer:** under `heads` and `layers`, one GPU scans every key for every
  index layer. `context` halves the scan per GPU. Measured: 83 us per full
  layer at 32K for 1 row (315b8212). If that scales linearly, GLM 5.3 at
  256K spends about 14 ms per decode step on the scan (21 layers) under
  `heads`, against about 7 ms under `context`.
- **Prefill:** 8K is byte-identical to `heads` plus the gather (a few MB per
  layer). At 128K the gather adds about 96 ms per GLM 5.3 chunk, overlapped
  with compute.

**Expected per-family default** (TJ's rule: `context` wherever its C1 ≥
`layers`' C1 at 2M on max):

| family | expected default | basis |
|---|---|---|
| GLM 5.3 | `context` | about 3.5 points ahead of `layers` at short context, more at long |
| V4 Flash/Pro | `context` | 11-14 points ahead |
| GLM Flash | `context`, borderline | 0.8 points ahead, within noise |

`layers` is expected to win C8/C16 only once two-lane decode exists.

**Uncertain:**
- Peer bandwidth while Spark replies saturate GPU0's ingress: the probe saw
  4x on small hops. This is the largest risk to `context`'s C1.
- How much the q and partial pushes hide behind the indexer and the
  peer-heads-first order.
- Selection balance between the GPUs.
- The indexer's scaling with context.
- GLM Flash's value of a local TP2 layer (P6 hasn't measured it).
- P2 and P4 numbers are from unmerged branches.

### 8. Migration

**Ordered PRs.** GLM 5.3 first, then V4, then GLM Flash DSA. V4.1 stays on
its 20/20 ranges. Each family's default comes from its own decision gate.

| # | PR | size | after | gate |
|---|---|---|---|---|
| K0 | Solver and launcher. `AttentionPlacement::{Heads, Context, Layers}` in `PlacementRequest`; `LayerMode::ContextSplit` (residual transitions as `HeadSplit`); `KvDemand.unit_bytes_context`; staging and context-exchange fixed demands; `ExecutorModes` stays plan-only. `cuteafd plan --attention-placement`, `ATTENTION_PLACEMENT=auto\|context\|layers\|heads` in run-family (`auto` = the family's recorded default, `heads` until decided). The card records the choice. | S | P3, P2 | planner fixtures reproduce section 6 per family; cargo/script tests by failing id |
| K1 | Fork kernels (sparkinfer-glmrt master, then pin): scored index top-k (GLM, GLM Flash pools, V4 C4); `dsa_candidate_merge` (2k → k by (score, logical), emits the compacted local slot list); sparse MLA decode `partial` route (head range, partial + LSE, no sink) for 656 / 528 / 584 records (V4's "584" is a planar 576 B payload row plus an 8 B per-row footer, in 576-aligned padded pages: C4 37,440 B, SWA 149,760 B; indexer keys keep planar scales); `lse_combine2` (optional sink); paged → contiguous staging gather of whole pages, footers and padding included, byte-exact | M | — | SM120 + SM121 tests vs an FP64 reference: merged selection bit-equal to single-GPU top-k on random and all-ties inputs; empty shard gives −inf / 0; combine matches the 2-split merge |
| K2 | Shared parts: `shared/peer_split/context.rs` (`ContextExchange`, page owner map), `RefPagePool` parity half-pools, `PrefixFamily::page_device`, per-device host-tier copies; `order::check` fixtures for the context decode and the two-lane `layers` schedules | M | K0 | prefix-cache tests (fork, tail copy on owner, eviction balance, host round trip); order fixtures |
| K3 | GLM 5.3 `context` (needs P9's solver port, pulled ahead of P7: it depends only on P1/P3): attention path of section 1, gather prefill, single-owner prefix pages, host tier on | L | K1, K2, P9a | section 4's GLM 5.3 row; exact-prefix byte-exact incl. host tier; planner == runtime; the 2M pool admits at max; quick A/B at max: C1 ≥ `heads` C1 (at its 1.29M pool), C8, 8K prefill within 3%; one 256K-context C1 decode for the indexer claim |
| K4 | GLM 5.3 `layers`: per-layer executor (`Whole{g, Split}` by indexer group, `AfterAttention` hops via `HopLink`), two-lane decode/verify | L | K2, P9a | quick fidelity; order fixtures per interleaving; A/B at max at 2M |
| D-GLM | **Decision:** C1 `context` vs `layers` at 2M on max (3 interleaved sessions; 6 if within 1%), plus C8/C16 for both. Default per TJ's rule, recorded in `docs/models/glm5.md` with the numbers. | — | K3, K4 | — |
| K5 | V4 Flash/Pro `context` on C4 layers (C128 and window layers stay `HeadSplit`), sink in the combine, with P4 TP2 (+2 / +1 pairs) | M | K3, P4 | section 4's V4 rows; exact cache; A/B at max at 2M |
| K6 | V4 `layers` (`Whole{g, Split}` alternating, P4's fused combine on split FFNs) on S4c/P7's per-layer executor | M | K5, P7 | as K4 |
| D-V4 | **Decision**, as D-GLM | — | K5, K6 | — |
| K7 | GLM Flash DSA `context` (11 MLA layers, compact cache on two GPUs); `layers` is P7's mixed mode | M | K3, P7 | section 4's GLM Flash row; D-GLMF on the EXL3 K3.25 max card and the Spark-free card |

**Fork DCP audit (K1, 2026-10-11).** The fork's `b12x/comm/pcie` decode
context parallel stack covers less than it seemed:
- **`pcie_dcp_topk`** only transports rank-major score/index planes. It
  has no selection and no tie policy, so `dsa_candidate_merge` stays a K1
  kernel.
- **`pcie_dcp_a2a`** fuses an exchange with an LSE reduce-scatter over
  heads. It handles BF16/FP16 normalized partials, but has no sink, sums
  the local rank first and then the peers cyclically (not a fixed
  GPU0→GPU1 order), and needs 64 fully resident SMs. K2 may use it for
  no-sink paths if its numerics pass, and if graph/IPC lifetime and
  occupancy measure well next to decode.
- **`all_gather_heads` / `all_gather_pair`** move heads and pairs, not
  paged KV, so the K1 gather stays.
- **The sparse MLA split merge** already does base-2 LSE with the sink
  added once, and K1's partial route reuses it.
- Choice per piece is by exactness first, then measured speed and SM
  occupancy (TJ: reuse is not a goal).

**Interactions.**
- **P4 TP2.** The solver's pool step charges `context`/`layers` KV, and the
  freed bytes go to TP2 pairs in step 5. Both modes leave the residual
  replicated before the FFN, so TP2 halves stay in the existing FFN slot.
- **Memory lever (solver step 7).** It may flip `heads` → `context` →
  `layers` per kind, but only under `auto`. An explicit
  `ATTENTION_PLACEMENT` is strict.
- **Draft policy.** `context` adds peer bytes per verify row: about 5.4 MB
  per row per step for GLM 5.3 (q + partial over 78 layers).
  - `Placement` exports `peer_row_bytes`, and D4's bindings add it as a
    `Peer` resource class (`bytes/bandwidth`, its own Huber fit). Otherwise
    the allocator underprices wide verifies.
  - `layers` with two lanes changes the lane count, not the per-row price.

### Deferred experiment: alternating vs one-cutover `layers` ownership (TJ, 2026-10-11)

**Default:** `layers` uses one cutover point, k, chosen for byte balance (K0).
Alternating ownership is a later experiment, possibly after v3. It will be
measured, not assumed.

**Where alternating could win.** Only with two-lane decode (C > 1), and
mostly at long context, where attention dominates a step.
- One cutover pipelines too. The two lanes sit half a forward pass apart:
  lane A's layer i runs on GPU0 while lane B's layer i + n/2 runs on GPU1.
- The difference is balance within each time slot. Alternating pairs
  adjacent layers (similar cost). One cutover pairs layer i with layer
  i + n/2, which can be a different kind, and attention cost grows with
  context. So bubbles from slot imbalance grow with context too.
- That is why a win, if any, appears only at high context. The experiment
  needs C8/C16 at short, 128K and 256K+ contexts, not only C1 at short
  context.

**Costs.**
- An ownership change needs a residual hop only where the FFN is not split.
- With TP2-split FFN, every layer broadcasts anyway (K0: 43/61), so
  alternating adds no transfers there.
- With Spark or owner FFN, each change is a hop of hidden × rows in BF16:
  a few µs at decode rows over P2P. Measure it before the full experiment.

**Per family.** Groups that share index or source state never split across
GPUs (`colocate`), so alternation granularity is the group:
- **V4.1:** a compressed layer's reindex reads its source group's full
  CSA. Alternation would go group by group. Window-only (SWA) layers could
  alternate freely, but their attention is small, so moving them buys
  little balance. Expected: little gain; V4.1 keeps its 20/20 ranges unless
  this measures a win.
- **GLM 5.3:** shared-indexer layers colocate with their full-indexer
  layer, so alternation would go in indexer groups.
- **V4:** C4 and C128 interleave, so a slot pairs C4 with C4 only when the
  layer offset is even. One cutover at an even offset may already balance.
- **GLM Flash:** 11 MLA layers between KDA layers.

**When.** After two-lane decode exists (K4/K6/P7). The comparison is
one-cutover vs alternating on the same build, at 2M, C1/C8/C16 × short/128K/
256K, on V4 max and GLM 5.3 max. A family switches only if alternating wins
C8/C16 at long context and loses no C1.

### 9. Open questions for TJ (with recommendations)

**Decided (TJ, 2026-10-11): all eight as recommended.**
- prefill is gather only;
- BF16 partials on the wire;
- V4 splits its C4 layers only;
- `layers` is single-lane first;
- the selection check is off in serving and on in the gates;
- the host tier is on under `context`;
- a GLM Flash tie goes to `context` after 6 sessions;
- add a 256K GLM 5.3 C1 card.

K1 starts now; K0 starts after P2 merges.
1. **Prefill route.** Recommend gather only: byte-exact with `heads`, no new
   prefill kernel, exact cache independent of chunk boundaries. It costs
   GLM 5.3 about 0.2 s per chunk past 771K tokens of history.
2. **Partial dtype on the wire.** Recommend BF16, the in-kernel split
   partials' own class. FP32 (2x payload) only if the quick tier misses.
3. **V4 scope.** Recommend `context` on C4 layers only: C128 and window
   layers stay head-split and replicated, 0.26 GiB per GPU at 2M.
4. **Two-lane decode for `layers`.** TJ's default rule is C1-only, so
   recommend building `layers` single-lane first (enough for the decision),
   then two-lane decode gated on C8/C16. It also pipelines Spark waves for
   every mode.
5. **Selection check.** Recommend `CUTEAFD_CONTEXT_CHECK` off in serving and
   on in K3/K5/K7's gates (exact equality with a full single-GPU top-k).
6. **Host tier under `context`.** Recommend turning it on with the
   exact-cache host gate. Pages live once, so it is cheaper than the replicas
   `heads` would need.
7. **GLM Flash tie.** The arithmetic puts the two options within 1%.
   Recommend 6 interleaved sessions if within 1%, and `context` on a tie (the
   ≥ in TJ's rule).
8. **Long-context card.** Recommend adding one 256K-context C1 decode to
   GLM 5.3's card set: it is where `context` differs most from both
   alternatives, and the agentic floor sits there.

## v3 API gateway and sessions (design, 2026-10-09)

TJ: Claude Code and Codex CLI must use cuteafd directly as drop-in
clients, for real users and with no LiteLLM in the path. The Realtime API is
also required. The APIs are the deliverable.

Work happens on `work/api-gateway` (Opus lead, Sol components). Phase A runs
against an upstream API with no GPUs. Phase B adds the engine backend and
the session hooks. Phase C adds Realtime audio (transcription and TTS).

### Layering

```text
 Anthropic Messages  /v1/messages, count_tokens ─┐
 OpenAI Responses    /v1/responses (+ get, input_items, ...) ─┼─ TurnRequest ─► Gateway::run ─► Backend
 OpenAI Realtime     /v1/realtime (WebSocket) ─┘    ▲              (hosted tools,   ├ Upstream (HTTP, now)
 OpenAI Chat         /v1/chat/completions (unchanged)│              aliasing)       └ Engine   (phase B)
                                              SessionStore (items, ops, snapshots)
```

- **Front ends** (`cuteafd-api/src/gateway/{anthropic,responses,realtime}`)
  parse their wire format into one `TurnRequest`:
  - the system prompt;
  - `Item` history: message, reasoning, tool call and result, and server-tool
    call and result;
  - tools and tool choice;
  - sampling and reasoning controls;
  - hosted web search;
  - output modalities.

  Each front end renders the `TurnEvent` stream in its own event sequence.
  The events are reasoning, text and tool-call deltas with stable indexes,
  server-tool events, cumulative usage, and a typed stop reason. Errors are
  one `GatewayError`, rendered in Anthropic, OpenAI or Realtime shape.
- **`Gateway::run`** resolves the model alias and runs hosted tools. Web search
  goes to the backend as a plain `web_search` function. The driver executes
  each call through a `SearchProvider` and continues the turn, so no backend
  needs to know about search.
- **`trait Backend`** has four methods: `start(turn) -> stream`,
  `count_tokens`, `models` and `capabilities`. Dropping the stream cancels the
  turn.
  - **`Upstream`** is a test/dev seam, not a shipped way to run cuteafd. It
    is one generic OpenAI-chat or Anthropic-compatible client, configured by
    base URL, key env var, model, flavor and an explicit capability set.
    Generic `/models` metadata supplements served-model capabilities. Product
    code has no provider names or capability tables; choices live in test
    tooling under `scripts/gateway/`. `json_schema` and `strict_tools` pass
    through when enabled, otherwise return typed Unsupported naming the field.
    Strict Chat tools may use an explicitly configured absolute endpoint path;
    rejection is propagated, never retried without strictness. PCM16 realtime
    input is wrapped as mono 24 kHz WAV for Chat `input_audio`. These are
    backend capabilities, not frontend restrictions.
  - **`Engine`** (phase B) feeds the scheduler directly. It replaces today's
    chat handler internals without changing that route's behaviour.
- `/v1/chat/completions` stays the engine's existing path, byte-identical,
  until the engine backend has a parity gate. The gateway mounts beside it.

### Session layer

`SessionStore` holds two kinds of state:
- **Live sessions:** Realtime connections, and phase-B explicit sessions.
- **Immutable response snapshots:** each a chain of `(parent, new items)`.
  `previous_response_id` continues from any snapshot in O(new items), which
  is already a virtual fork.

Every operation is a typed `SessionOp`. Each returns an `EngineEffect`, the
honest cost the engine will pay:

| Operation | Phase A | Engine needs (phase B) | Cost |
|---|---|---|---|
| append | real | extend the cached prefix | `PrefixKept`: prefill only the new items |
| insert / edit / delete | real | invalidate KV from the first changed item | `RecomputeFrom(index)`: positions and attention depend on every earlier token |
| truncate (Realtime) | real | drop KV after the cut | `RecomputeFrom(item)`; cheap when it is the last item, the barge-in case |
| cancel | real | the scheduler's cancel path (drop the request) | — |
| fork | real (copy) | prefix-cache mark at the fork point + `RefPagePool` page sharing (`PrefixCache` fork/restore, `PrefixFamily::capture/restore`) | `SharedPrefix`: no recompute; copy-on-write tail page |
| steer: inject | stub | a scheduler hook between decode steps or prefill chunks: append tokens to the running sequence | prefill the injected tokens; no recompute |
| steer: replace | stub | cancel, then recompute from the edited item, then resume | `RecomputeFrom` |
| compact | stub | summarize with a side turn (forked session), then replace items | a new prefix: full prefill of the summary |
| splice | stub | general case = edit; recompute from the splice start | `RecomputeFrom(from)`; only an exact token-identical prefix survives |
| KV pin / evict / mark | stub | `PrefixCache` retention: pin = refcount hold; evict = release; mark = named capture point | none |

Mid-sequence splicing cannot reuse KV after the edit point: rows after it
were computed against the old tokens. Fork at a prefix is cheap because the
pages are shared and refcounted. Steering mid-prefill needs the scheduler to
accept appended tokens on a sequence that is still prefilling chunked work.
That is the one new engine primitive phase B adds.

### Front ends: what each client needs

- **Claude Code** (`ANTHROPIC_BASE_URL=http://host:port`, key in
  `ANTHROPIC_API_KEY` (x-api-key) or `ANTHROPIC_AUTH_TOKEN` (Bearer)):
  - routes: `POST /v1/messages?beta=true` (stream) and
    `/v1/messages/count_tokens`, plus any others the live capture shows;
  - beta and cache-control headers are accepted and ignored;
  - thinking signatures are synthesized as opaque values and accepted on
    input;
  - server web search (`web_search_*`) runs gateway-side.

  The model ids come from `ANTHROPIC_MODEL`,
  `ANTHROPIC_DEFAULT_{OPUS,SONNET,HAIKU}_MODEL`,
  `ANTHROPIC_SMALL_FAST_MODEL` and `CLAUDE_CODE_SUBAGENT_MODEL`. With
  `--accept-any-model`, any of them maps to the served model.
- **Codex CLI** (`model_providers.cuteafd` with `base_url =
  "http://host:port/v1"`, `wire_api = "responses"`, `env_key`):
  - `POST /v1/responses` (stream) with `store:false`;
  - reasoning `encrypted_content` round-tripped (an opaque token decoding to
    the reasoning text);
  - function, custom (freeform `apply_patch`), `local_shell` and
    `web_search` tools;
  - `previous_response_id` from the snapshot store;
  - a model listing, if Codex fetches one.
- **Realtime** (`GET /v1/realtime?model=` WebSocket, GA and beta event
  names):
  - each connection is a live session;
  - `conversation.item.create`, `delete` and `truncate` are `SessionOp`s;
  - `response.create` runs a turn over the session items (or out-of-band
    input), and `response.cancel` drops it.

  Audio input is committed (manually or by server VAD) as an `input_audio`
  part:
  - **an audio-capable backend** handles it: our MiMo audio path through the
    engine backend in phase B, or an upstream with audio input;
  - **otherwise** the response fails with a clear capability error.

  Spoken output needs TTS we don't have. Audio output modality is refused
  per spec, and a `Synthesizer` seam waits for phase C, as does a
  `Transcriber` seam for input transcription.
- **Home Assistant:**
  - Assist's own voice pipeline speaks Wyoming (STT/TTS/wake word) and a
    conversation agent.
  - The first-party OpenAI integration hardcodes api.openai.com. The
    llama.cpp integration (2026.8) and the community
    `ha-openai-compatible` integration take a base URL; they use chat
    completions or Responses, `/v1/models`, `audio/transcriptions` and
    `audio/speech`.
  - None speaks Realtime today. Cuteafd connects as a conversation agent
    through Responses or chat completions now. A Realtime path for
    multi-microphone rooms means a small bridge, Wyoming satellites →
    Realtime, or `audio/transcriptions` + `audio/speech` once phase C has
    STT/TTS.

### Model aliasing

`ModelMap` holds:
- the served id;
- explicit aliases (`PATTERN=MODEL`, with a trailing `*` matching a prefix);
- `--accept-any-model`;
- extra advertised ids for listings.

Responses echo the id the client requested, so CLIs that check it stay happy.
`GET /v1/models` returns OpenAI and Anthropic fields side by side, and
`GET /v1/models/{id}` resolves through the map.

### Hosted web search

The client's server tool (Anthropic `web_search_*`, Responses `web_search`)
becomes a gateway-executed search. Providers: Exa (keyed) and self-hosted
SearXNG (no key). Results return in each protocol's own server-tool shapes,
and `usage.server_tool_use.web_search_requests` is counted.

### Testing without GPUs

Upstream providers exist only to test the gateway without GPUs. User docs
describe the front ends, aliasing, official names, search and client setup,
not upstreams.

- `cuteafd gateway --upstream-url ... --model ...` serves every front end over
  an upstream.
- `--record DIR` writes sanitized fixtures: the client exchange plus every
  upstream and search exchange beneath it.
- Replay tests serve recorded upstream traffic from an in-process fake, offline
  in `cargo test`.
- A secrets scan fails on key-shaped strings, home paths or emails in any
  fixture.

### Phase B: engine backend

**Decisions (TJ, 2026-10-10: "Sure go ahead").** Phase B starts now on
work/p0:
- B1 (engine backend) goes first; the gateway routes mount in the serving
  coordinator.
- `--official-model-names` is on by default.
- Per-model `json_schema` is advertised only after a probe passes.
- B1's gate includes a live Codex WebSocket check.
- Usage hooks (usage steps 2 and 3) run alongside, plus the C1 overhead A/B
  that decides whether `--usage` defaults to on.
- Then B2 and B5, the usage dashboard (steps 7 and 8), then B3, B4 and TTS.


1. Add an `Engine` backend that builds the family's prompt from `TurnRequest`
   (reusing each family's chat template and tool syntax) and submits
   `NativeRequest`. Gate it on golden NLL / byte-exact replies against the
   chat path.
   **Done (work/api-b1, 2026-10-10):**
   - `openai/engine.rs` maps a turn onto the chat body, which runs through
     the chat route's own build/submit pipeline. Each source assistant turn
     becomes one chat assistant message.
   - System items follow each template's placement rule, probed at startup.
     For Qwen 3.8, leading system items merge and later ones become user
     turns.
   - Every serving family mounts the gateway by default (`--gateway off`
     drops it). Official model names are on by default.
   - Gates on Qwen 3.8 EXL3 1 RTX passed:
     - identical prompt hashes and outputs across Chat, Messages and
       Responses (plain, tool, image; thinking off and on);
     - Claude Code and Codex (Responses WebSocket) complete a read, edit
       and test task.
   - Follow-ups from the final review:
     - Fold choices: reasoning blocks of one turn concatenate with no
       separator, and text after a call is rendered before it (chat has no
       slot after calls).
     - Strict function tools are refused until the structured-output probe
       passes; the Agents SDK defaults to strict, so it needs the probe or a
       non-strict opt-out.
     - Images inside a `tool_result` on text-only models fail on the chat
       path's media guard instead of degrading to text.
     - The review's test gaps: Realtime turns through the engine backend,
       MiMo audio over Realtime, hosted search rounds end to end on
       hardware, and `count_tokens` for encoder-prepared media.
2. Add session-aware prefix-cache hooks: fork through `PrefixCache` marks,
   pin and evict by session, and `RecomputeFrom` mapped to token positions.
3. Add steer-inject between decode steps, then mid-prefill injection.
4. Add MiMo audio input under Realtime, plus a `Transcriber` from the same
   encoder.
5. Make `max_tokens=0` a real cache prewarm on the Engine backend: a
   prefill-only turn that publishes the prefix to the prefix cache and
   returns empty output. The Upstream backend keeps today's local empty
   answer.

### Phase C: voice

1. **TTS:** a `Synthesizer` behind Realtime audio output and
   `audio/speech`.
2. **Ephemeral client secrets: not needed for now** (TJ, 2026-10-10). The
   agent workspace is DSH embedded in our own dashboard, not a third-party
   browser app holding a key. `/v1/realtime/client_secrets` and
   `/v1/realtime/sessions` keep answering unsupported.
3. **WebRTC: not in v3** (TJ, 2026-10-10: "For v3 I think we will skip
   webrtc"). Realtime is WebSocket only. Every headless client works over it:
   openai-python, openai-node, Agents JS/Python, Pipecat and LiveKit.
   `POST /v1/realtime/calls` keeps answering unsupported. The cost of
   skipping: the official browser apps (`openai-realtime-console`,
   `openai-realtime-agents`) and Agents JS's browser WebRTC transport don't
   run unmodified. Adding it later needs an ICE/DTLS-SRTP stack (e.g.
   webrtc-rs), Opus encode/decode, and a data channel carrying the same event
   model.
4. **Home Assistant bridge:**
   - Wyoming satellites → Realtime.
   - HA `openai_stt_ha` speaks `?intent=transcription` and needs a real
     `Transcriber`.
   - `ha-openai-realtime` (Pipecat) needs its base URL exposed.

## v3 agent workspace: DSH embedded in the dashboard (research, 2026-10-10)

TJ: the workspace is DeepSeek Harness (DSH, MIT) embedded in the dashboard.
It uses the local model by default, and other models can still be registered
normally. Configuration lives in a mounted folder. The main connection is a
session-oriented WebSocket. All execution runs over SSH, with keys and hosts
registered in the UI and folder picking on those hosts. The early features
are editing any message (including replies and thinking) and steering while
a prefill or decode runs.

Read at `deepseek-harness` d7432673 (0.2.1-alpha.2; npm `latest` is
0.2.0-rc.2), `dsh-desktop` 03dcfa1, `awesome-dsh-plugin` dc8396d and
`@earendil-works/pi-ai` 1.1.0 (dist only). Clones are in
`~/.cache/cuteafd/builds/dsh-research/src/`. Paths below are relative to
`deepseek-harness/` unless marked `pi-ai:` or `ours:` (this repo).

### What DSH is

- **Runtime.** A Node host (Node `^22.19 || >=24`) composed of Cordis plugins
  from ordered YAML patches, plus a React SPA. Everything is a plugin,
  including the agent loop (`docs/architecture.md:9-13`). The SPA calls
  `POST /api/<ns>/<method>` and streams over one WebSocket,
  `/api/remote.mux` (`packages/api/gateway/src/stream-protocol.ts:7`). The
  Electron desktop and `dsh-desktop` (a community shell) are both this same
  host plus a window; `dsh-desktop` pins upstream commits and adds only
  plugins (`dsh-desktop/upstream.json`).
- **Agent loop.** A step is one model request plus the tools it calls. Each
  attempt derives the whole model history from the session log, freezes it,
  and streams it through one adapter call (`packages/core/agent-loop/src/agent.ts:688`).
  The loop's rule is "**model-visible means logged**": every request must be
  reconstructable from the append-only log (`docs/architecture.md:124`). DSH
  owns its transcript.
- **Models.** `llm-pi-ai` routes are pure config: `api`
  (`openai-completions`, `openai-responses` or `anthropic-messages`),
  `baseURL`, `models` and a credential reference (`packages/llm/llm-pi-ai/src/provider.ts:47`,
  `config.ts:91-160`). The Models settings page can add a custom provider,
  and settings persist into the profile's `cordis.patch.yml`
  (`packages/settings/settings/src/index.ts:377`).
- **Tools.** They run through the `ctx.fs`, `ctx.subprocess` and
  `ctx.sandbox` seams. The SSH family (`packages/ssh/*`) implements all three
  on one remote host.
- **State.** Everything lives under `$DSH_HOME` (default `~/.dsh`,
  `packages/util/home-paths/src/index.ts:124`): profiles and plugins,
  `.credentials.yaml`, `sessions/`, `storages/` and `attachments/`. Agent
  instructions use `$DSH_AGENTS_HOME` (default `~/.agents`).
- **Stability.** Upstream declares its public APIs pre-stable
  (`AGENTS.md:7`). Our plugins will break on bumps, so we pin an exact
  version.

### Findings per requirement

**a. Local model as default: config only, no code.** Add a `cuteafd`
pi-ai route and set `agent-default-model` to it. The base bundle ships
`deepseek-official/deepseek-flash` (`packages/bundle/base/cordis.patch.yml:85-89`,
`packages/core/agent-default-model/src/index.ts:24`). Other providers stay
registrable; only a duplicate route name fails. Two caveats:
- The route must carry a credential reference. Otherwise the post-login
  `initializeDefaultModel` can switch the default to `deepseek-account`
  (`packages/api/session-controller/src/index.ts:300`), and pi-ai's
  OpenAI-compatible client insists on a key anyway. Give the route our
  gateway key.
- **Today only `/v1/chat/completions` reaches the engine.** Messages,
  Responses and Realtime still run on the upstream test backend until
  phase B step 1 (ours: `README.md:346-348`). Stage 1 therefore uses
  `api: openai-completions`, which carries `reasoning_content`
  (pi-ai: `api/openai-completions.js:400`). Switch to `anthropic-messages`
  or the adapter in stage 6 once the Engine backend lands.

**b. Session-oriented main connection: the central question.** These are the
facts that decide it:
1. **DSH must keep the transcript.** The adapter seam is
   `stream(GenerateOptions) -> AsyncIterable<StreamChunk>`, carrying the
   full frozen `messages`, a `sessionId` and an `AbortSignal`. It ends in one
   terminal `finish` with "nothing afterward"
   (`packages/llm/llm/src/index.ts:208`, `types.ts:444-462,546`).
   - Retries re-derive the same step and must not duplicate items
     (`agent.ts:406,505`).
   - Compaction replaces history ranges (`packages/compaction/compaction-basic/src/region.ts:507`).
   - The system prompt is a logged node that can be replaced or cleared
     (`agent.ts:416`).
   - Any replacement starts a new request series (`agent.ts:631`).

   A server that owns the history and edits it can't push those edits back:
   DSH's next request would contradict them. **So our server holds a cache of
   DSH's history, never the truth.**
2. **Realtime as-is clashes.** The adapter would have to mirror the log into
   Realtime items. Each step it would diff the frozen messages against the
   server's item list, then issue `conversation.item.create`/`delete` and
   `session.update` (instructions) before `response.create`, mapping DSH
   message ids to item ids. That is workable, but it is the most code for
   the least gain:
   - Realtime has no reasoning events (ours: `gateway/realtime/response.rs:208`),
     so thinking would be lost.
   - Sessions expire after 1 h (ours: `gateway/realtime.rs:158`).
   - Every compaction, edit or retry becomes delete/insert churn.

   Realtime stays the voice API.
3. **Responses with `previous_response_id` fits as-is.** pi-ai already
   implements exactly this, for Codex only, as transport `websocket-cached`.
   It keeps one socket per session, and when the new input extends the last
   request plus its output, it sends `previous_response_id` and only the
   delta. Otherwise it sends everything (pi-ai:
   `api/openai-codex-responses.js:1131-1158,1175`). Snapshots are immutable,
   so retries and forks are free.
   - Our gateway already serves a Responses WebSocket with snapshot
     continuation (ours: `gateway/responses.rs:39,338`).
   - pi-ai refuses Codex transport on hand-declared routes
     (`packages/llm/llm-pi-ai/src/provider.ts:37-41`), and its plain
     `openai-responses` path is HTTP only. Reaching it needs our own adapter
     plugin, `dsh-llm-cuteafd` (about 300 lines: the same delta rule plus
     our events).
4. **Plain HTTP already reuses KV.** The engine prefix cache keeps the
   finished turn and matches the next full-history prompt by tokens. That
   holds only if each family's template re-renders earlier turns
   byte-identically. Templates that drop older reasoning break the match at
   the first assistant turn. Measure the per-step hit rate in the spike.

**Decision (recommended).** The main connection is a per-session WebSocket:
the Responses WebSocket with `previous_response_id` deltas, plus cuteafd
extension events on the same socket (`cuteafd.steer.inject`, later KV
hints). It is session oriented in the sense that matters: one socket per
DSH session, the server keeps the live sequence, input is delta-only on the
happy path, and the channel stays open mid-stream for steering. The history
contract stays DSH's. Edits need no `SessionOp`s on the wire: a non-matching
prefix sends the full input, and the prefix cache recomputes from the first
differing token, which is the same `RecomputeFrom` effect.

**c. Execution over SSH: seams exist, Web integration does not.**
- **What exists.** `dsh-ssh` launches the system `ssh` with an OpenSSH host
  alias that carries its own user, key and known-host settings. It sets
  `BatchMode=yes`, `StrictHostKeyChecking=yes` and `ForwardAgent=no`, and
  discards stderr (`packages/ssh/ssh/src/index.ts:33,299-309`).
  - It needs a pre-installed helper with a pinned hash.
  - Linux arm64 and x64 (glibc 2.28) helper builds embed Node, so Sparks
    need no Node (`packages/ssh/ssh-helper-runtime/README.md:40-41`).
  - There is one host per `SshConnection`. Per-host agent presets with
    `isolate` realms look feasible but are not a shipped feature.
- **What is missing** (upstream: "Web workspace views … need separate
  integration", `docs/subsystems/ssh.md:27`):
  - The folder picker reads the host's own disk through `node:fs` from
    `homedir()` (`packages/host/directory-picker-browse/src/index.ts:12,217`).
    It needs an SSH-backed `ctx.directoryPicker` provider; the seam is
    replaceable.
  - Workspace creation calls `stat`/`realpath` on the host
    (`packages/workspace/workspace/src/index.ts:237`). This needs a
    host-aware `WorkspaceRegistry` replacement, which is the main fork risk.
  - `@file` completion is local (`file-reference-local`).
  - `fs-ssh` has no `watch`, so live file-change views fail.
  - There is no key or host registration UI.
- **Verdict.** Plugins plus a probable small upstream PR for workspace
  identity. The `dsh-ssh-ops` plugin in the catalog is a separate ssh tool
  set, not an executor. Ignore it.

**d. Persistent config in one mounted folder: yes, with three extra
variables.** Set `DSH_HOME=/data/dsh`, `DSH_AGENTS_HOME=/data/agents` and
`HOME=/data/home`. `HOME` matters because OpenSSH reads
`~/.ssh/config`/`known_hosts`, which hold the generated host aliases.
`.credentials.yaml` is plaintext YAML at 0600 in a 0700 directory
(`packages/credentials/credentials-local/src/index.ts:702`).

**e. Edit any message: plugin, no fork.**
- **DSH side.** Committed events are frozen. A plugin-owned event with a
  registered message projection (`packages/core/session/src/index.ts:1066`,
  `docs/subsystems/session.md:363`) replaces a message's content in place,
  including reasoning blocks, and keeps its identity. The loop then starts a
  new request series. Restore and fork need the same plugin loaded.
- **UI.**
  - The visible transcript reads append-origin events, not the projected
    surface (`docs/subsystems/session.md:358`). The edit UI must therefore
    render the projected text through a keyed `conversation.chat.node`
    renderer.
  - Upstream has an assistant-actions slot (`packages/client/ui-chat/src/client/contract/slots.ts:414`)
    but no user-actions slot. `dsh-webchatlike` patches one in and does
    edit-as-fork; its fork model is the prior art for "edit then branch".
- **Signatures.** pi-ai replay checks only the block count and type, so
  edited thinking would replay a stale signature
  (`packages/llm/llm-pi-ai/src/replay.ts:187`). The plugin must drop replay
  metadata on edited messages.
  - Real Anthropic and OpenAI upstreams then see unsigned or plain
    reasoning. Anthropic rejects modified signed thinking, so edits to an
    upstream model's thinking go out as unsigned or omitted thinking; say so
    in the UI.
  - Our engine has no constraint. Anthropic signatures are synthesized
    opaque values, accepted unverified on input (ours:
    `gateway/anthropic/render.rs:86`, `request.rs:78`).
  - **One gateway fix:** on Responses input, the text inside a cuteafd
    `encrypted_content` token wins over the visible reasoning text (ours:
    `gateway/responses/parse.rs:554-560`). A client that edits the summary
    and replays the old token silently gets the old thought. Prefer the
    visible text when both are present and differ, or reject the mismatch.
- **Our session layer** needs nothing new for edits over full-history
  requests (see b).

**f. Steering mid-prefill or mid-decode.**
- **What DSH does today.** `steer()` queues next-step input
  (`agent.ts:166`) that is claimed only after the current request and its
  tools finish (`agent.ts:361`). The composer already has busy-Enter
  steering (`packages/client/ui-conversation/src/submission-settings.ts:12`).
  The adapter has no injection concept. That is fine for agent turns, but
  it is not mid-generation.
- **Cheap version, no engine work.** Make steer cancel with `keepInbox`.
  DSH commits the delivered prefix as an `interrupted` assistant message
  (`agent.ts:465`), then the steer message starts the next request.
  - What it costs: recomputing the decoded tail. The prefix cache parks a
    cancelled prefill but does not keep a cancelled decode (ours:
    `rust/crates/cuteafd-engine/src/prefix/cache.rs:14`).
  - Engine fix: park a decode cancelled at a step boundary as a `Prompt`
    snapshot. This is small, and the same fix serves Realtime barge-in.
- **True version (phase B step 3, `Steer::Inject`, ours:
  `gateway/session.rs:60`, stubbed at `:190`).** The adapter sends
  `cuteafd.steer.inject` on the session socket. The engine appends the
  injected tokens at the next decode step or prefill chunk and acks with the
  split point.
  - The adapter then ends its stream at the split with a normal `finish`.
    DSH commits part 1, claims the steer as next-step input, and the next
    request matches the server's live sequence, so the server attaches to the
    already-running continuation instead of starting over.
  - The engine's token sequence must equal what the template renders for
    `…assistant(part 1) + user(steer)` as a closed assistant turn. That is a
    golden gate. Inject only at safe points: never inside a tool-call
    argument or before the first visible token.
  - Phase B's listed primitive covers the engine half. The missing half is
    the wire event, the split-and-adopt rule, and the byte-identity gate.

### Embedding as a styled facet

- **Where it runs: a sidecar container** on the coordinator host, from a
  pinned `@deepseek-ai/dsh` image. It binds 127.0.0.1:PORT, and the volume
  above is its only mount.
  - It never goes in the engine image (Node toolchain, separate lifecycle
    and crashes) and never in the browser alone (the browser-only WebWorker
    preview is not a supported launcher, `docs/architecture.md:45`).
  - `run.sh --agent` (or a separate `scripts/launch/agent.sh`) starts it.
- **Mount: a same-origin sub-path, framed.**
  - The coordinator reverse-proxies `/agent/app/*`, including the
    `api/remote.mux` WebSocket upgrade, to the sidecar with the prefix
    stripped. DSH supports this: relative Vite base, `--public-url …/ui/
    --trusted-host` (`packages/bundle/web-app/src/startup.ts:77`,
    `apps/web/vite.config.ts:168`).
  - The dashboard page `/agent` is our header plus a full-height
    same-origin iframe of `/agent/app/`.
  - The iframe isolates our global CSS (`header`, `*` and `body` rules in
    `cuteafd-ui.css`) from DSH's CSS modules and back. DSH sets no
    `X-Frame-Options`, and same origin passes its Origin fence. A direct
    mount would need a fork-sized CSS audit for no gain.
  - Add `{id:'agent', href:'/agent', label:'AGENT'}` to `PAGES` (ours:
    `assets/cuteafd-ui.js:53`).
- **Theme: a plugin, no fork.** A client plugin (`@cuteafd/dsh-facet`)
  calls `ctx.theme.register({id:'cuteafd', colorScheme:'dark', tokens})`,
  mapping our `--bg/--panel/--ink/--line` and accents onto `--dsw-*`
  (`packages/client/ui-theme/src/client/index.ts:311,344`). It replaces
  `ui-brand-official` with our mark and sets the preference at boot, because
  custom theme ids don't persist. The dashboard is dark only (ours:
  `cuteafd-ui.css:5`), so lock DSH to dark.
- **Bundle.** `dsh-web-frontend` is 5.7 MB unpacked (npm 0.2.0-rc.2,
  132 files), plus the per-plugin client bundles. The sidecar serves it, not
  our binary, so the console stays self-contained. There are no runtime CDN
  fetches (fonts are local). The cost is paid only on `/agent`.
- **Auth.** Rule kept: the console cookie never grants API access, and no
  exception is needed.
  1. The coordinator gates `/agent` and `/agent/app/*` with `ConsoleGate`.
     DSH does not authenticate its static assets itself.
  2. DSH has its own auth: a process-random launch `?token=` buys an HMAC
     cookie (`HttpOnly; SameSite=Strict; Path=/`, the name hashed per
     authority). See `packages/client/connection/src/browser-auth.ts:52,129,250`.
     - Our sidecar plugin disables `printUrl` (`packages/bundle/web-app/src/index.ts:67,279`)
       so the token never reaches `docker logs`. It writes the
       authenticated URL to a 0600 file on a tmpfs shared with the
       coordinator.
     - On an unlocked visit without a DSH cookie, the coordinator performs
       the token exchange server-side and relays the `Set-Cookie`, so the
       token never reaches the browser.
  3. The DSH host calls the gateway with its own key, stored in
     `.credentials.yaml` and resolved as the route's `apiKeyEnv` reference.
     The browser never calls `/v1`.
  4. Recommended: a second gateway key, `agent`, so the usage tracker
     attributes this traffic. `require_key` takes one key today (ours:
     `openai/auth.rs:76`).

### Security

- **SSH private keys.** `/data/home/.ssh/keys/<alias>`, 0600, inside a 0700
  tree owned by the sidecar uid. Pin `known_hosts` at registration, with
  the fingerprint shown and confirmed in the UI. The generated `ssh_config`
  sets `IdentitiesOnly yes`. Keys never enter git, images, settings YAML or
  logs; `dsh-ssh` already discards ssh stderr.
- **Readable by the model?** Not through normal tools when every execution
  seam is SSH: `fs-ssh` has no local fallback. Three holes need closing:
  - `plugin_manager` installs host code after one approval, and the
    standard preset includes it (`packages/bundle/web-app/presets/standard.patch.yml:135`).
    Disable it and the `cordis` creator preset.
  - Mount no local shell, fs or ptc rows.
  - Any trusted host plugin can read `.credentials.yaml`. File modes don't
    stop same-uid code, which is why only our pinned plugins run.
- **What reaches a remote provider.** The full transcript on every request:
  prompts, tool outputs, remote file contents, paths and host names. The
  same goes for title and compaction side calls (`purpose`,
  `packages/llm/llm/src/types.ts:552`), which go to the default model. That
  is acceptable per TJ for LAN details. Never keys: they don't reach tools,
  agent forwarding is off, and secrets stay out of tool output.
- **Telemetry.** DSH's feedback telemetry posts to `deepseeksvc.com` by
  default (`packages/bundle/base/cordis.patch.yml:200-220`), and its
  redaction ships no rules. Set `DSH_TELEMETRY_DISABLED=1` in the sidecar.

### v3 scope: an experimental workspace, plugins only (TJ, 2026-10-11)

TJ narrowed v3 to a base for later experiments, possibly thrown away, that
lets Hugh try it soon. **In:** DSH from a container, the Responses WebSocket,
persistent config, UI consistency, plugins that users add and configure, an
"Experimental" tab. **Out for v3:** every custom engine feature (edits with
recompute, steering, server-owned sessions, repack, KV hooks, Realtime).
The "Staged steps" table below is the long-term list; v3 runs W0-W3 here.

**Our own fork, patched where that is the better design (TJ, 2026-10-11).**
TJ: patching DSH in our own GitHub repo is fine; no upstreaming, and nothing
done a worse way just to stay plugin-only.
- The fork is `deepseek-harness` under our GitHub account, pinned in this
  repo like SparkInfer: a submodule at `third_party/deepseek-harness` plus a
  tree lock. The sidecar image builds from that source (`pnpm install`,
  `pnpm run build`), not from npm.
- Our work is a `cuteafd` branch on top of an upstream release tag
  (`dsh-v0.2.1-alpha.2` first). Our own packages live in-tree under
  `packages/cuteafd/*`.
- Seams stay the first choice where they fit cleanly: the LLM adapter,
  `ctx.theme`, settings, profile patches. Where a seam is missing or
  awkward, we patch core directly. Examples: the user-actions slot, the
  remote-aware workspace registry, the folder picker. Each patch is its own
  small commit, so a rebase shows exactly what conflicts.
- Adopting a new upstream version: rebase `cuteafd` onto the new tag, run
  `scripts/agent/dsh-smoke.sh`, then bump the pin and lock here.

| # | Step | Size | Gate |
|---|------|------|------|
| W0 | Sidecar image from the pinned npm release (`@deepseek-ai/dsh` + web frontend), `scripts/launch/agent.sh` (also `run.sh --agent`), one mounted folder `~/.local/share/cuteafd/agent` (0700) with `DSH_HOME`, `DSH_AGENTS_HOME`, `HOME` under it, telemetry off, bound to 127.0.0.1. Default route: our gateway via pi-ai `openai-responses` (HTTP; the Engine backend serves it since B1) with a dedicated `agent` key, set as `agent-default-model`; other providers registrable as usual | S | A coding task completes on the local model; config, sessions and installed plugins survive a container restart; `docker logs` holds no token or key |
| W1 | Facet: an **Experimental** nav tab, `/agent` page with our header and a same-origin iframe of `/agent/app/` (reverse proxy incl. the `api/remote.mux` WebSocket), server-side DSH cookie bootstrap behind the console lock; theme plugin mapping our `--bg/--panel/--ink/--line` and accents onto DSH's `--dsw-*` alias tokens via `ctx.theme`, both light and dark | S | Visual check in both themes at desktop and phone width; locked console means locked agent; no DSH token in browser history |
| W2 | `dsh-llm-cuteafd` adapter plugin: Responses WebSocket per DSH session, `previous_response_id` plus delta input when the new input extends the last request and its output, full input otherwise (pi-ai's rule, which it offers only for Codex OAuth routes); reconnect sends the full input. Selectable per route (`transport: websocket`), default for the cuteafd route | M | Requests the engine sees are byte-identical to the HTTP path on recorded sessions with retry, compaction and edit; bytes per step before -> after; prefix-cache hit tokens per step; no C1 regression |
| W3 | Plugins and version bumps: DSH's **Plugins** page stays on so the operator can add, enable and configure bundles (npm, git, path); the agent's `plugin_manager` tool and the creator preset stay off, so the model cannot install code. `dsh-smoke.sh` starts the pinned image, runs one task with a tool call, checks the theme and our bundle loading, and exits | S | A third-party bundle installs from the Plugins page and survives restart; the smoke passes on the pin and on the newest upstream release at the time |

**Every session works on a remote host over SSH (TJ, 2026-10-11).** A
session's workspace is a folder on an SSH host chosen when the session starts.
Every tool runs there: shell, files, terminal, search, LSP, jobs. Nothing runs
inside the container; it holds only the UI, the agent loop, config and
credentials. Our fork makes this the only execution mode.
- **Session binding.** New session: pick a registered host, then a folder on
  it with the SSH-backed picker. The session stores `(host alias, path)`, and
  every `ctx.fs`, `ctx.subprocess`, `ctx.sandbox` and terminal call for it
  routes to that host through `dsh-ssh` (`fs-ssh`, ssh subprocess, the ssh
  realm). The shell is the remote user's login shell with the remote
  environment; nothing is copied from the container.
- **Local execution is removed, not just hidden.** The local fs, shell, ptc
  and sandbox rows are unmounted. A session with no reachable host fails
  with a clear error; it never falls back to the container.
- **Fork patches** (the seams are missing or local-only):
  - a host-aware `WorkspaceRegistry` keyed by `(host, path)`;
  - the SSH directory picker;
  - remote `@file` completion;
  - polling `watch` for `fs-ssh`;
  - session-to-host binding in the session record;
  - restoring a session reconnects to its host.
- **Hosts and keys.** A settings card registers hosts: alias, hostname, user,
  port, key paste or generate, and a `known_hosts` confirm showing the
  fingerprint.
  - Keys stay at 0600 under `/data/home/.ssh/keys/`. The generated
    `ssh_config` uses `IdentitiesOnly`, `BatchMode`,
    `StrictHostKeyChecking=yes` and `ForwardAgent=no`.
  - Because no tool runs in the container, the model has no path to read
    the key files. Plugin install stays operator-only, as in W3.
  - The pinned `dsh-ssh` helper is installed on first connect. The arm64
    and x64 builds embed Node, so Sparks need nothing preinstalled.
- **Cluster hosts.** The cluster's hosts can be registered like any other
  host. A session on a host currently serving a model shows a warning.

This makes W4 part of the first usable version: W0 ships with a minimal host
picker and the remote binding, so no container execution path ever exists.

| # | Step | Size | Gate |
|---|------|------|------|
| W4 | Remote-only execution: host and key settings card, SSH picker, host-aware workspace registry, session to host binding, remote `@file`, `fs-ssh` watch, local execution rows unmounted. Lands with W0 for the first usable build | M | On one Spark: bash, read, edit, search, terminal and file watch run remotely with the remote user's shell and environment; nothing executes in the container (process audit while a session works); two sessions on two hosts at once; restart then restore reconnects; the model cannot read the key file through any tool |

### Remote-first harness design (2026-10-11)

DSH was built for local execution, where processes, watches, terminals and
searches are free and instantaneous. Its SSH family (`packages/ssh/*`) makes
one remote host look like the local world through the `ctx.fs`,
`ctx.subprocess` and `ctx.sandbox` seams, with the harness keeping sessions,
model transport and approvals (`docs/subsystems/ssh.md:170-192`). This section
decides, subsystem by subsystem, how each works when every session's workspace
is remote, what we patch, and in which W-step. Paths are `deepseek-harness/`
at `d7432673`; "helper" is the `dsh-ssh` helper we build from the fork.

**Facts that drive the design.**
- The connection is one OpenSSH master whose own command is the helper
  (`ssh -T -M -S <ctl> -o ControlPersist=no … <host> <helper>`,
  `packages/ssh/ssh/src/index.ts:299-303`). Administrative RPC is 4-byte
  framed JSON on the exec channel, bidirectional (the peer handles inbound
  requests, `protocol.ts:197-227`); every program stream is a separate
  forwarded Unix socket (`ssh -O forward`, a subprocess per stream,
  `index.ts:171`) authenticated with TLS-PSK. Heartbeats every `leaseMs/3`
  (10 s); the helper kills every managed process when the lease (30 s) or
  the channel ends (`helper.ts:84-87`, `helper-processes.ts:363`). Loss is
  final: "never reconnects to replay a possibly executed action"
  (`ssh.md:188`), `SshRpcPeer.close` rejects all pending calls.
- The helper is a Node program running the **local** providers on the
  remote (`SandboxedFileSystem`, `LocalSubprocessRuntime`,
  `LocalSandboxProvider`, `helper.ts:27-35`), so remote reads, atomic edits,
  version guards, bwrap/Landlock confinement and PTYs are the same code as
  local. `fs-local` already has chokidar `watch`; `fs-ssh` does not expose
  it (`fs-ssh/src/index.ts`, 107 lines, no `watch`).
- Tools reach providers through the context they were applied with. Preset
  plugins mount once in a shared preset subtree
  (`packages/preset/agent-preset-registry/src/mount.ts`, `index.ts:281`),
  so a provider installed on `agent.ctx` does not move a tool whose closure
  holds the preset `ctx` (W0+W4 finding). Prior art for call-time
  resolution exists: `terminal-controller` reads
  `agent.ctx.get('subprocess')` per call
  (`packages/api/terminal-controller/src/index.ts:332-339`) and the tool
  runtime resolves `workingDirectory` and `sandboxPolicy` per execution
  (`packages/core/tools/src/index.ts:947-961`).
- `tool-fs-search` spawns the **container's** `@vscode/ripgrep` path through
  `ctx.subprocess` (`packages/fs/tool-fs-search/src/search-core.ts:85-87`):
  remotely that is a binary that does not exist on the host. `glob` and
  `grep` are broken until this is fixed, so it belongs to W4's gate.
- The helper is started by sshd as `$SHELL -c <command>`: a non-login,
  non-interactive shell. PATH, conda, modules and `~/.profile` exports are
  absent, and every spawned process inherits that environment
  (`helper-processes.ts:180-181`, merged onto the helper's own env).
- `ssh` stderr is discarded (`index.ts:306`), so an auth failure, a changed
  host key and an unreachable host all surface as "SSH helper disconnected".

**0. Routing: one execution world per session, resolved per call.** A
`cuteafd-execution-worlds` host plugin owns `ExecutionWorld {host, path,
connection, fs, subprocess, sandbox, facts, state}` per live session, built
from the header's `cuteafd_workspace` (W0's typed field, fork `c3c68dd9`).
The root `ctx.fs`, `ctx.subprocess` and `ctx.sandbox` rows are **dispatching
providers**: each method forwards to the current world's provider, where
"current" is an `AsyncLocalStorage` set at three choke points: a prepended
`tools/execute` waterfall listener (`within(exec.agent, next)`, no core
patch), a prepended `agent/pre-step` listener (covers `agent-instructions`
and `working-directory.ensure` outside tool calls), and the API gateway's
Remote dispatch when it resolves an `Agent` or a session-scoped request (one
patch in `packages/api/gateway`, plus a 5-line entry in `workspace-files`,
whose `WorkspaceFileScope` names the session). Outside any world the
dispatcher throws `NoExecutionWorldError`; there is no local fallback, which
is the "local execution removed" rule made structural. This replaces W0's
`exec.agent?.executionCtx ?? ctx` edits in nine tool packages: zero per-tool
patches, and every operator-installed plugin that uses `ctx.fs` or
`ctx.subprocess` routes remotely without knowing. Handles created inside a
world (a `RemoteProcess`, a PTY, a text stream) close over their connection,
so the jobs pump, terminal follow streams and `readOutput` work outside the
ALS scope. The guard test mounts a recording fake world, runs every tool the
standard preset registers in a remote session, and asserts each dispatched
through that world and that no local `fs`/`subprocess`/`sandbox` provider is
reachable from the root. v3 (W4/W5).

**1. Transport.** One `SshConnection` plus helper **per host alias**, in an
isolated child context (`ctx.isolate` over `ssh`, `fs`, `subprocess`,
`sandbox`; `sandboxPolicy` stays shared), refcounted by the sessions bound
to that host and disposed 60 s after the last one goes. One master per host
rather than per session: fewer processes, one warm channel for a session's
subagents, and the failure domain is already the host. Upstream's
non-reconnecting semantics stay: a drop fails every in-flight call with a
typed error (outcome unknown, never replayed), the world enters
`disconnected`, the agent loop sees a normal tool error ("host `emu`
unreachable; the command did not run" or "…lost mid-call; outcome unknown")
and decides, and the next call reconnects lazily (new master and helper,
exponential backoff capped at 30 s, a "Reconnect" action in the banner).
A host reboot is the same path; nothing remote survives it in v3 (see 2).
OpenSSH settings: `ServerAliveInterval=5`, `ServerAliveCountMax=2`,
`ConnectTimeout=10`, `LogLevel=ERROR`, `leaseMs` 15000, so a dead host is
detected within ~15 s instead of 30. Liveness in the UI: a per-session dot
(connecting / ready / degraded / disconnected) with the host alias and the
heartbeat round trip. Patch size: the pool and dispatcher are our package;
`dsh-ssh` gets a bounded stderr ring (next item 12) and the login-shell
launch (8). v3 (W5).

**2. Process lifecycle.** Commands, background jobs and PTYs are helper
children in a managed range; cancellation is `process.terminate` over the
admin channel; the helper's lease kills orphans within 15 s of a drop. In v3
a connection loss therefore kills every job and terminal on that host: the
job settles `failed` with "connection lost; exit code unknown", the terminal
shows `exited: connection lost`. Honest, and the same as upstream. **Later
(W8): detached execution.** The helper grows a per-host daemon mode
(`dsh-helper --daemon`, started with `systemd-run --user --scope` when
available, else `setsid`), `process.detach(id)` reparents a job to it with
stdout/stderr spilled to `~/.cache/cuteafd/dsh-helper/jobs/<id>/`, and
`process.attach(id)` after a reconnect or a container restart returns the
spill offsets, live tail and exit code. Bash background jobs and terminals
opt in; the jobs registry records `(host, remoteJobId)` in a plugin-owned
session event so a restored session lists and reattaches them. Not tmux: it
is not guaranteed on a host, and it owns no exit codes or offsets.

**3. File watching.** What watching does: the sidebar file tree
(`ui-sidebar-files` → `workspaceFiles.changes`, one non-recursive directory
watch per open node, `packages/api/workspace-files/src/changes.ts:79`) and
`fs/observed` invalidation. The model never watches, and `workspace-changes`
uses git. So: no recursive watches, no watching build output, and nothing an
agent needs. Decision: `cuteafd.fs.watch {target}` / `cuteafd.fs.unwatch
{id}` helper RPCs backed by the helper's own `fs.watch` (chokidar, already
in `fs-local`), with a helper-to-client `cuteafd.watch.changed {id}`
notification through the existing bidirectional peer, debounced 250 ms per
watch, at most 64 watches per connection (the 65th fails `FS_IO_ERROR`,
the sidebar falls back to its refresh button). Non-recursive only, as the
consumer. W4 ships `watch` throwing unsupported (the sidebar's refresh works
and `changes` reports `watch-unsupported`); W5 lands the helper watch. No
polling: it costs a round trip per open directory every few seconds for a
view nobody is looking at most of the time. v3 (W5).

**4. Terminal.** Already remote: `terminal-controller` spawns through the
world's `subprocess.spawnTerminal`, which is a helper PTY with its own SSH
channel; resize, signals and foreground inspection are RPCs
(`subprocess-ssh/src/index.ts:293-330`). The host keeps the recovery screen
and scrollback, so a browser reconnect replays as today. An SSH drop kills
the PTY (2); reconnecting to a live terminal after a drop is W8's attach.
Nothing to patch in v3 beyond routing (0).

**5. LSP.** `lsp-stdio` spawns the configured server through `ctx.subprocess`
and reads through `ctx.fs` (`packages/lsp/lsp-stdio/src/index.ts:146-157`),
so a server installed on the host works through the world as-is. Cost: one
server process per language per host, seconds of startup, hundreds of MB,
and a `command` that must exist on that host. Benefit to an agent with `rg`:
modest (four queries). Decision: off in v3; W7 adds a per-host "language
servers" list on the host card (command, extensions), registered through the
world and started on first query. Later.

**6. Search and reads.** `glob`/`grep` must run the host's ripgrep: patch
`resolveRgPath` to `ctx.subprocess.resolveExecutable('rg')` (dispatched to
the host) and, when absent, to the `rg` our helper archive ships beside the
helper (the path arrives in `hello.cuteafd.tools.rg`). This is W4's gate
("search runs remotely"), so it lands with W4. Reads are already one RPC
for files up to 8 MiB (`fs.readText`, `helper.ts:185`) and a chunk stream
above; the `read` tool is resolve + stat + read, three round trips, which
at LAN latency is a few milliseconds (9). Binary reads go through
`readBytes`/`readByteRange` as base64 inside the 64 MiB frame cap
(`read_image` included). No batching API in v3; measure first (9), batch if
the numbers say so.

**7. Sandbox and approvals.** `sandbox-ssh` asks the helper to confine the
argv with the host's own bwrap or Landlock (`helper.ts:149-154`), and
`fs.write`/`fs.edit` carry the per-call policy the helper enforces
(`helper.ts:228-245`). The meaning is unchanged on a remote host: file
effects of the SSH user, confined to the session's path under
`workspace-write`, with the `full`/`partial` enforcement fact reported in
results. Approvals stay host-side (the UI answerer), unchanged. Danger
levels map one to one: `danger-full-access` on a cluster host means
"whatever the SSH user may do", including sparknest and the lock files,
so the presets keep DSH's defaults (`workspace-write` + ask, `danger` +
never) and the host card shows the effective preset and the "serving" tag.
A dedicated remote user is an operator choice when registering a host, not
v3 work. Remote sandboxing beyond file effects (network, process
visibility) is out of scope, as upstream.

**8. Environment and paths.** Launch the helper through the remote login
shell: the exec command becomes `exec "$SHELL" -l -c '<helper command>'`
(bash, zsh and fish all accept `-l -c`), so `/etc/profile`,
`~/.bash_profile`/`~/.profile`, conda's and modules' exports and the user's
PATH are what every spawned process inherits. Commands keep upstream's
`bash -c` (`bash-local/src/index.ts:188`): interactive-only hooks (direnv,
`.bashrc` aliases) are not present, and the prompt says so; the model can
`source` what it needs. cwd is the header's remote path; relative paths
resolve on the host against it; `~` is the remote home; temp is the
host's `/tmp`. `hello` grows `cuteafd: {hostname, user, home, shell, arch,
release, tools}` (one `helloSchema` line on the client, a few in the
helper) for the prompt (11) and the host card. Hosts: anything that accepts SSH
and runs bash where the helper has a build (Linux x64/arm64 glibc >= 2.28,
WSL included, and macOS); the host card probes and names what is missing
(decision 6). v3 (W5).

**9. Latency budget.** LAN RTT is ~0.2 ms and a helper RPC ~1-3 ms, so file
tools are cheap. The expensive part is process launch: `process.prepare`,
then one `ssh -O forward` **subprocess per stream** (stdout, stderr, stdin
or control), each a 20-40 ms exec, then `process.start`, `process.done`
and `process.terminate`: roughly 100-150 ms of overhead per `bash`, `rg` or
`git` call before the command runs. Targets: fs tool ≤ 10 ms p50, process
tool ≤ 50 ms p50 of overhead over the command itself, connection open ≤ 2 s
on a Spark. Plan: W5 measures (table per tool, raptor→Spark and
raptor→raptor); W6 adds `cuteafd.process.run`, a single RPC for the
collect-mode, no-stdin, no-control spawns (bash foreground and background,
rg, git): argv, cwd, env, caps in, outcome, bounded tails and spill paths
out, with `cuteafd.process.output` notifications for jobs, no forwarded
sockets at all. PTYs, LSP and PTC keep the forwarded-stream path. If
forwards still dominate, W6 also replaces the `ssh -O forward` spawn with
the OpenSSH mux protocol spoken directly to the control socket
(`PROTOCOL.mux`, `MUX_C_OPEN_FWD`), about 150 lines inside
`controlCommand`. Prefetch: open the host's connection at session create or
restore, not at the first tool call, and `stat` the cwd then. Stop bar: the
targets above, or a measured floor of RTT plus helper work.

**10. Multiple hosts.** One host per session (the header), for v3 and
after: a single execution world keeps every path the model sees
unambiguous. Subagents and forks inherit the parent's header
(`cuteafd_workspace` copied in `agents.create` meta, W0 does this for forks;
spawn is checked in W5). No per-tool host targeting. Moving a session to
another host is a fork with a new header plus a notice to the model that
paths in its history refer to the old host (W9, S, only if asked for).
Agent teams across hosts are out of scope.

**11. The model's view.** A `cuteafd:remote-host` prompt section (order
beside the persona) says: the workspace is `<path>` on host `<alias>`
(`<hostname>`, `<os> <arch>`, user `<user>`), reached over SSH; every tool
(bash, read, edit, glob, grep, terminal) runs on that host; the assistant's
own machine is not accessible; there is no display or browser on the host;
network locality is the host's (`localhost` means `<alias>`); commands run
in a non-interactive shell with the login environment. The
`working-directory` context line becomes `Current working directory:
"<path>" on host <alias>` (`packages/session/working-directory/src/index.ts:172`,
one line). Tool descriptions are not patched: the section is cheaper and
the KV prefix stays one block. The transcript header in the UI shows the
alias and state. v3 (W5).

**12. Failure UX.** `dsh-ssh` keeps a bounded (4 KiB) ring of `ssh` stderr
and classifies exit 255 into `unreachable` (`Connection refused`, `No route
to host`, `Could not resolve hostname`, `timed out`), `auth-failed`
(`Permission denied (publickey`), `host-key-changed` (`REMOTE HOST
IDENTIFICATION HAS CHANGED`), plus our own `helper-mismatch` (digest or
protocol) and `helper-missing` (install needed); raw stderr never reaches
the model (it can contain paths). The session banner shows the state with
one action: Reconnect, Re-trust (shows the new fingerprint, the card's
confirm flow), Fix key (opens the host card), Install helper. The agent
loop needs nothing: a tool call during `disconnected` fails fast with the
typed message and the model answers or stops; a call that was in flight
reports "outcome unknown". Restoring a session whose host is down opens it
read-only with the banner; tools fail fast until reconnect. Disk full on
the host: `fs.write` fails `FS_IO_ERROR` (ENOSPC) and spill files are
dropped with a tail-only note (`logSpillFailure`); a helper that cannot
create its temp root fails `hello` with a clear message. v3 (W5).

**13. Change view.** `workspace-changes` snapshots the git working tree at
turn start and end (`git add --all` into a private `GIT_INDEX_FILE`, then
`write-tree`; `diff-tree --numstat` between the two; `cat-file` for a
file's sides, `packages/deliverables/workspace-changes/src/git.ts:162-235`)
and, for paths git does not cover, copies whole files around each file-tool
edit with `node:fs` into a local temp tree (`capture.ts:41-65`,
`recorder.ts:2`). Routing git through the world moves the commands but
leaves the index file, the captures and the blob reads on the wrong
machine, so W4 unmounts it in remote sessions. Decision: run upstream's
recorder **inside the helper**. The helper already runs the local
providers; it imports the recorder (`git.ts`, `capture.ts`, `recorder.ts`,
exported by a `./recorder` entry we add to the package) and exposes
`cuteafd.changes.start {session, cwd}`, `turnStart {turn}`, `observe
{toolResult}` (the file-tool result paths), `turnEnd {turn}` → summary, and
`diff {turn, path}` → both sides, with temp index and captures under the
host's `/tmp`. A `cuteafd-workspace-changes-remote` host package replaces
the `workspace-changes` row: it implements the same `workspaceChanges`
service and `workspace/changes` session event by forwarding, so the client
file-change view is untouched. Caps stay the recorder's (`maxFiles`,
`fileMaxBytes`); a disconnect mid-turn abandons that turn's record, as a
failed snapshot does today. Later (W7).

**14. Skills.** `skill-filesystem` scans project roots
(`<projectRoot>/.dsh/skills`, `<projectRoot>/.agents/skills`), user roots
(`$DSH_HOME/skills`, `$DSH_AGENTS_HOME/skills`) and the bundled root with
`node:fs` and chokidar (`packages/skill/skill-filesystem/src/index.ts:244-262,
459-491`). Remotely the project roots and the remote user's own
`~/.agents/skills` are on the host; the operator's roots in the mounted
folder (`/data/agents/skills`, `/data/dsh/skills`) and the bundled root are
not, and they are trusted operator content. Decision: two providers on
`ctx.skills`. The upstream provider stays mounted with `includeDefaultRoots:
false` and the two mounted-folder roots as custom roots (local, watched,
unchanged). A `cuteafd-skill-remote` provider serves the remote roots:
`<projectRoot>/.dsh/skills`, `<projectRoot>/.agents/skills` and
`~/.agents/skills` on the host, ranked as upstream ranks them. One helper
RPC, `cuteafd.skills.scan {roots, known: {path: mtime}}`, returns every
`SKILL.md`'s frontmatter and mtime in one round trip and bodies only for
entries whose mtime changed, so a scan is a few milliseconds. Refresh: at
session start, at every `turn/start` (one RPC, cached by mtime), on the
Skills card's refresh button, and through a `watch` on each root directory
from W5's budget (three per session). `load()` reads the body through the
world's `fs.readText`. The W4 interim (read once at session start) becomes
this in W5.

**Order after W0-W4.**

| # | Step | Size | Gate |
|---|------|------|------|
| W5 | Remote correctness: execution-world dispatcher (ALS) replacing per-tool `executionCtx` edits; per-host connection pool; login-shell helper launch; `hello.cuteafd` facts; remote-host prompt section and cwd line; helper `fs.watch` replacing unsupported; remote skill provider with `skills.scan` and refresh; stderr classification, states, banner and lazy reconnect; subagent header inheritance; per-tool overhead table | M | Guard test: every standard-preset tool dispatches through the recording world, no local provider reachable; two sessions on two hosts; `bash -c 'echo $PATH'` equals `ssh host 'echo $PATH'` under a login shell; prompt names the host; killing sshd mid-call yields the typed error within 15 s, the banner, and a reconnect on the next call; the sidebar updates after a remote `touch` within 1 s; a skill added under the remote `.agents/skills` is listed on the next turn and the operator's mounted skills stay listed; overhead table recorded |
| W6 | Latency: `cuteafd.process.run` one-RPC path for collect-mode spawns with `process.output` job notifications; mux-protocol forwards if forwards still dominate; connection prefetch at session open | M | Overhead before → after per tool on raptor→Spark; bash ≤ 50 ms p50, fs ≤ 10 ms p50, or a measured floor; byte-identical tool results to the forwarded path on recorded sessions |
| W7 | Change view in the helper (`changes.*` RPCs running upstream's recorder on the host, forwarding `workspaceChanges` service); LSP on the host: per-host language-server list on the host card, `lsp-stdio` through the world, off by default | M | A turn that edits tracked, untracked and ignored files on a Spark shows the same summary and diffs as a local session on the same repo; no temp files left in the container; goToDefinition and findReferences on a Rust and a TS repo on a Spark; no server process when the list is empty |
| W8 | Detached execution: helper daemon mode, `process.detach`/`attach`, spill directory, job and terminal reattach after reconnect and container restart | L | A 10-minute job survives `ssh -O exit`, a container restart and a reconnect; its output and exit code are read back; orphan count on the host is zero after the session ends |
| W9 | Host move: fork a session onto another host with a model notice | S | Paths in the fork's history are flagged; tools run on the new host |

**What the helper grows (ours, in the fork)** versus plain OpenSSH:
- Helper RPCs, all under a `cuteafd.` prefix handled by one
  `helper-cuteafd.ts` module that upstream `helper.ts` calls through a
  three-line hook, so a rebase re-applies one hook: `fs.watch`/`fs.unwatch`
  with `watch.changed` notifications, `skills.scan` and `hello.cuteafd`
  facts (W5); `process.run` and `process.output` (W6); `changes.*` running
  the upstream recorder on the host (W7); daemon mode with
  `process.detach`/`attach` (W8). Protocol version stays upstream's; our
  capabilities are advertised in `hello`.
- Archive contents: `rg` beside the helper (W4), later the daemon
  launcher. We build the archive in the sidecar image build (the fork
  already builds helpers) and pin its hash there; first connect installs it
  to `~/.cache/cuteafd/dsh-helper/<version>/` over the same SSH alias.
- Client side in `dsh-ssh`: login-shell launch, stderr ring and
  classification, `helloSchema` extension, tuned keepalive and lease, and
  (W6) mux-protocol forwards. Everything else, connection pooling, the
  dispatcher, states, banner, prompt section and host card, lives in
  `packages/cuteafd/*`.
- Plain OpenSSH does the rest: `ControlMaster`, `ServerAlive*`,
  `ConnectTimeout`, `IdentitiesOnly`, per-alias `UserKnownHostsFile`,
  `StrictHostKeyChecking=yes`, `BatchMode`, `ForwardAgent=no`.

**Rebase risks and how the patches stay small.** The core touches are the
session header field (done), the gateway ALS entry, `workspace-files`'s
entry, `tool-fs-search`'s `rg` resolution, the `working-directory` line, and
the three `dsh-ssh` client changes; each is its own commit prefixed
`cuteafd:` and under 40 lines. Everything with real logic is a new package
under `packages/cuteafd/*` or the `helper-cuteafd.ts` module. The risks:
upstream reworking `SshConnection` or the helper protocol (our RPCs are
namespaced and hook in at one line, but a protocol-version bump means
rebuilding and reinstalling helpers, which first-connect install handles);
upstream adding its own `watch` or reconnect (we drop ours); the preset
registry changing how tools capture `ctx` (irrelevant to the dispatcher,
which sits below the tools); `ToolRuntime` ceasing to expose `tools/execute`
as a waterfall (then the dispatcher enters from `agent/pre-step` alone and
one core patch around dispatch). `dsh-smoke.sh` runs the guard test and one
remote task on every rebase.

**Decided (TJ, 2026-10-11).**
1. **One SSH connection per session**, not per host. A session's subagents
   share it. This replaces the per-host pool in item 1: each session owns
   its `SshConnection` and helper in its own child context, closed when the
   session closes or after 60 s idle, and a drop affects only that session.
2. **Detached jobs (W8) through our own helper**, which we control: daemon
   mode, with `systemd-run --user` when present. Not tmux.
3. **Permission defaults unchanged.** The presets keep DSH's own defaults
   (`workspace-write` + ask; `danger-full-access` as an explicit
   per-session switch).
4. **LSP off by default**, which is also upstream's default: the `lsp` group
   ships no servers and no bundle mounts it. W7 adds a per-host opt-in list.
5. 15 s dead-host detection (5 s keepalive, 15 s lease), as recommended.
6. **Hosts: anything that accepts SSH and runs bash**, WSL included (its
   sshd is Linux). Registration probes the host: `bash` present, `uname -sm`,
   libc. It accepts the host when the helper has a build for that platform:
   Linux x64/arm64 with glibc >= 2.28 (WSL included), and macOS x64/arm64.
   Otherwise it names what is missing. musl (Alpine), FreeBSD, and native
   Windows via MSYS or Cygwin bash need a helper build first, and are added
   when a host needs them.
7. The ALS dispatcher replaces the per-tool edits before W4's gate (done).

### Staged steps and gates

| # | Step | Size | Needs phase B | Gate |
|---|------|------|---------------|------|
| 0 | Spike: sidecar plus `/agent/app/` proxy, `openai-completions` route to the engine's chat path, default model, telemetry off, volume layout | S | no | A coding task completes through the dashboard on the local model; the WebSocket survives the proxy; per-step prefix-cache hit rate and bundle size measured; `docker logs` holds no token or key; the console cookie gets 401 on `/v1/*` |
| 1 | Facet: nav entry, iframe page, server-side DSH cookie bootstrap, `@cuteafd/dsh-facet` theme and brand, lock-down patch (plugin manager, creator preset and telemetry off) | S | no | Visual check; locked console means locked agent; no DSH token in browser history |
| 2 | Remote execution: `@cuteafd/dsh-remote` with a host and key settings card (alias, user, port, key paste, `known_hosts` confirm), helper install and hash, one preset per host (ssh realm), SSH directory picker, host-aware workspace registry | M | no | On one Spark, bash, read, edit and terminal run remotely; the picker browses the Spark; tools that try every path to the key fail; a second host works in a second session at once |
| 3 | Edit any message: projection event, edit UI on user and assistant nodes (reasoning included), signature drop, gateway fix for Responses reasoning precedence | S | no | Edited reasoning appears in the next request (recorder tape); recompute starts at the edit, not at 0; restore and fork of an edited session work |
| 4 | Steer by interrupt: busy-Enter cancels with `keepInbox` and resends; engine parks a cancelled decode as a `Prompt` snapshot | S | engine change only | Steer-to-first-token latency before → after; no recompute of the kept prefix |
| 5 | `dsh-llm-cuteafd` adapter: Responses WebSocket per session, `previous_response_id` deltas, full resend on mismatch | M | step 1 (Engine backend) | Server-side requests byte-identical to the HTTP path on recorded sessions with retry, compaction and edit; bytes per step before → after; no C1 regression |
| 6 | True mid-generation steer: `cuteafd.steer.inject`, split-and-adopt in the adapter | L | step 3 (inject) | Injected tokens prefilled only; the logged history re-renders the engine's exact token sequence (golden); emitted tok/s under steering |

**Spike unknowns** (step 0, about a day):
- whether the proxy needs header rewriting beyond `trustedHosts`;
- the template prefix-stability hit rate per family;
- whether the sidecar's pnpm plugin install works offline from the volume.

**Unknowns for step 2:**
- whether a `WorkspaceRegistry` replacement is enough, or an upstream PR is
  needed;
- whether per-host realms isolate the terminal, LSP and jobs cleanly.

**Hooks for later (not designed):**
- **Server-side auto-compaction:** replace `compaction-basic` with a plugin
  that requests a summarize op.
- **History manipulation and KV splicing:** projection events.
- **Virtual forking:** DSH fork at a sequence number maps to a
  `previous_response_id` parent, which shares KV pages.
- **Model introspection ("j space"):** a sidebar panel slot fed by extension
  events.

### Option under discussion: the engine owns the session (2026-10-11)

TJ wants to consider a server-owned conversation, like Realtime: the engine
can repack history and tell the client, and it gives disk-persisted KV a
natural owner. How it would work:

- **Traffic is incremental both ways.** Client to server: item ops (append,
  update, delete, truncate). Server to client: item events plus a new
  `cuteafd.conversation.repacked` (range, replacement items, reason,
  revision). DSH still derives each step's history from its own log, so the
  adapter keeps the two in step:
  - Each step it hashes DSH's messages, finds the longest common prefix with
    the server's items, and sends delete/insert ops for the rest. That is
    O(n) hashing and a few ops.
  - It writes each server repack into DSH's log as a plugin-owned projection
    event (the edit mechanism in e), so "model-visible means logged" holds.
    DSH's own compaction is turned off; the server compacts.
  - Every server event carries a revision. A step whose history hash
    disagrees with the server's falls back to a full resync. The sync then
    heals itself instead of needing to be perfect.
- **Realtime as the wire, plus extensions.** Realtime has the item ops and
  live events, but lacks five things we would add as `cuteafd.*`: reasoning
  items and deltas, `item.update`, no 60-minute expiry, `session.resume` with
  the last applied revision (the official protocol has no resume: clients
  replay their own buffer), and fork/steer/KV ops. Our `Session`/`SessionOp`
  is already server-owned; `Compact`, `Splice`, `Steer` and `Kv` are stubs.
- **Responses + Conversations instead.** OpenAI's documented server-owned
  text state is the Conversations API: a durable `conv_` id, item list/add/
  delete, and `conversation` on Responses. Reasoning, tools and resume (the
  id is durable) come built in; only server pushes (repack) and item update
  are extensions. We don't serve `/v1/conversations` yet.
- **New work either way:** durable session store (items, revisions, op log)
  surviving restart; resume; engine-side compaction behind `Compact`/
  `Splice`; disk tier for prefix snapshots keyed to sessions (S2 reports what
  that needs); the DSH adapter grows to roughly 1-1.5K lines.

### Open questions for TJ (with recommendations)

1. **Main connection.** Realtime as-is, or the Responses WebSocket plus
   cuteafd events? *Responses WebSocket.* Realtime fights DSH's transcript
   and drops reasoning; it stays for voice.
2. **Where DSH runs.** Sidecar container, or inside the coordinator image?
   *Sidecar.*
3. **Facet form.** Same-origin iframe, or a direct mount? *Iframe.*
4. **A separate `agent` API key** for usage attribution? *Yes*; gateway auth
   gains a small multi-key file.
5. **Volume location.** *A host-local `~/.local/share/cuteafd/agent` (0700),*
   never sparknest or the repo, because it holds SSH keys.
6. **Which hosts the agent may target.** *Any registered host,* with the
   cluster hosts tagged "serving" and a confirm when a model is up there.
7. **DSH version.** Decided (TJ, 2026-10-11): start on upstream's newest
   web release, not on what `dsh-desktop` tracks. That is `dsh-v0.2.1-alpha.2`
   (`d7432673`, npm `alpha` tag, upstream master tip as of 2026-10-11) for
   both `@deepseek-ai/dsh` and `@deepseek-ai/dsh-web-frontend`. Pin it exactly
   and bump deliberately, because the APIs are pre-stable.
8. **Upstream PRs** (user-actions slot, remote-aware Web workspace) or
   local patches? *Upstream PRs first,* with our plugin carrying a shim
   until they land.
9. **Single operator.** DSH has one operator identity, so the workspace is
   single-operator. *Accept.*

## API usage tracker and console access (design, 2026-10-09)

TJ: "add to the dashboard an API tracker that keeps some request log data,
token use, which APIs etc.; to avoid bloating space not retaining payload data
or long term data; async store into sqlite inside the coordinator container;
7 days default retention; interesting analytics about performance; aggregate
on sessions where possible with the better APIs." Then: "can keep 1 day full
log too, probably won't be too big; let those be configured on that dash";
"the cookie can be persisted locally where the launch script can reuse it, so
you don't have to keep reauthing"; "having cool and useful visualizations of
it will be great".

Three deliverables on top of the gateway branch: a **metadata tier** (7 days,
never payloads), a **full-log tier** (1 day, payloads, separate file,
switchable), and a **console unlock cookie** that gates token text, the usage
dashboard and its settings. Design branch `work/api-usage-design`; the code
lands on `work/api-gateway` after the gateway's front ends exist, since the
record is built from their `TurnRequest`/`TurnEvent`s.

### Where the record comes from

One `UsageScope` per HTTP request, created by a middleware and put in the
request extensions (the same seam as `gateway::record::Tape`). Handlers fill
what they know; the engine fills what only it knows; the record is emitted
when the **last reference drops**, so cancellations, disconnects and errors
all produce a row without any handler-side bookkeeping.

```text
 HTTP layer (cuteafd-api)                       engine side (cuteafd-daemon)
 ┌──────────────────────────────────────┐       ┌──────────────────────────────────┐
 │ middleware: t_arrival, route, method,│       │ console::Ticket (one per admitted│
 │   UA → client_kind, key label,       │       │   request): admit/first/retire   │
 │   status, bytes in/out               │  Arc  │   already timed; add per-round   │
 │ handler: protocol, models, session,  │◄─────►│   drafted/accepted/rounds        │
 │   stream, item/tool/media counts,    │ handle│   counters (3 relaxed adds per   │
 │   usage chunk, stop reason, TTFT     │       │   decode round, zero per token)  │
 └──────────────┬───────────────────────┘       └──────────────────────────────────┘
                │ last Arc drop → try_send(Record)           (never blocks)
                ▼
     bounded channel (4096) ──► usage writer thread ──► usage.sqlite (WAL)
     bounded channel (1024, 64 MiB) ──► log writer thread ──► usage-log.sqlite
```

- `cuteafd_api::usage` holds `Record`, `UsageScope` (an `Arc` of atomics and
  `OnceLock`s, no mutex on the hot path), the `UsageSink` trait and the
  middleware; no SQLite there. A new `cuteafd-usage` crate (mirroring
  `cuteafd-bench`'s `store.rs` + `http.rs` split, rusqlite bundled) holds the
  writers, schema, pruning, settings, queries and the `/console/usage/*`
  routes. The daemon wires both, as it wires the bench.
- `NativeRequest` gains `usage: Option<UsageHandle>`; `TurnRequest` gains the
  same beside `tape`. The generic families pass it to `console::admit`, which
  stores it in the `Ticket`; `Ticket::retire` and `Step::member` write the
  engine figures into the handle. V4.1's own console module gets the same
  two calls (it is the parity engine; S).
- Realtime: one record per `response.create` turn (protocol `realtime`,
  session = the connection's `SessionId`), plus audio seconds in/out as
  counts. The connection itself is one `realtime_session` row.
- Requests refused before admission (auth, 400, 429, bench lockout) are
  recorded with their outcome and no engine fields. Unauthenticated callers
  can therefore write rows; the size cap bounds that, and the row is ~300 B.

### Metadata tier: what is recorded (never payloads)

Per request: `rid` (also returned as `x-request-id` and used as the
response id where the protocol has one), arrival time, protocol (`chat`,
`completions`, `messages`, `count_tokens`, `responses`, `realtime`, `models`,
`other`), route and method, `client_kind` and a 120-char user agent,
`key_label`, optional client IP, requested and served model, session id and
how it was found, turn index in the session, stream flag, counts of items,
tools, images and audio parts, tokens (input, cached, output, reasoning,
draft proposed and accepted, decode rounds), timings (HTTP queue, engine
admission wait, TTFT, total), derived prefill and decode tok/s, concurrency
at admission (HTTP in-flight count at arrival, engine active count at
admit), HTTP status, outcome class (`ok`, `client_error`, `auth`,
`overloaded`, `cancelled`, `engine_error`, `bench_locked`), stop reason,
error class, and request/response byte sizes. No prompt, completion,
reasoning, tool argument, search query, image or audio bytes: the privacy
test writes a request whose payload contains a sentinel and asserts the
sentinel never appears in the metadata file's bytes.

- **Client identification.** `client_kind` from headers: Claude Code
  (`user-agent: claude-cli/…`, `x-app: cli`), Codex (`user-agent:
  codex_cli_rs/…`, `originator`), OpenAI and Anthropic SDKs, curl, browser
  (console pages), `bench` (the bench runner's token or probe header), else
  `other`. The exact header set is confirmed from the gateway's `--record`
  captures; the classifier is a table in one file.
- **PII stance.** The API key is stored as `key_label =
  "k:" + hex(sha256(key))[..8]`, never the key; ephemeral `ek_` tokens
  (phase C) label as the key that minted them. Client IP is off by default
  (`CUTEAFD_USAGE_CLIENT_IP=1` stores the peer address; the server sits on a
  private network, so no truncation when on). Auth headers never reach a
  record: the scope reads only the normalized key through `auth::presented`
  to hash it.
- **Session inference**, recorded as `session_source`:
  - `explicit`: Realtime connection id; Responses `previous_response_id`
    (the session is the chain's root response id, stored on the snapshot);
    a client-provided `x-session-id` header on any route.
  - `cache_key`: Responses `prompt_cache_key` (Codex sets it per
    conversation while sending `store:false` and full history, so chains do
    not exist for it); Anthropic `metadata.user_id` (Claude Code encodes its
    session id there; hashed to 16 hex, never stored raw); OpenAI `user`.
  - `prefix`: the engine restored this prompt from a prefix-cache entry that
    an earlier request created; the session id is inherited from that
    request. Each cache entry carries the creating `rid`; the restore reports
    it on the handle. Lower confidence, shown dimmed on the dashboard.
  - `none`.
  Turn index counts requests with the same session id in arrival order.

### Storage

- **Location.** Both files live in `/root/.cache/cuteafd/usage/` in the
  coordinator container, a bind mount of
  `~/.cache/cuteafd/<instance>/usage/` on the host (beside the instance's
  `api-key`, which `release_prepare_api_key` already creates under
  `~/.cache/cuteafd/<instance>/`; `run.sh` uses `default`). The bench does
  the same with `~/.cache/cuteafd/bench`. A restart, `--restart`, a WIP slot
  or a new image all mount the same directory, so history survives all of
  them; `--usage off` disables recording; no directory (tests, `cuteafd
  gateway` without one) means an in-memory store.
- **Two files**, so pruning, disabling or deleting the full log never locks or
  touches the metadata, and the big file's vacuum never stalls the small one:
  `usage.sqlite` (requests, sessions, daily, settings) and
  `usage-log.sqlite` (log). Both: `journal_mode=WAL`, `synchronous=NORMAL`,
  `auto_vacuum=INCREMENTAL`, `busy_timeout=2000`; one writer connection per
  file on its own thread; readers (dashboard queries) use a small pool of
  read-only connections on the tokio blocking pool.
- **Schema (metadata).**

  ```sql
  CREATE TABLE requests (
    id INTEGER PRIMARY KEY, rid TEXT NOT NULL UNIQUE, ts_ms INTEGER NOT NULL,
    protocol TEXT NOT NULL, route TEXT NOT NULL, method TEXT NOT NULL,
    client_kind TEXT NOT NULL, client_ua TEXT, key_label TEXT, client_ip TEXT,
    model_requested TEXT, model_served TEXT,
    session_id TEXT, session_source TEXT, turn_index INTEGER,
    stream INTEGER NOT NULL, n_items INTEGER, n_tools INTEGER, n_images INTEGER, n_audio INTEGER,
    tokens_in INTEGER, tokens_cached INTEGER, tokens_out INTEGER, tokens_reasoning INTEGER,
    draft_proposed INTEGER, draft_accepted INTEGER, rounds INTEGER,
    t_queue_ms REAL, t_admit_ms REAL, t_ttft_ms REAL, t_total_ms REAL,
    prefill_tps REAL, decode_tps REAL, concurrency_http INTEGER, concurrency_engine INTEGER,
    status INTEGER NOT NULL, outcome TEXT NOT NULL, stop_reason TEXT, error_class TEXT,
    bytes_in INTEGER, bytes_out INTEGER, bench INTEGER NOT NULL DEFAULT 0);
  CREATE INDEX requests_ts ON requests(ts_ms);
  CREATE INDEX requests_session ON requests(session_id, ts_ms) WHERE session_id IS NOT NULL;
  CREATE INDEX requests_model ON requests(model_served, ts_ms);
  CREATE INDEX requests_client ON requests(client_kind, ts_ms);
  CREATE TABLE sessions (                       -- writer-maintained upsert, same transaction
    session_id TEXT PRIMARY KEY, source TEXT, client_kind TEXT, model TEXT,
    first_ms INTEGER, last_ms INTEGER, turns INTEGER, tokens_in INTEGER, tokens_cached INTEGER,
    tokens_out INTEGER, errors INTEGER);
  CREATE TABLE daily (                          -- the only data older than the retention
    day TEXT, protocol TEXT, client_kind TEXT, model TEXT, requests INTEGER, errors INTEGER,
    tokens_in INTEGER, tokens_cached INTEGER, tokens_out INTEGER, draft_proposed INTEGER,
    draft_accepted INTEGER, ttft_hist TEXT, decode_hist TEXT, PRIMARY KEY (day, protocol, client_kind, model));
  CREATE TABLE settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
  ```

  A row is ~300 B; a busy agentic day (20k requests) is ~6 MiB, a week ~40
  MiB. `ttft_hist`/`decode_hist` are 24 log-spaced bucket counts as JSON.
- **Writer.** A `std::sync::mpsc::sync_channel(4096)` of `Record`s and a
  dedicated thread owning the connection (rusqlite is synchronous). The
  thread drains up to 256 records or 50 ms into one transaction (one
  `INSERT` per request plus the `sessions` upsert). `try_send` on the serving
  side; a full channel drops the record and bumps `usage_dropped`, exported
  in `/v1/stats` (`usage: {recorded, dropped, log_recorded, log_dropped,
  db_bytes, log_bytes}`) and shown on the dashboard. The channel bounds
  memory at 4096 × ~400 B.
- **Retention and caps.** Settings (in `settings`, editable from the
  dashboard, applied without restart through an `ArcSwap<Settings>` the
  writers and pruner read): `metadata_days` 7, `metadata_cap_mb` 256,
  `log_enabled` on, `log_hours` 24, `log_cap_mb` 1024, `daily_days` 90,
  `client_ip` off, `record_bench` on. The writer thread prunes every 60 s:
  roll the oldest expiring day into `daily`, `DELETE … WHERE ts_ms < cutoff`
  (index scan), then while `page_count × page_size` exceeds the cap delete
  the oldest 10% of rows, then `PRAGMA incremental_vacuum` so the file
  shrinks. `0` for a retention means off (no rows kept; the full log tier
  also stops recording). Tests inject the clock.
- **No long-term raw retention.** Only `daily` outlives the metadata
  retention: per-day counts, token sums and latency histograms by
  (protocol, client, model), 90 days by default, no ids, sessions or
  timings per request. Stated on the settings panel.

### Full-log tier (payloads, 1 day, separate file)

- **What.** `log(rid TEXT PRIMARY KEY, ts_ms INTEGER, protocol TEXT,
  request BLOB, response BLOB, bytes INTEGER, truncated INTEGER)`, index on
  `ts_ms`. `request` is the client body after redaction; `response` is the
  final response object (non-streaming shape): for streams the handler folds
  the deltas it already emits into text, reasoning, tool calls and usage, so
  the stored object is what a non-streaming call would have returned.
  Realtime stores the turn's input items and output items, not the audio
  frames.
- **Redaction, always.** Auth headers never enter the scope (the log stores
  no headers at all except `user-agent`). Media are stored as references:
  `data:` URIs and `input_audio` blobs become `{"type":"image_url",
  "ref":{"mime":…,"bytes":…,"sha256":…}}` (no bytes; TJ's images are
  re-sendable, and 8 MiB images × a day would dwarf the text). Tool calls,
  tool results, search queries and reasoning are payload and are kept. A
  record above `log_record_cap` (1 MiB) is truncated with a marker.
- **Async, bigger channel.** Its own `sync_channel(1024)` and its own writer
  thread, so a payload burst never competes with metadata. Admission is by
  count and by bytes (64 MiB in flight); over either, drop and count
  `log_dropped`. On the serving path the cost is one `Bytes` clone of the
  already-buffered request body (a refcount), and for streams a
  `push_str` per delta into a pre-reserved `String` behind one
  `AtomicBool` (`log_enabled`); redaction, folding and serialization run on
  the writer thread. Off means no clone and no push.
- **Retention.** `log_hours` (24) and `log_cap_mb` (1024), pruned by the log
  writer on the same 60 s tick; `log_enabled` off stops writes and leaves
  existing rows to expire, or the user clears them.
- **Clear now.** The dashboard's "Clear full log" (`POST
  /console/usage/log/clear`, cookie) makes the log writer `DELETE FROM log`
  + `incremental_vacuum` within one tick; with the server stopped, `rm
  ~/.cache/cuteafd/<instance>/usage/usage-log.sqlite*` is equivalent.
  "Clear everything" does both files.
- **Security.** Readable only through `/console/usage/log/:rid` behind the
  cookie; no `/v1/*` route reads it; an API key does not unlock it. README
  and the settings panel say: **with the full log on, user prompts and model
  outputs are stored in plain text for the retention period**; the
  coordinator logs one line at startup saying the full log is on and for how
  long.
- The metadata tier's guarantee is unchanged: payloads exist only in
  `usage-log.sqlite`.

- **Virtual sessions and delta entries (TJ, 2026-10-10).** Agent clients
  resend the whole conversation every turn, so a day of one session stores
  the same history hundreds of times. The log stores each turn as a delta
  against the earlier entry it extends:
  - **Matching.** Requests chain to the earlier request whose rendered
    prompt they extend. Two signals give the chain:
    - the engine's prefix-cache hit, whose snapshot already carries the
      session it was captured under (usage hooks, c8040142);
    - the handler's own sources: `previous_response_id`, `prompt_cache_key`,
      `metadata.user_id`, Realtime session ids.

    The full log keeps its own matcher on the redacted request's message
    list, not on tokens. That way a prefix-cache eviction or a server restart
    doesn't break the chain.
  - **Entry shape.**
    - When a turn's messages extend the parent entry's messages exactly, the
      entry stores `parent_rid`, the parent's message count, and only the new
      items (plus the response).
    - When the history was edited, truncated or spliced (a prefix mismatch),
      the entry stores the full message list and starts a new chain
      position. The divergence point is recorded, so the viewer can show what
      changed.
    - Tools, system prompt and settings are stored again only when they
      change.
  - **Viewer.** `/console/usage/log` groups entries by virtual session and
    shows a conversation view (each turn's new items and response in
    order), with edits marked where a chain restarted.
    `/console/usage/log/:rid` rebuilds the full request by walking parents.
  - **Retention.** Pruning never orphans a child. An entry whose parent
    expires is rewritten with its full history at prune time, or the chain
    is pruned as a unit. Pick whichever keeps the cap honest; the size cap
    counts stored bytes.
  - **Cost.** Matching runs on the log writer thread, keyed by a hash of each
    message prefix, so the serving path is unchanged. The full log stays off
    by default.

### Overhead bound (the C1 proof)

On the serving path, per request: four `Instant::now()` (arrival, admit,
first token, end; admit and first exist today in `Ticket`), filling a ~400 B
struct from values the handler already holds, one `try_send`, and in the
engine three relaxed atomic adds per decode round inside `Step::member`,
which already runs per round for the console. No SQLite call, no
serialization, no allocation beyond the scope, and no lock on the request
path. The CUDA thread touches only the `Ticket` counters. With the full log
on: one `Bytes` refcount bump and one amortized `push_str` per delta. The
gate is an identical-config A/B, tracker on vs `--usage off`, C1 code decode
and 8K prefill on matched prompts, three interleaved runs: emitted tok/s and
TTFT within noise. The bench's own requests are recorded (`bench=1`) and
hidden by default on the dashboard.

### Analytics: queries, not rollups

Everything on the dashboard except the live tiles is a SQL query over the
retained rows at request time; with the `ts_ms` index a 7-day range of tens
of thousands of rows aggregates in milliseconds, and the queries cache for
5 s. Percentiles come from Rust over the fetched column (SQLite has none).
The only rollup is `daily`. Routes (all cookie-gated, JSON, `range=1h|6h|24h|7d`
or `from,to`, filters `client,protocol,model,key,session,bench`):

| Route | Serves |
|---|---|
| `GET /console/usage/summary` | the KPI tiles: counts, token sums, error rate, cache hit, acceptance, p50/p95 TTFT and decode tok/s, with per-bucket sparklines |
| `GET /console/usage/series?bucket=1m` | time series by protocol, client or model: requests, tokens in/cached/out per bucket, concurrency max/mean, error count |
| `GET /console/usage/latency` | TTFT and decode tok/s histograms with p50/p95/p99, faceted by prompt length and cached fraction; prefill tok/s |
| `GET /console/usage/flow` | client → protocol → model matrix (requests and tokens) |
| `GET /console/usage/sessions` | session list with per-turn token strips; `/:id` a session's turns |
| `GET /console/usage/cache` | hit rate and tokens saved per bucket; TTFT by cached-fraction bucket |
| `GET /console/usage/speculation` | acceptance and tokens per round per bucket, by model and client |
| `GET /console/usage/errors` | outcomes and stop reasons per bucket; a (class, route, client) table with last seen |
| `GET /console/usage/requests?cursor` | the drill-down list; `/:rid` one record |
| `GET /console/usage/log/:rid` | the full-log record, when retained |
| `GET/PUT /console/usage/settings` | the tiers' settings and the files' sizes and drop counters |
| `POST /console/usage/log/clear`, `/clear` | clear the log, or both files |

### The usage dashboard (`/usage`)

Same shell as the console: `cuteafd-ui.css`/`.js` (`CuteUI.header` with a
third nav entry `USAGE`), `.tile`/`.panel`/`.track` styles, the palette
tokens, inline SVG only, no library and no CDN (the existing
`page_is_self_contained` test extends to it). `cuteafd-ui.js` gains five
primitives beside `sparkline`/`columns`/`timeline`/`meter`: `area` (stacked
series with crosshair and brush), `histogram` (log-x columns with percentile
ticks and a ghost overlay), `ribbons` (three-column flow with bezier
ribbons), `strip` (per-turn token blocks), and `heat` (hour × day grid).
Series colours by protocol: chat `--target`, messages `--prefill`,
responses `--accepted`, realtime `--grammar`, completions `--restore`;
cached tokens are always the series colour at 38% (as the KV meter does
today).

Page layout, top to bottom; every chart is a filter: clicking a series, bar,
node or row narrows the page's filter chips, and the drill-down list at the
bottom follows.

1. **Header row.** Range (1h / 6h / 24h / 7d / custom), split control
   (protocol | client | model), filter chips, `exclude bench` toggle, the
   unlock state, and `recorded N · dropped M` from `/v1/stats`.
2. **KPI tiles** (eight `.tile`s with sparklines): requests, tokens in /
   cached / out, error rate, TTFT p50 · p95, decode tok/s p50, cache hit %,
   acceptance %, concurrency now (live from the console feed).
3. **Flow over time** (full width `area`): stacked tokens/s (prefill, cached,
   output) coloured by the split, a request-rate line, and a concurrency
   band beneath (max/mean per bucket). Crosshair tooltip lists the bucket;
   brushing zooms and sets the range for the whole page.
4. **Latency** (two `histogram`s side by side): TTFT (log ms) and decode
   tok/s, p50/p95/p99 ticks, the previous period as a ghost outline; a row
   of small multiples by prompt length (<1K, 1–8K, 8–32K, 32K+). Hover: bin
   count; click: the requests in that bin.
5. **Prefix cache** (two panels): hit rate and tokens saved over time; and
   "what the cache buys": TTFT columns by cached fraction (0, <50%, ≥50%),
   median with p95 tick (the `columns` primitive with `median`).
6. **Speculation**: acceptance over time by model; tokens-per-round
   histogram; the same by client, since agentic clients accept differently.
7. **Clients → protocols → models** (`ribbons`): width by requests or tokens;
   a node click filters.
8. **Sessions**: a table sorted by last activity (client, model, turns,
   tokens, cache reuse %, span) with a `strip` per row: one block per turn,
   width ∝ input tokens, the cached part shaded, output in `--target`.
   Clicking a session opens its detail: the `timeline` primitive with turns
   on a time axis, TTFT and decode tok/s per turn as small columns, cache
   reuse per turn as a line, and the turn list; inferred (`prefix`) sessions
   are dimmed and labelled.
9. **Errors and stops**: stacked columns of outcomes over time; a `meter`
   of stop reasons; a table by (class, route, client) with counts and last
   seen, each row a filter.
10. **Requests** (drill-down): a cursor-paged table of metadata; expanding a
    row shows all fields; a `FULL LOG` tab renders the request and response
    (messages as text, tool calls as JSON, media as references) when the row
    is within log retention, else "expired"; locked viewers see the unlock
    prompt here.
11. **Settings** (cookie): per tier retention and cap, the full-log switch,
    client IP, record bench, with the file sizes and drop counters; the
    plain-text warning; "Clear full log" and "Clear everything" with
    confirmation. The panel is the only place these change.

Locked viewers see panels 1–9 and the metadata table; the full-log tab and
settings show "open the link from the run script's log to unlock". Live
tiles subscribe to `/v1/console/events`; the SQL panels poll their routes
every 10 s while visible. Sizing: primitives ~250 lines, page ~1,100 lines
of HTML/JS; L.

### Console access: the unlock cookie

- **Secret.** One per user per host: `~/.cache/cuteafd/console-secret`, 32
  random bytes hex, 0600, created by `release_prepare_secret` (the
  generalized `release_prepare_api_key`: mkstemp, owner and mode checks) the
  first time a launcher runs and reused forever after. Both launchers mount
  it read-only at `/run/cuteafd-console-secret` and pass
  `--console-secret-file`. Per host rather than per instance because
  browsers scope cookies by host, not port: two instances on raptor share
  the cookie jar whatever we do, so separate secrets would only mean
  re-unlocking; different hosts have different cookie domains and the file
  can be copied if wanted. Release and WIP launches behave the same.
- **Cookie.** `cuteafd_console=v1.<issued_unix>.<hex(hmac_sha256(secret,
  "console|" + issued))>`; `HttpOnly; SameSite=Lax; Path=/;
  Max-Age=31536000`, plus `Secure` when the request arrived over TLS or
  with `x-forwarded-proto: https` (the tailnet `tailscale serve` endpoint)
  or `--console-cookie-secure`. Verified with constant-time comparison
  (`auth::constant_time_eq`) and re-issued on any protected request when
  older than 30 days, so a browser that visits at all never re-authenticates.
  The cookie never holds the secret; `hmac` (RustCrypto, beside the existing
  `sha2`) is the one new crate.
- **Unlock.** `GET /console/unlock?token=<secret>` compares in constant
  time, sets the cookie and answers `303 /` with `Referrer-Policy:
  no-referrer` and `Cache-Control: no-store`. The handler never logs its
  query; there is no request-URI access log in the server today and the
  design adds none; the usage tracker records `/console/*` requests with the
  path only, never the query string. The launchers print
  `console unlock: http://<host>:<port>/console/unlock?token=…` once at
  ready time (host stdout, where `API ready at …` is printed; `CONSOLE_URL`
  overrides the base, e.g. the tailnet https name). The coordinator logs
  only `console: protected views unlock through the launcher's link`; the
  secret never appears in container logs, access logs or either database.
- **Rotation.** `scripts/launch/console-secret.sh rotate` (also `run.sh
  --rotate-console-secret`) rewrites the file; the coordinator re-reads it
  when its mtime changes (checked at most every 10 s), so old cookies fail
  within 10 s with no restart. A restart does not rotate.
- **Gate.** `ConsoleGate` middleware state (secret, mtime, verify) on
  `/usage`'s data routes, `/console/usage/*`, and any future admin route;
  pages themselves stay public and show the prompt when their data answers
  401 `{"error":{"type":"console_locked"}}`. The gate reads only the cookie:
  an API key does not unlock the console, and the cookie is never read by
  `require_key` (it reads `Authorization`, `x-api-key` and the WebSocket
  subprotocol only), so the cookie grants nothing under `/v1/*`.
- **Token text.** `ConsoleHub::text_enabled()` keeps its meaning (the
  server's switch or a bench run), but text now reaches only unlocked
  viewers: when text is on and viewers exist, the worker publishes each
  frame twice (with and without `text` pieces) and the hub's socket and SSE
  handlers pick the variant by the connection's unlock state, checked once
  at connect. The snapshot's `text` becomes `"on" | "locked" | "off"`; the
  TEXT button reads `TEXT (unlock)` when locked and links to the prompt. The
  bench exception stays: while a run holds the server, the only text is
  the bench's synthetic prompts, so `set_bench_active(true)` makes text
  public as today. `CONSOLE_TEXT=on` therefore no longer exposes sessions to
  anyone on the port; its warning in `run-family.sh` goes, and it can
  default to on (viewer-count gating keeps the cost zero while nobody
  watches).
- **Public without a cookie**: `/`, the console's time view, `/bench` as
  today (its controls keep the API-key rule), `/health`, `/v1/*` under their
  own key.

### Implementation steps

1. **Metadata tier core (S).** `cuteafd_api::usage` (record, scope,
   middleware, sink), `cuteafd-usage` (schema, writer thread, settings,
   pruning, `/v1/stats` counters). Gates: unit tests for pruning by age, the
   size cap and `incremental_vacuum` with an injected clock; overflow drops
   and counts; the privacy test (sentinel payload never in the metadata file
   bytes); the settings round-trip.
2. **Chat path hooks (S).** `openai.rs` fills the scope; `NativeRequest`
   carries the handle; `Ticket` and `Step::member` write admit, first,
   retire and round counters; V4.1's console module does the same. Gate:
   the C1 overhead A/B above, and a test that a cancelled stream still
   yields a row with `outcome=cancelled`.
3. **Gateway hooks (M).** `TurnRequest` carries the handle; Messages,
   Responses and Realtime front ends set protocol, models, session and
   counts; the driver fills usage and stop reason; `prompt_cache_key`,
   `metadata.user_id` and `previous_response_id` session inference; the
   prefix-chain fallback through the handle. Gates: replay fixtures produce
   the expected rows per protocol; session-inference unit tests per source.
4. **Console secret and gate (S).** `release_prepare_secret`, mounts and
   flags in both launchers, `/console/unlock`, `ConsoleGate`, rotation
   script and mtime reload, the printed link. Gates: auth tests: constant
   time, wrong and stale tokens refused, cookie set with the right
   attributes and `Secure` behind `x-forwarded-proto`, renewal after 30
   days, rotation invalidates, an API key does not unlock, a cookie does
   not authorize `/v1/*`, the unlock query never appears in logs (a tracing
   capture) or in the usage DB.
5. **Per-viewer token text (M).** Dual frames, unlock state per connection,
   snapshot `text` tri-state, page changes. Gates: worker tests for the two
   variants; a locked SSE client never receives a `text` piece while an
   unlocked one does; the bench exception test in `cuteafd-bench` still
   passes.
6. **Full-log tier (M).** Log channel, writer, redaction, stream folding,
   retention hours, cap, clear, startup line, README warning. Gates:
   privacy tests (auth headers and the raw key never in the log file;
   `data:` URIs replaced by references; a record above the cap is
   truncated); retention by hours and cap; `log_enabled=false` leaves
   `usage.sqlite` untouched (file hash before and after); clear empties the
   log within one tick and the metadata row survives.
7. **Dashboard queries and routes (M).** The table of routes above, with
   percentiles and histograms in Rust, 5 s cache, cursor paging. Gates:
   query tests on a synthetic 7-day DB (known percentiles, buckets, session
   aggregates); every route 401 without the cookie.
8. **Dashboard page (L).** The five primitives and the eleven panels. Gates:
   `page_is_self_contained` for `/usage`; a headless render smoke test
   against the synthetic DB; the locked and unlocked states.
9. **Daily rollups (S).** Roll before prune, the 90-day cap, the
   long-range view switching to `daily` beyond the retention. Gate: a rolled
   day's sums equal the rows it replaced.
10. **Docs and launch polish (S).** README section (what is stored, where,
    how long, the plain-text warning, how to clear and rotate), the
    launcher's ready block prints the unlock link, `cuteafd usage
    {clear,clear-log,export-csv}` for use without a browser.

Order: 1, 2 and 4 are independent and start first; 3 after the gateway
front ends land; 5 after 4; 6 after 1; 7 and 8 after 1 and 4; 9 and 10 last.

### Decisions (TJ, 2026-10-10)

- **Full log is on by default.** Users with a heavy workload turn it off, or
  set its retention or cap to zero.
- **Media are stored in full** behind a flag (`log_media`, default on), not
  only as references. Each image and audio blob is written once as a file in
  the persistent store, under a content-addressed layout:
  `usage/media/<sha256[0:2]>/<sha256>.<ext>`, deduplicated by hash. Log
  entries reference it by hash. Pruning deletes a file when no retained entry
  references it. Media bytes count toward the log's size cap, and a cap of
  zero (or the flag off) stores references only.
- **Bench requests:** the benchmark dashboard has an option to include or
  skip them. Default is skip.
- **Expired parents:** when a chain's base expires, the oldest retained
  entry is rewritten with its full history and becomes the new base.
- **Edited turns:** the entry stores the full history, marked at the point
  where it diverged.
- **Console secret:** one per host. The box has one user.
- **Not raised by TJ, so the recommendations stand:**
  - `CONSOLE_TEXT` defaults on once text is cookie-gated;
  - client IP off;
  - 90-day daily aggregates;
  - the unlock link in browser history is accepted, with rotation as the
    remedy.

### Open questions for TJ, each with a recommendation (answered above)

1. Full log on by default (1 day) or off until switched on? Recommend on,
   with the startup line and the panel warning; it is what was asked for and
   the box is single-user.
2. One console secret per user per host (recommended, see above) versus per
   instance.
3. `CONSOLE_TEXT` default on once text is cookie-gated? Recommend on.
4. Client IP: off by default (recommended); when on, the full address.
5. Keep 90-day `daily` aggregates (recommended; a few KB per day, no ids),
   or keep nothing past 7 days?
6. Media in the full log as references (recommended), or bytes under a
   separate small cap?
7. Record the bench's requests (recommended, flagged and hidden by default)
   or skip them?
8. The unlock link lands in browser history as a GET; accepted, with
   rotation as the remedy. A POST form would need a page before the cookie.

## v3 draft policy: resource-priced, shared (design, 2026-10-09)

Design for v3 item 6. Inputs: `cuteafd-core::dspark_policy` (1,214 lines) and
its V4.1 binding (`v41_native_serve/speculative/policy.rs`), today's
`shared/draft_policy.rs` (`CycleCost`, `allocate`, `DraftHistory`,
`Calibration`), work/draft-policy-v2 (5b99895e: concurrency buckets, keyed
`SelectorFit`/`ConfidencePolicy`, `compete_copies`; parked), the Astra review
(`builds/draft-policy-review/REPORT.md`), the glmf-defaults-on evidence and
three read-only engine audits (GLM 5.3/Flash, MiMo/Qwen, V4.1) relayed in
this design's branch report. Nothing here ran on hardware.

**Why V4.1's model transfers across concurrency and `CycleCost` does not.**
`CycleCost` fits `a + b*table(rows) + c*(sequences-1)`. At C16 the verify step
is sub-additive in rows (identical and similar requests share expert reads),
so the fitted row slope `b` is small; a C1 request arriving inside the ~50-step
forgetting window is then priced with that slope and buys ~5.4 drafts where
3.2 pay (C1 -9.8%, glmf-defaults-on). The sub-additivity is a property of the
*traffic*, not of the row count, and V4.1's model prices it where it lives:
per layer `alpha + beta*rows + bytes/bandwidth`, with bytes forecast from the
union of the step's routes. `beta` and `1/bandwidth` are physical and the same
at C1 and C16; a C16 step costs less per row because its route union has fewer
new expert groups per row, not because rows got cheaper. The per-concurrency
buckets on work/draft-policy-v2 patch the symptom (one fit per regime); the
resource-priced model removes the cause. No buckets in the target design; if
the C1-after-C16 gate still fails for a family, add buckets to the *round
residual* only, never to the layer fits.

### 1. The shared core

`cuteafd-core::dspark_policy` moves to `cuteafd-core::draft_policy` and takes
its geometry as a value. It is CPU-only and does no device work, as today.
Everything below is the existing algorithm with constants turned into inputs;
the first binding (V4.1) must make byte-identical decisions on recorded
observations.

```rust
pub struct PolicyGeometry {
    pub layers: Vec<LayerResource>,       // every backbone layer, in order
    pub experts: u16,                     // routed experts per layer (<= 65_535; routes are u16)
    pub topk: u8,
    pub max_requests: usize,              // the lane's row budget in requests (V4.1 16, GLM Flash DECODE_ROWS)
    pub max_positions: usize,             // drafts per request (V4.1 7, DFlash block-1, MTP depth)
    pub widths: Vec<usize>,               // draft widths the drafter can switch between; one = fixed
    pub classes: Vec<ResourceClass>,      // fitted separately; usually [LocalRtx, Spark]
    pub regimes: usize,                   // lane regimes (V4.1 solo/shared = 2; single-lane loops 1)
}
pub struct LayerResource {
    pub class: Option<u8>,                // None: dense or untimed layer (no routed traffic)
    pub slice_bytes: f64,                 // bytes one device reads for one expert's gate+up+down slice, this format and TP
    pub group_rows: u8,                   // rows per weight-read group of the installed kernel (V4.1 slice kernels 16)
    pub timed: bool,                      // has a preceding boundary event; layer 0 is false (its time is in the residual)
}
pub struct ResourceClass { pub label: &'static str, pub prior_us_per_mb: f64 }  // weak physical prior only
```

- **Cost model (unchanged).** Per timed layer `alpha_c + beta_c*rows +
  MB/bandwidth_c` for its class `c`; per round `A + B*rows + C*requests`; per
  draft pass `D + E*requests + wide*(F + G*requests)`; Huber-weighted,
  exponentially forgotten, nonnegative least squares with weak priors; a mean
  residual bias per regime. Traffic of a layer is `slice_bytes * sum_e
  ceil(routes_e / group_rows)` over the step's route multiset.
- **Route history and forecast (unchanged).** 24 committed tokens per request,
  four shifted stand-in windows, novelty rate for short histories. Storage is
  `Vec<u16>` of `layers * topk` per token, so Qwen's 512 experts fit; the
  `& 511` decode moves into the binding.
- **What stays V4.1-only inside the core:** nothing. The `(5, 7)` widths, the
  "native block" boundary (`widths[0]`) for request-local outcomes, 40 layers,
  384 experts, top-6 and group 16 are the V4.1 `PolicyGeometry`.
- **What stays family code:** the dSpark drafter and its three windows, taps
  and RNG; the DFlash2 selector kernel; MTP stage execution; the copy-window
  *search* (indexed 8-gram, longest-backward); Engram, CED replay and mHC
  (prefill and model arithmetic, not round pricing).

### 2. Engine plumbing, every family

Three signals per round, all taken where the engine already has the data.

**Per-layer time: CUDA events between graph launches, not host marks.** Every
generic engine already calls `console::layer_mark(index)` once per layer
(GLM 5.3 `engine.rs:913,1166`, GLM Flash `:3008,3134`, MiMo `:1506,1639`,
Qwen `:1951,2116`, V4 `:949`). Those are host clocks, armed only while a
console viewer is connected, and on a local-expert layout they measure launch
enqueue, not GPU time. V4.1 instead records one timing event per layer on the
FFN stream after each layer's FFN finish (`v41_backbone_lane.rs:923,1017`),
outside the stage graphs, and reads 39 `cudaEventElapsedTime` pairs once per
round after the round's own drain (`independent.rs:262`). The shared version:
- `shared/draft/clock.rs::LayerClock`: `layers + 1` timing events allocated
  once per lane/rank; `mark(layer)` records on the engine stream at the
  existing `layer_mark` site (which keeps feeding the console); `read()` after
  the round's existing completion sync returns `Vec<Option<f64>>` µs.
  Recording between graph replays does not touch capture: GLM 5.3 and GLM
  Flash run one graph segment per layer (`glm5/engine.rs:1043`, `glm5_flash/
  engine.rs:3030`), so the marks fall between segments exactly as V4.1's do.
  Where a graph spans several layers the engine records the event inside the
  capture (legal; the event object must outlive the graph), and `read()` is
  still one pass after the step.
- Boundary semantics are "whatever the engine's segment boundary is": GLM's
  segment `l` reduces layer `l-1`'s experts and runs layer `l`'s attention,
  router and exchange; the fit attributes layer `l`'s traffic to the interval
  that contains its exchange. The residual absorbs the small shift. V4.1's
  distributed exception (a layer after a GPU handoff is timed from input
  arrival, omitting the hop) stays as is; the hop is in the residual.
- Cost: ~`layers` event records and elapsed queries per round, no added
  synchronization. V4.1 pays this today at C1 187 tok/s. The step's A/B
  measures it on each family; if it shows, read every Nth round.

**Routes: copy ids where they are already on the host; one async D2H ring
where they are not.**
- Spark layers: GLM 5.3, GLM Flash, MiMo and Qwen already stage `u32 ids +
  f32 weights [rows, topk]` to pinned host memory per MoE layer to build the
  `ExpertProtocolV2Request` (`glm5/engine.rs:1366`, `glm5_flash/engine.rs:
  3753`). `RoundRoutes::push(layer, ids)` copies the ids (`rows*topk*2` B as
  u16) into a per-round host buffer before the staging is reused. Zero D2H,
  as V4.1 (`v41_backbone_lane.rs:423`).
- Local RTX layers (GLM Flash `Experts::Local`/`LocalExl3` `engine.rs:3708`,
  MiMo local, Qwen local, V4 local, V4.1 local/TP2): the router's ids never
  leave the device. Add V4.1's mechanism (`v41_backbone_router.rs:680`): a
  24-48 B/row D2H of the ids per MoE layer, queued on the engine stream right
  after the router into a per-layer pinned host ring, decoded after the
  round's existing sync. No per-layer host wait. Where the router runs inside
  a multi-layer graph, the router kernel writes a layer-indexed ids slot and
  one D2H after the step moves the whole `[layers, rows, topk]` array.
- Committed routes: after acceptance, keep each request's accepted-input
  prefix only (`offset..offset+accepted`, then `offset += rows`), as the V4.1
  binding does; the newly emitted bonus token's routes enter history when it
  is verified as the next anchor. Transport descriptors use step row indices,
  so the binding maps rows to requests through the scheduler's sequence spans,
  not `source_request_id`. Target and draft passes stay distinct; MTP draft
  stages that route through experts (Qwen) are not backbone traffic.

**Round clocks: one shared definition.** `RoundClock { round_start,
draft_start, draft_end, observe }` on the host `Instant`, with `total_us =
observe - round_start` and `draft_us = draft_end - draft_start` bracketing
the draft call alone (sequence collection, copy lookup and length selection
are round work, not draft work). This normalizes V4.1's two boundaries (the
serial path times copy lookup and selection inside `draft_us`, `scheduler.rs:
1346`; the independent path does not, `speculative.rs:465`) and the GLM
difference (GLM 5.3's verify bracket includes `selector.select`, GLM Flash's
also includes grammar setup). Verify time is not a separate observation: it is
the layer sum plus the residual.

**Per family, where each signal comes from today:**

| family | layer events | remote routes | local routes | draft clock | round clock | notes |
|---|---|---|---|---|---|---|
| V4.1 | has them (events, `layer_us`) | has them | has them (ring) | has (two boundaries) | has | binding moves onto the shared types; decisions unchanged |
| GLM Flash | add at `layer_mark` sites; one segment per layer | staged per layer (`spark_dispatch`) | add the ring (fp8moe and EXL3 local) | `serve.rs:1133` (bracket narrows) | add | DFlash2 and dSpark drafters; `observe_host` never called today |
| GLM 5.3 | same structure as Flash | staged per layer (`moe_stage`) | none (all routed layers remote) | `serve.rs:751` | add | shares `glm5/dflash_policy.rs` with Flash |
| MiMo | add at `layer_mark` sites; one segment per layer, routed experts between segments (`engine.rs:1581`) | staged per layer (`stage_routes`/`spark_send`, `engine.rs:2515,2594`) | add the ring (fp8moe local) | `serve.rs:956` (DFlash), `:1000` (MTP, all stages in one bracket) | add | priced by the GLM adapter with the Pro TP6 table today (`serve.rs:528`); DFlash selector features computed and ignored; MTP stages are dense SWA blocks, ids only |
| Qwen | add at `layer_mark` sites; per-layer segments, `moe_front` outside the graph in bucket mode (`engine.rs:2035`) | staged per layer (`spark_moe`, `engine.rs:2382`) | add the ring (FP8/EXL3 local) | `DraftTiming` steps around `speculate::draft` (`serve.rs:997`) | has `cycle_ms` in its trace | chained MTP, depth chosen before drafting; MTP experts route at index `cfg.layers`, coordinator-local even with a Spark backbone |
| V4 | add at `layer_mark` site | staged | add the ring | `StepShape.draft_us` | add | fixed dSpark block today; first a fixed-width binding, adaptive is its own gate |

**Geometry per family** comes from the loader, not constants: `slice_bytes`
from `Fp8Layer::bytes_for` per rank (`shared/experts/fp8.rs:169`: FP8
`3HS + 3*ceil(H/128)*ceil(S/128)*4`, MXFP4 `3HS/2 + 3HS/32`, NVFP4 `3HS/2 +
3HS/16 + 24`, with `S` the rank's *actual* stored width: TP6 exact slices are
384/384/384/384/256/256, not `I/6`; MiMo MXFP4 is 12.75 MiB per expert at
TP1 Flash, 19.125 MiB Pro, 3.19 / 4.78 MiB at TP4), from the EXL3 manifest's
per-projection
tiers (`v41_exl3_residency.rs:151`: `H*S*(Kg+Ku+Kd)/8` payload plus the
rotations actually read, not the arena size), and from V4.1's
`expert_slice_bytes` as today. The class per layer follows the placement
(`ExpertHome::{RtxTp2, RtxWhole, Spark}` from the v3 solver once it lands;
until then the family's `Experts` enum or `has_local_layer`). `group_rows` is
the installed kernel's row tile: 16 for the CuTe grouped slice kernels, read
from the AOT manifest where it records one; a wrong constant is partly absorbed
by the fitted bandwidth, and the per-family step's A/B is the check.

### 3. Calibration and drafters: one chain, one online owner

Every drafted position carries one piece of **evidence** and the policy turns
it into a conditional acceptance probability in three fixed stages. The chain
is the same for every family; only the first stage differs by drafter.

1. **Evidence (family).** `Evidence::Head(p)`: a trained confidence head's
   sigmoid (dSpark on V4.1 and GLM Flash: one logit per position from
   `confidence_head.proj`, `v41_dspark.cu:38`). `Evidence::Selector([margin,
   p_top, entropy, rank])`: DFlash2's 16-candidate selector features
   (`glm_dflash.cu:438`), no probability of its own; MiMo's DFlash already
   computes them (`mimo_v2/dflash.rs:804`) and discards them, so MiMo gets a
   selector prior under its own key once fitted. `Evidence::History`: MTP
   (Qwen; MiMo native MTP returns ids only), nothing but the request's past
   outcomes.
   `Evidence::Copy { match_len }`: a copy span.
2. **Prior (shipped per (family, drafter, numerics) key).** Maps evidence to
   `p0` per position. For `Head` it is the identity on the logit. For
   `Selector` it is the keyed logistic fit over `[history_logit, log1p
   margin, logit p_top, entropy, log1p rank, position/7]` from work/draft-
   policy-v2's `SelectorFit` (`shared/draft_confidence.rs`), today's frozen
   glmrt fit as the `generic` key. For `History` it is `DraftHistory::pooled`
   (3-in-4 prior, pooled trials capped at 8). For `Copy`, a per-match-length
   table with its own history. This is PLAN "First after rc3" items 1-2: the
   key selects the file, the fit is the prior, and `CUTEAFD_DRAFT_CONFIDENCE_
   KEY`/the launcher spelling stay as v2 defined them.
3. **Online calibration (shared, one owner).** V4.1's per-position Platt
   scaling on the prior's logit (`logit' = a*logit + b`, Newton step on a
   decayed Fisher matrix, slope clamped to `[0, 3]`, memory ~500 reached
   samples; `dspark_policy.rs:232`), one `Platt` per (source, position), per
   deployment, never persisted. It absorbs target quant, drafter numerics and
   content mix. The shared affine `Calibration` (Qwen) and v2's
   `ConfidencePolicy::apply` correction are deleted: two online corrections on
   one signal double-count. Positions reached fewer than 32 times use the
   pooled Platt of their source (Qwen's trace: per-position with 16-outcome
   histories was worse than pooled; the shrinkage is what makes per-position
   safe).
4. **Request-local outcomes past the trained block** (V4.1 `outcomes`,
   decay 0.7, prior weight 2): applied to `Head` evidence only, beyond
   `widths[0]`, where the head saturates. For `Selector`/`History` the
   request's history is already the evidence.

**Censoring is explicit.** Only positions whose predecessors were all
accepted carry evidence; grammar truncation, EOS, output limit and
cancellation are censor reasons, not misses. GLM Flash and MiMo record
`(planned, accepted)` after grammar truncation today (`glm5_flash/serve.rs:
1232`, `mimo_v2/serve.rs:1013`), which books a miss the verifier never saw;
the shared observation takes `executed` and `accepted` and a reason. Copied
rows train costs and route history, never neural evidence (today's binding,
`policy.rs:44`).

**Rules kept as adapter data until measured away:** GLM's cold five drafts
for four cycles and the lone-request five-draft reference (`dflash_policy.rs:
204`; 8c800f8f lost 144.6 -> 132.0 tok/s when the head blend changed), GLM
Flash's dSpark head/history blend at 0.75, Qwen's one-draft probe after
prolonged zero plans, `DraftSkip`. Each is removed only by its own A/B.

### 4. Allocation

`select_core` (forward growth over every request's next row, best expected
committed tokens per predicted µs, continuing through temporary losses; exact
for one request, `dspark_policy.rs:647`) is the allocator. Changes:
- **One global row budget replaces equal quotas.** GLM Flash, MiMo and Qwen
  reserve `DECODE_ROWS / active - 1` drafts per request before allocating
  (`glm5_flash/serve.rs:1061`, `mimo_v2/serve.rs:810`, `qwen4/serve.rs:956`),
  so a confident request cannot use rows a hopeless one leaves. The allocator
  gets one anchor per request plus `max_rows` and the per-request
  remaining-output cap. This is a behavior change with its own gate (C16 on
  heterogeneous prompts), separate from the cost-model swap.
- **Identical sequences need no duplicate-row price.** Two requests with the
  same tokens at the same position have the same routes, so the route union
  adds no traffic for the second; only `beta*rows` remains. `Group.members`
  and `duplicate_row_ms` go; GLM's `(position, digest)` grouping stays as the
  way the binding tells the forecast two requests share history.
- **Pre-draft action, one function.** `choose_action` generalizes
  `choose_width`: dSpark picks a width in `widths`, a block drafter picks
  draft-or-skip (`DraftSkip`), a chained drafter (Qwen MTP) picks the depth
  cap, scoring each candidate with `select_core` under that candidate's draft
  cost (today's `allocate` loop over caps for `Drafter::Chain`). Width
  exploration (a width unused for 64 rounds runs once) applies to every
  multi-valued action.
- **Copies extend the neural draft; they don't compete with it** (TJ,
  2026-10-10; replaces the earlier two-candidate design).
  - **The problem with competition:** a per-match-length acceptance table
    can't be compared meaningfully with a calibrated neural confidence.
  - **Agreement:** the neural drafter proposes its window as usual. A copy
    span qualifies only if its start agrees with the neural draft over the
    whole window. That agreement is the confidence: the target model's own
    drafter independently predicts the copied tokens, so no separate copy
    table is needed, and short coincidental matches drop out.
  - **Extension:** on agreement, the copy's continuation is appended past
    the drafter's horizon (DFlash2 7, dSpark 8, MTP 3), and the target
    verifies the whole sequence in one round. Inside a long copy stride
    (file rewrites, echoed diffs, quoted code), a round can verify 16-64
    tokens instead of 7-8.
  - **Tail confidence:** positions beyond the drafter use a per-position
    decay owned by the online calibration (Platt on "agreed copy, position
    k beyond the drafter"), seeded from this request's earlier copy
    outcomes.
  - **Length:** the resource model prices the extra rows, so the extension
    length is the policy's usual expected-tokens-per-time decision. A
    higher per-row cost (the head split on max) shortens it.
  - **Later, measured:** relaxing agreement to a prefix of the window;
    suffix-index drafting over the session history (v3.x).
  - **Replaces** v2's `compete_copies` and `Evidence::Copy`'s per-length
    table.
- **Expected tokens per predicted time stays the objective.** The long-run
  `E - R*T` experiment lost 2-6% on Qwen (f32e22e3); not revived.

### 5. Migration

| step | what | size | gate |
|---|---|---|---|
| D0 | `dspark_policy` -> `draft_policy` with `PolicyGeometry`; V4.1 binding passes its geometry; `u16` routes, `max_requests`, `max_positions`, `widths`, `regimes` as inputs | S | unit: identical decisions and fits on recorded V4.1 observations (replay the existing tests through the geometry); V4.1 golden byte-exact; one quick A/B pair on V4.1 min |
| D1 | `shared/draft/{clock,routes,evidence,binding}.rs`: `LayerClock`, `RoundRoutes`, `RoundClock`, `Evidence`/`DraftSource`, the observation builder with censor reasons; V4.1 binding moves onto them (its serial path adopts the independent `draft_us` boundary) | M | tests; V4.1 golden; A/B min (shared hot path: full 3-session parity is the release-cut gate, not this step's) |
| D2 | **GLM Flash onto the shared policy** (first family: the failure was measured here, it has per-layer segments, both drafter kinds and both expert homes). Events at `layer_mark`, remote routes from staging, local ring, geometry from `Fp8Layer::bytes_for`/EXL3 manifest, `Selector` prior from v2's `SelectorFit`, Platt online, same quotas as today | L | matched prompts (`wip-cards.py --interleave --matched-prompts`): C1 on a fresh server, **C1 after a C16 sweep** (the glmf-defaults-on failure case, decode rows 128 and 64), C4, C16, emitted tok/s, on min and max, EXL3 and FP8; C1 >= 0.99 and C1-after-C16 >= 0.99 of fresh C1 |
| D3 | Global row budget and copy competition on GLM Flash (behavior change) | M | same card; C16 heterogeneous prompts is the metric; C1 unchanged |
| D4 | **V4 first** (TJ, 2026-10-11): fixed-width binding, then adaptive width priced under P4's TP2 placement; cards Flash/Pro min and max at 2M, Pro min code C1 included. Then MiMo (DFlash + MTP block), GLM 5.3 (shares the GLM binding), Qwen (chain depth as the pre-draft action) | S, M, S, M | per family: the D2 card on its min/max |
| D5 | Delete `CycleCost`, `Calibration`, `allocate`, the v2 buckets, `glm5/dflash_policy.rs` planning, per-family copy-length loops; `shared/draft_policy.rs` keeps `DraftHistory` and the trace path only | S | tests; failing ids unchanged |

**Delete as you go (TJ, 2026-10-10).** The old per-family policies are ad
hoc code, so they don't survive behind a switch:
- **D3 / D4:** each family's port deletes that family's old policy path in
  the same PR that makes `shared` its default, and the launcher rejects the
  old key. No fallback onto `CycleCost` anywhere, including cold start:
  priors are seeded from the geometry and resource classes.
- **D5:** only removes what is left once no family uses it: `CycleCost`,
  `Calibration`, `allocate` and the v2 buckets.
- **The D2 `GLM5_FLASH_DRAFT_POLICY` switch** is temporary; it goes in D3.

**D2 outcome (2026-10-10, work/v3-d2; opt-in `GLM5_FLASH_DRAFT_POLICY=shared`).**
GLM Flash's geometry: 45 layers, 3 dense (`class: None`), 42 MoE layers
`remote` (Spark) or `local` (coordinator FP8/EXL3), `slice_bytes` from
`Fp8Layer::bytes_for` (widest rank / experts) or the EXL3 manifest's
per-projection tiers (payload + rotations over the widest rank's slice),
`group_rows` 16, one regime, the drafter's block as the single width, the
step's row budget as `max_requests`. Layer events at the `layer_mark` sites,
Spark ids from the staging, local ids through a pinned D2H ring behind the
router (checked on hardware against the staged ids: 48,426 layers, 0
mismatched), DFlash2 through the keyed `SelectorFit`, dSpark through its head
blend, Platt in the core, executed rows with EOS/output-limit censoring, the
draft bracket around the draft call. The cold-five and lone-five DFlash2 rules
stay as adapter data. Planning costs 36-45 µs per round (max 2.2 ms at C16);
the events and ring do not move the verify step.

Matched-prompt pairs on one sealed build (`--arm-wip`, `cycle` vs `shared`),
rc3 kit cards at C16 admission; the C1-after-C16 point is the code request of
the `decode_content` panel run right after the C1..C16 sweep. Paired medians
shared/cycle:

| card | pairs | fresh C1 | sweep C1 | C1 after C16 | C4 | C16 | emitted |
|---|---:|---:|---:|---:|---:|---:|---:|
| EXL3 min, rows 64 | 1 | 1.041 | 1.020 | 1.114 | 1.003 | 1.019 | 1.017 |
| EXL3 min, rows 128 | 3 | 0.961 | 1.019 | **1.313** | 0.979 | 1.057 | 1.039 |
| FP8 min, rows 64 | 3 | 0.954 | 1.042 | 1.017 | 0.958 | 0.982 | 1.039 |
| EXL3 max, rows 64 | 3 | 0.996 | 0.956 | 1.021 | 1.008 | 0.983 | 1.059 |
| FP8 max, rows 64 | 3 | 1.057 | 1.153 | 1.046 | 1.027 | 1.004 | 1.078 |

The glmf-defaults-on failure reproduces on `cycle` (EXL3 min rows 128: C1
after C16 / sweep C1 0.712, 0.901, 0.701) and `shared` removes it (0.975,
0.979, 0.935). "Fresh C1" is the server's first 320-token code request, ~14
rounds after start, while the shared fits are barely warm; single-request
points swing ±6-10% between same-arm repeats. C4 is one 4-wide wave whose
aggregate follows its slowest member (±7% same-arm); per-request C4 is
1.02 median. On max the fitted per-layer row slope is twice min's (11.0 vs
5.7 µs/row/layer: the head split adds attention and exchange per row), and
shared verifies slightly shorter C1 prefixes there (sweep C1 0.956).
FP8 post-sweep decode drops on both arms alike (server state, not policy).
Fidelity, cache and speculation checks pass or report near-ties identically
on both arms. Default stays `cycle` for now: fresh C1 on the min cards (first
cold request) and sweep C1 on EXL3 max sit at 0.95-0.96, under the 0.99 bar.
The no-Spark (2 RTX) and mixed-placement edge cards need P6/P7 and gate D3/D4;
GLM Flash experts (115.6 GiB at K3.25) do not fit one RTX.


**V4.1 moves byte-exactly** because D0 and D1 change types, not decisions: the
geometry reproduces its constants, the clock reproduces the independent
path's boundaries, and the binding's copy handling is unchanged. The serial
single-lane path's `draft_us` boundary narrows (D1); that changes a fit input,
not an output token, and the quick A/B on min covers it.

**Interim.** work/draft-policy-v2's keyed `SelectorFit` and
`compete_copies` are reused by D2/D3 as the prior and the copy search; its
`CycleCost` concurrency buckets are superseded by D2 and need no hardware gate
of their own (decode rows 128 stays opt-in until D2's C1-after-C16 card).

### Risks

- **Event cost at C1.** ~45 records and reads per round on GLM Flash. V4.1
  pays it; D2's card measures it. Fallback: read every Nth round.
- **Local-route ring.** One small D2H per MoE layer on the engine stream. If
  a family's graph covers several layers, the layer-indexed slot variant is
  required before adoption; a per-layer host wait is never acceptable.
- **Identifiability.** Rows and MB are correlated; the weak priors resolve
  the near-null direction and predictions stay stable even when individual
  coefficients wander (the Astra review's point). Judge by prediction error
  per shape and the card, not coefficients.
- **Warm-up.** Until the fits are warm (12 rounds, 120 layer samples per
  class) every available draft is verified, as on V4.1. For DFlash's 7 drafts
  on a short C1 request that is the first ~12 rounds; D2's card includes
  short requests.
- **Kernel tiles.** `group_rows` for fp8moe/EXL3/NVFP4 kernels may not be 16;
  a wrong tile biases the fitted bandwidth, not the decisions' ordering.
  Record the tile in the AOT manifest where missing.
- **Determinism.** Adaptive lengths change verify arithmetic (WP-9). This
  design keeps adaptivity; a deterministic cohort mode is after v3.
- **Behavior drift masked as a model swap.** D2 swaps the cost model under
  today's quotas and GLM rules; D3 changes the quotas. Keep them separate so
  a regression names its cause.

### Decisions (TJ, 2026-10-09: "The fable plan 6 questions sound good")

1. **First family:** GLM Flash, MiMo next.
2. **Always-on layer events** on every family, as V4.1 does; cost measured in
   D2, with every-Nth-round reads as the fallback.
3. **One online calibration owner:** per-(source, position) Platt with pooled
   shrinkage; the shared affine `Calibration` and v2's online correction go.
4. **Global row budget** replaces equal per-request quotas (D3), gated
   separately on C16 heterogeneous prompts.
5. **Interim buckets:** no hardware gate for v2's `CycleCost` buckets; merge
   v2's opt-in `SelectorFit` and `compete_copies` on CPU gates, since D2/D3
   reuse them.
6. **Earlier decisions stand:** V4.1's 20/20 split and no unified serve loop
   (three family loops carry the D1 binding).

## First after rc3: per-key draft confidence calibration (TJ, 2026-10-09)

Items 1-2 are the prior stage of the v3 draft policy's calibration chain
(section 3 above) and are built on work/draft-policy-v2 (`SelectorFit`,
`ConfidencePolicy`); item 4's buckets are superseded by that design's `D2`.

The shared draft policy refines acceptance with one frozen logistic fit over
the selector features (margin, top probability, entropy, rank), fit in glmrt
on 32 GLM-5.3 K4 requests (`glm5/dflash_policy.rs`). GLM 5.3, GLM Flash and
MiMo DFlash all reuse it. Feature distributions depend on the target model
and quant, the drafter, and the drafter's numerics, so other combinations are
miscalibrated. Measured: GLM Flash with the tensor-core head and W8A8 drafter
kept acceptance (67.1% -> 66.5% median) but the policy chose ~20% shorter
verify prefixes (tokens/step 2.6 -> 2.1), C1 -12% despite ~7% cheaper steps.
Option isolation then showed the drafter options alone are not that cause
(tensor head + W8A8 alone: C1 86.0 vs 84.3, acceptance 69.9%); the
calibration gap remains real but is not this regression.

1. One fit per (family, drafter, drafter numerics) (TJ: avoid per-quant
   fits). Numerics is a short key such as `bf16-r1` or `fp8-w8a8-r1`; bump the
   revision on any major change to the drafter's math (head kernel,
   accumulation order, quantization). ~32 fixed-K7 requests on one
   representative quant, 6 coefficients shipped beside the drafter config;
   generic fit plus a log line when the key is absent. The selector
   features are the drafter's own outputs, so target quant mostly shifts the
   acceptance level, which step 2 absorbs.
2. Online refinement always on: the per-drafter fit is the prior,
   coefficients updated from verified outcomes with forgetting and clamped
   slopes (as Qwen's `Calibration`), absorbing target quant and drafter
   numerics (tensor head, W8A8) per deployment.
   Check once: fit on one quant, measure log loss and C1 on another quant and
   with the drafter options on; add a narrower fit only where online
   refinement does not close the gap within a few hundred drafts.
3. Gate: emitted tok/s at C1/C4/C16 on min/max per family, then re-enable the
   GLM Flash drafter options if they win with their own calibration.
4. **Cost-model state across concurrency (found 2026-10-09).** After a C16
   sweep with GLM Flash decode rows 128, a C1 request selected ~5.4 verified
   drafts per call (baseline 2.9-3.9): 3.27 tokens/step at 46.7-47.1 ms vs
   2.44-3.08 at 34.1-39.7 ms, C1 -9.8% with identical 320 output tokens.
   `CycleCost`'s online fit carries the wide-verify step costs learned at
   high concurrency into single-sequence decode. Its forgetting (FORGET 0.98,
   ~50-step memory: 10% weight left after ~115 steps, ~350 tokens at C1)
   would settle in steady traffic, but a short request after a load change
   runs inside the stale window, and the wide plans it chooses keep feeding
   the fit wide-step costs. Fix: one fit per concurrency bucket (1, 2-4, 5-8,
   9-16+) sharing the table prior, so a regime switch lands on that regime's
   learned costs; agentic traffic switches regimes constantly. Then re-gate
   decode rows 128 at C16 and C1-after-C16 (it was 0.952 C16 in rc3 gating).

## Explore after v2

- **Transient Spark expert-wait spikes in prefill (2026-10-09).** GLM Flash
  EXL3 max, old defaults, three warmed identical 8,192-token prefills: TTFT
  1.153 / 1.818 / 1.159 s with expert wait 331 / 990 / 332 ms and GPU wait
  flat (462-463 ms). One spike made a single-card 8K look like a 0.845
  regression. Find the cause (Spark page cache, RoCE contention, worker
  scheduling) and log per-rank expert time per step so cards can flag it.

Ideas TJ wants kept for later; not v2 work.
- **First after v2: retire ds41rt; V4.1 becomes an ordinary family (TJ, 2026-10-09).**
  Design and stages: "v3: retiring ds41rt" above.
  The goal is to remove ds41rt as a separate engine, not only to move its
  scheduler: V4.1 should be a model the shared engine runs, as GLM Flash and
  MiMo are. It came in as the ds41rt speed floor. Under the old "never slower
  than the replaced engine" rule its hot path was left alone while `shared/`
  grew beside it. Today:
  - ~60K lines in 141 files in `families/deepseek_v41/` (the next largest
    family has ~12.5K);
  - 42 native files (~4.2K lines);
  - 228 ds41-named references;
  - none of `shared/prefill_share`, the generic prefix cache, shared decode
    graphs or the expert service is used.
  Features and fixes land on it separately, or not at all.
  1. **Inventory (Fable design session).** Classify every V4.1-specific
     mechanism as (a) truly model-specific, (b) a generic capability
     `shared/` lacks, or (c) a ds41rt vestige.
     - Model-specific: compressed/sparse KV and indexer attention, HC/mHC,
       Engram tables, weight formats.
     - Likely generic: two-lane encoder pipelining, HC-lagged replay as a
       "lagged state" concept, independent decode lanes, memory placement.
  2. **Grow `shared/`** for the (b) items, so any family can use them.
  3. **Migrate in stages,** each gated by the quick A/B at the 2M operating
     point: serve loop and decode share (`PrefillQueue`, with time-sized
     chunks), prefix cache (`PrefixFamily`/`RefPagePool`), decode graphs, the
     expert exchange and service, memory planning. Delete the (c) vestiges
     as each stage lands.
  4. **End state:** `families/deepseek_v41/` holds only model code, roughly
     a GLM Flash-sized module; no ds41/ds41rt names remain in configs,
     scripts or docs.
  Input from `work/v41-decode-share` (2026-10-09; the branch is closed, not
  merged; the unification supersedes it):
  - a 2048-row text encoder wave costs 480-680 ms and the 128-row replay
    ~250 ms, so wave-boundary interleaving can't bring decode gaps to tens of
    ms;
  - smaller prefill units need the fidelity gate;
  - the shared queue should take time-sized chunks (generalise MiMo's
    `--prefill-chunk-s`);
  - share-0 C16 recheck (1 RTX + 4, 3 interleaved pairs, pre0 vs the branch at
    share 0): the first single-run -25% did not reproduce (the earlier arm had
    run an extra probe panel first). Paired ratios 0.931 / 1.003 / 0.947,
    median -5.3%; arm medians 734.6 -> 714.5 (-2.7%). The CPU audit found no
    hot-path cause: the share-0 queue stays empty and lane moves are
    identical. The unification must keep the share-0 path free of per-round
    overhead and re-measure C16.
- Deterministic Spark expert reduction for prefill (ordered FP32 route planes;
  an export option today): cold V4.1 prefill isn't bit-reproducible run to
  run because of FP32 atomics, which blocks exact cache/golden/A-B checks.
  Measure the cost; make it an opt-in, or the default if cheap.
- Embedding in host RAM on the RTX PRO 6000: benchmark per model before any
  default change. It likely depends on vocabulary head and embedding size,
  and on whether the freed memory actually changes allocation enough to
  onboard another layer. Until then it stays on the GPU (TJ, 2026-10-08).
- On two RTX cards, keep the embedding (or other cold-ish parts) on only one
  GPU where that helps, as DS41 did (TJ, 2026-10-08).
- Segmented full-attention prefill packing for MiMo: pack many short prompts
  into one ~1,024-row pass (Hugh's FR-M.8a). Needs a new exporter/engine
  route; MiMo keeps two prefill lanes for v2.
- MiMo ports from Hugh (merged opt-in, e6ec0318): indexed copy windows LOSE on our
  engine (C1 0.957, copy-heavy rewrite -14.5%: copied tokens accepted ~45% vs our DFlash
  ~99.5%, and the indexed path ignores draft_limit/draft_pause). Keep them off; revisit only
  with an acceptance-gated copy (copy when it beats the neural draft). Snapshot wait + 4 s
  prefill chunks + queue 32 without copies: C1 0.999, rewrite +1.6%, C16 TTFT 91 -> 67 ms,
  C16 throughput 0.971 in one pair. Promote only after 3 interleaved pairs show C16 >= 0.98.
  Warm drafter marks and the host tier remain unmeasured on hardware.
- MiMo RoPE computed on the fly instead of `max_context × 64` tables (FR-M.5b),
  if 32 GB plans still need the memory; a SparkInfer export change.
- MiMo host prefix tier on by default: needs `--host-cache-bytes` to become
  `Option<u64>` in the shared PrefixArgs so the direct CLI can tell "unset"
  from 0.
- Whole-wave sparse-MLA blocks for the 5090 (FR-G.10) and per-SM-count
  exports, only where measured to pay.
- V4.1 asynchronous image encode: today the V4.1 scheduler waits for each
  image's encode before stepping text lanes. Make admission pending (keep the
  prepared job, cache lease and `EncoderTicket`s, poll between decode
  iterations, install features then prefill, release on cancel/failure).
  `EncoderClient` already has the bounded queue/poll/cancel; the work is the
  scheduler restructure.
- One shared adaptive-draft policy for every family (TJ, 2026-10-08): now v3
  item 6, "v3 draft policy: resource-priced, shared". The Astra review
  (`builds/draft-policy-review/REPORT.md`) is an input; its deterministic
  cohort mode and cross-lane contention pricing stay here, after v3.

## Backlog (lowest priority: only when nothing planned is left)

- Qwen 3.8 Flash Next NVFP4 without Sparks, competitive with vLLM on one
  RTX PRO 6000 (localmaxxing card, 2026-10: batch 1, MTP, 2,821 in / 2,048
  out: 373 tok/s output, 12,078 tok/s prefill, 234 ms TTFT; ours today ~100-130
  C1 with EXL3 local experts, NVFP4 W4A4 8K prefill ~9.5K tok/s). Outside the
  usual scope; reference configs for it in spirit: min = simulated RTX 5090
  (32 GB) + 1 Spark, max = 2x RTX with no Sparks.
- Media encoder on a separate host (TJ, 2026-10-06: not for several
  versions; the expert Spark is fine for now). The encoder is already a
  self-contained TCP service with an identity handshake. Traffic per
  1024-token image is ~3 MiB of RGB in and 8–12 MiB of BF16 embeddings out,
  so 1–10 GbE is enough. Needs:
  - a planner placement kind for an external host;
  - a native build for that GPU's arch (SM86 for TJ's RTX 3090 in another
    box, the test plan);
  - its own EncoderId and G2/G3/G4 qualification, since bytes are exact
    only within an arch.
  Budget ~2.3 GiB of GPU memory. Expect ~100–200 ms per 1024-token image on
  a 3090 (estimate).

## Decisions (2026-09-28)

- Fresh copy of ds41rt at `3067d06` into this repo; no history import. The
  import is also a purge: release evidence, perf traces, per-release configs,
  render/summarize scripts and archived patches stay behind in ds41rt.
  Keep qualification and bench tools that exercise live code.
- Delete `real_full` and the legacy commands outright; DS4 Flash/Pro are
  rebuilt as the `deepseek_v4` family.
- All seven hosts and all storage are ours to manage. Replicate a model to
  every rank while working on it, then shrink to one copy or 1/N when done.
  `/mnt/scratch` archive is slow (150 MB/s write, 500 MB/s read).
- Work on `main`, small commits, push often, tag phase boundaries.

## Cluster and rules

- raptor: 2× RTX PRO 6000 (SM120), capped at 325 W while TJ is away.
  Sparks: ostrich, dodo, emu, kiwi, rhea, moa (GB10, SM121, 121 GiB each),
  fabric 10.55.0.1–6. Six Sparks are the pool now; TP4 stays the qualified
  V4.1 default until a six-rank layout beats it.
- Never build on `/mnt/scratch` (NTFS kernel bug). Build under
  `~/.cache/cuteafd/builds/<task>` and keep ds41rt's filesystem assertion.
- Serialize WIP builds and performance runs; one model served at a time.
- Root via `agent-sudo`. Storage via `nest`; whole-file placement is
  explicit, reads never replicate silently.
- Fork changes go to `sparkinfer-glmrt` master and `GPTQModel` main first,
  then bump pin + tree lock here.
