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
   - **Memory management.** One admission solver for every family:
     - reserve the KV pool first (2M PRO / 1M <=32 GB);
     - then graphs and workspaces;
     - then RTX expert layers, TP2 across both GPUs, with non-split items
       (drafter, encoders) moved to balance the two GPUs;
     - the planner equals the runtime admission, with a test per family.
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
   - **Starting point:** work/v4-placement b2f26af9 (pool-first solver,
     planner/runtime equality) and `builds/v4-placement/SUMMARY.md` (V4 TP2
     design and kernel/loader audit). V4 Flash/Pro measured in rc1:
     KV pool 581K-1.39M tokens because experts are placed first, GPU1
     13-24 of 90 GiB used.
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
   item 1; build the shared versions once, not twice.
Gate per family: golden/fidelity, then the quick A/B at the 2M operating
point on the min and max reference configs. Requalify each family's cards
as it moves.

## Explore after v2

Ideas TJ wants kept for later; not v2 work.
- **First after v2: retire ds41rt; V4.1 becomes an ordinary family (TJ, 2026-10-09).**
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
- One shared adaptive-draft policy for every family (TJ, 2026-10-08): a
  CPU-only core with per-family topology/traffic/shape adapters, replacing
  V4.1's own dSpark policy and the separate GLM/Qwen/generic ones. Design
  review under way (Astra); decide after discussing it.

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
