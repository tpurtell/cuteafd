# Working on cuteafd

Read `PLAN.md` first. This file is the standing rules. Code is king: keep
external documentation to this file, `PLAN.md`, `USING_AGENTS.md` (which
model does what, how to brief, launch and review agents), one README
and the `benchmarks/` index; measurements go in commit messages as short
before → after tables with conditions.

## Working method

- Disk: builds live on raptor's root NVMe and fill it fast (1.4 TB of
  `builds/` once filled the disk and crashed runs). When a task finishes,
  delete its Cargo `target*` directories and release staging; keep sources,
  logs and results. Check `df -h /` before large builds.
- Profiling with root: run Nsight Compute directly as
  `agent-sudo -n --agent-context "<why>" /usr/local/cuda/bin/ncu <args> <program>`
  with the full command line visible. Never put ncu (or anything else) under
  sudo inside a wrapper script or interpreter (`sudo python …`, `sudo bash …`):
  the approver can't see what it runs and will flag or deny it.
- Lock order: when a run needs both locks, take `sparks.lock` first, then
  `gpu1.lock` (as `~/.cache/cuteafd/builds/tp2/locked2.sh` does); never hold
  one while waiting on the other in the opposite order — that deadlocks the
  cluster. Every lock wait and run has a timeout.
- Waiting costs tokens, sleeping doesn't: start a long run as one background
  command that blocks until it finishes (a lock wait or a loop sleeping ≥60 s)
  and let its completion wake you. Don't poll status or tail logs in between;
  never keep a wait loop alive after you stop needing its result.
- Parallelize: run independent feature work as separate agents in separate
  git worktrees, one branch per task, on disjoint hardware where the work
  needs a GPU. Serialize only what shares a GPU or a build cache.
- Merge fast: once a change is accepted (tests pass, the feature's own
  measurement clears its bar), commit and merge it into the working branch
  immediately. Run any remaining measurement afterward and resolve a small
  delta with a follow-up commit rather than holding the merge open.
- Measure-first experiments state, before spending hardware time, what the
  ceiling looks like (a quick estimate or a reference measurement) and a
  stop bar (what result ends the experiment, in either direction). Chasing
  a number past its stop bar without a new hypothesis is wasted hardware
  time.
- Be frugal with measurement: one launch per arm is enough to decide most
  things; escalate to interleaved repeated sessions only when a result is
  genuinely borderline (see Engineering rules for the exact tiers). Don't
  schedule comparison runs against other engines or forks out of curiosity
  — scanning prior art for implementation ideas is fine and encouraged, but
  spend hardware time on this engine's own performance, not on producing a
  comparison table nobody asked for.
- Explain warm-up instead of re-running. The first wide batch after a
  launch is reliably slower (per-token memory-mapped table reads, PyTorch/
  CUDA first-use workspace allocation, graph capture on first shape): run
  one untimed warm batch per concurrency level, then judge the warmed
  numbers. A dip that a known warm-up effect explains does not need a
  second session.
- Lock and container hygiene: acquire a hardware lock (see Build and run)
  only for the duration of the run that needs it, and stop every container
  you started before releasing the lock. Leave the hardware idle when a
  task finishes — don't keep a model served, and don't restore serving
  after a hardware run, just to leave something running. Bring a model up
  only when a task needs it up.

## Engineering principles

- Honor checkpoint numerics with native kernels. Converting a checkpoint's
  native format into one of the engine's pre-existing internal types
  (dequantizing NVFP4 or per-tensor FP8 to BF16, running W4A16 instead of a
  checkpoint's calibrated W4A4) is acceptable only as an interim fallback.
  The target is a properly optimized native kernel for the format the
  checkpoint actually ships, and that native path is the default once it
  measures as at least as good — even when it costs a small amount of
  nats/KL versus the converted path, if that is the checkpoint's intended,
  calibrated numerics.
- Load standard Hugging Face checkpoints directly from their own
  `config.json`, `quantization_config` and safetensors tensor headers.
  Derive layout (EXL3 trellis storage, ModelOpt NVFP4, compressed-tensors,
  FP8 block scales) from the checkpoint itself; a side file produced by our
  own tooling is read only if present, and only to cross-check agreement
  with what the checkpoint's own headers say — never required.
- A checkpoint release that only fits a crippled configuration (a tiny KV
  pool, or a speed far below what the model should deliver) is not a
  target: prefer a compatible community quant (EXL3, ModelOpt NVFP4) that
  fits the hardware comfortably over forcing the official release to fit.
- Exact prefix-cache restores: restoring from the deepest cached snapshot
  that is a prefix of an incoming request must produce state
  byte-identical to having prefilled that prefix from scratch — no
  approximate reconstruction. Snapshot at prompt end and at turn end so
  both single-shot and multi-turn agentic traffic hit. Logit or greedy-text
  differences from floating-point reordering elsewhere (chunk-size changes,
  kernel route switches) are acceptable; a cache that changes the model's
  actual state is not.
- Device-driven exchange: routes, requests and replies move GPU-to-GPU and
  GPU-to-NIC with no host hop and no idle CPU burn on a request's hot path.
  The host launches work and waits on it; it does not shuttle bytes between
  devices or poll in a spin loop that could instead be a device-side flag.
- Collectives are hand-rolled on our own verbs/RDMA and P2P layers, not
  NCCL. NCCL is acceptable only as an optional, occasional ceiling
  measurement to sanity-check a hand-rolled path, never as a dependency of
  the serving path.
- One SM120 build serves every supported RTX card (PRO 6000 and 5090) with
  no detriment to either. AOT exports and kernel launch configuration must
  not bake in one card's SM count or L2 size; query them at runtime
  (`cudaDevAttr`) for grid, wave and prefetch sizing, and let the placement
  planner handle the memory difference between cards.
- Multimodal input is supported only through a family's officially bundled
  encoder (the vision/audio tower the checkpoint actually ships). Don't
  attach a third-party encoder to a text checkpoint opportunistically.

## Build and run

- Build only under `~/.cache/cuteafd/builds/<task>` on root NVMe; run
  `scripts/build/assert-build-filesystem.py` on every path first. Never reuse a
  Cargo cache that has seen filesystem errors.
- Initialize submodules recursively at their pinned commits in every fresh
  worktree before building or testing. Hold `build.lock` only while compiling.
- `./wip.sh --slot S --role both` for iteration, `./run.sh --wip S
  --restart` to launch; `./build.sh` and `./run.sh` for
  release images. Slots isolate artifacts, not GPUs or ports: serialize
  builds and performance runs, one model served at a time per GPU.
  `~/.cache/cuteafd/{gpu1,sparks}.lock` (flock) are the hardware mutexes;
  hold one only around the run that needs it and stop its containers before
  releasing it, so the next task finds a clean device.
- Submodules are pinned with tree locks (`third_party/*.lock.json`).
  Kernel changes go to `../sparkinfer-glmrt` master, quantizer changes to
  `../GPTQModel` main; push there first, then bump pin and lock here.
  A SparkInfer bump also needs
  `scripts/build/build-dev-images.sh` (shared dev images on the coordinator
  and every Spark rank, then `./wip.sh --recreate`); `wip.sh` and `run.sh
  --wip` refuse a stale dev image and say so.
- Kernels: CuTe-DSL/Triton AOT exports from the b12x fork are the default;
  hand CUDA only where measured to pay. SM120 and SM121 are both targets.
- Run CUDA/PyTorch checks inside the matching architecture's container.
- Host checks: `cargo check/test --workspace` from `rust/` with
  `CARGO_TARGET_DIR=~/.cache/cuteafd/builds/<task>/target` (no Python
  needed). Script tests: `.venv/bin/python -m pytest -q scripts/tests`
  (`uv venv --python 3.12 .venv` + pytest numpy tokenizers jsonschema pyyaml pillow matplotlib safetensors);
  no failing ids since the codex/v1 merge (808 pass); add none.
- Kernel/exporter pre-merge gate: CPU-only `scripts/build/compare-sm-exports.py --sms 188,170`; review object/cubin differences.
- Release and WIP builds persist Cargo registry/git under
  `~/.cache/cuteafd/cargo-home/<arch>`, toolchain-keyed JIT caches under
  `~/.cache/cuteafd/jit/<dev-toolchain-hash>/<arch>`, and local compiler caches
  under `~/.cache/cuteafd/{kache,sccache}/<arch>`. Unsafe/non-NVMe paths warn
  and fall back to per-build storage. `CUTEAFD_BUILD_CACHES=off` bypasses all
  these stores; Cargo stays `--locked`, using offline resolution when the
  locked dependencies are already cached and populating missing inputs online.
  Release/WIP enable kache and CUDA sccache by default; dev shells share input
  caches but compiler wrapping stays opt-in. New WIP containers run as each
  host's uid with a writable NVMe `/wip` and home. Legacy root slots still run;
  rebuilding needs `--recreate` and a fresh writable `WIP_ROOT` when the old
  root is not owned by that uid. Normal builds never chown or remove old slots.
- Optional compiler cache override: set `CUTEAFD_KACHE=/absolute/path/to/kache` (v1.0.0)
  and optionally `CUTEAFD_KACHE_REMOTE=/shared/cache/directory` for host gates.
  Build containers use local stores only; missing kache or inaccessible remote warns and falls
  back. Cache invocations time out after 300 seconds (`CUTEAFD_KACHE_TIMEOUT_SECONDS`
  overrides for very slow compilers), then retry plain and disable caching for the
  rest of that build. Never make kache a gate prerequisite. Never run plain and
  kache Cargo builds on the same target directory: kache restores outputs as
  read-only hardlinks into its store, so a later plain build fails with "output
  file ... .rmeta is not writeable". Use a separate target per mode. Use a static executable compatible
  with the dev image; the Homebrew host toolchain and container do not share keys.
  `CUTEAFD_KACHE_CACHE_DIR` selects the NVMe local index/blob parent (architecture
  leaves are automatic); keep fresh per-task Cargo targets and hold `build.lock`.
  For host gates, source `scripts/build/compiler-cache.sh`, then call
  `cuteafd_compiler_cache_setup "$HOME/.cache/cuteafd/builds/<task>"` before Cargo.
  For Spark builds, `CUTEAFD_KACHE_SPARK` names a native ARM executable already
  installed on the Spark and `CUTEAFD_KACHE_SPARK_CACHE_DIR` its local cache.
  WIP mounts require `--recreate` when enabling/changing cache paths. Build-script
  execution caching stays off; Rust and native C/C++/CMake compiles are wrapped.
  CUDA launchers are wired but were not exercised by the CPU-only pilot.
  `scripts/build/cuteafd-dev.sh cpu -- COMMAND` runs CPU-only with the canonical
  `/workspace/cuteafd` source mount and passes optional cache mounts/setup. Container
  targets use `CUTEAFD_DEV_TARGET_DIR=$HOME/.cache/cuteafd/builds/<task>/target`
  (guarded NVMe bind mount); do not use an image-layer target cache as a substitute. Prefer canonical CPU gates
  for cross-worktree hits; never normalize `CARGO_MANIFEST_DIR` away (it is a real runtime input).
  Opt-in artifacts carry `COMPILER_PROVENANCE.json`: source commit/dirty, toolchain,
  actual cache mode/fallback, flags/remapping and binary SHA256. Vanilla hashes may
  differ from kache due to path remapping; identical-input kache cold/warm must match.
  Both sparknest and NFS-over-RDMA passed concurrent restores/checksums. Prefer a
  raptor-owned NVMe NFS cache for explicit placement; do not change exports or
  sparknest policy. SQLite stays local, never on either remote. Remote manifests
  may lose concurrent additions (misses, not corrupt artifacts). No BuildKit mount
  is needed in `Dockerfile.release`: it packages prebuilt artifacts, not compilers.
- `./build.sh` (release pair, coordinator + Spark leg): set
  `CUTEAFD_RELEASE_BUILD_ROOT` and `CUTEAFD_RELEASE_REMOTE_BUILD_DIR` under
  `~/.cache/cuteafd/builds/`, and `CUTEAFD_RELEASE_SPARK_TP_ROLES=` for a
  TP4-only pair. Git worktrees are supported: the compiler gets a copy without Git metadata
  after host-side submodule verification. Image assembly still reads the live
  checkout: keep that checkout unchanged and edit in another worktree until
  the build finishes. Crates download from crates.io each build
  and can crawl while the WAN is busy; it is slow, not stuck.
- Iterate with `./wip.sh --slot S` then `./run.sh --wip S --restart`; A/B two
  checkouts with `scripts/bench/bench-ab.py`. `cuteafd plan MODEL` (any HF id or
  snapshot dir) says what a checkpoint needs before any kernel work.
  `--coordinator-gpu-budget-gib` is the logical per-GPU ceiling for plan,
  serve and golden; plan defaults to 95.5 GiB. Its distinct weights-only
  cap is `--coordinator-weight-budget-gib` (default 80 GiB);
  `cuteafd fabric` shows ports, link/PCIe rates, subnets and the rail plan
  (services log the same line at startup).

## Engineering rules

- cuteafd is measured against itself, not a replaced engine (TJ,
  2026-10-08). Verify each change at the size it needs, as part of normal
  work; the release card set catches the rest. A regression found at release
  doesn't abort it: ship, and follow with a point release that fixes it.
- Correctness first, then warm-up, then identical-config A/B, interleaved,
  three runs for a final number. Judge speculation by emitted tok/s, not
  acceptance. Profiling perturbs timing.
- Benchmark each model on three card columns only (TJ, 2026-10-07):
  **5090** (1× RTX 5090 + the fewest Sparks it fits; measured on Hugh
  Madden's hardware by his agent), **1× RTX** (1× RTX PRO 6000 + the fewest
  Sparks it fits) and **2× RTX** (2× RTX PRO 6000 + 4 or 6 Sparks,
  whichever divides the model sensibly). A column whose GPUs hold the whole
  model is a **0-Spark** card: layer onboarding would leave the Sparks idle,
  so don't attach any. `cuteafd plan --layout` decides fit. Supplementary
  cards (e.g. GLM Flash on 2× RTX, 0 Sparks, for a quant that fits) are
  allowed. Other layouts need correctness gates, not perf tables; the
  planner's estimates cover them.
- Until the first official release, be frugal: measure only what a decision
  needs, one launch per arm, no repeat sessions unless a number is borderline.
  Release prep runs the "Release smoke" profile (basic card + quick quality,
  ≤5 min per config including load) per family, quant and reference config.
- Tiered gates. Merges and features: cargo/script tests (failing ids, not
  counts), golden NLL/byte-exactness on one GPU or loopback, and the
  feature's own measurement. Changes to shared hot paths (transport, expert
  exchange, native lib, sampler, memory placement) add one quick A/B pair
  on an affected model: candidate (WIP images) vs work/p0 measured the same
  day, C1 + C16 code decode, tools and memory headroom (~20 min). Correctness
  and memory safety gate; speed is reported, and only a clear drop (below
  ~0.95) earns more sessions (benches run one untimed batch per concurrency
  level; DeepSeek engram tables and first-use workspaces make the first wide
  batch after a launch ~10% slow — explain, don't re-run). Releases run the
  card set, not a multi-session parity campaign. Release images are built for
  release cuts, not to verify branches; agentic benches gate with 1–2 short
  sessions, the full bench runs at release.
- Unsupported is a result, not a crash: `cuteafd plan` names the tensors,
  formats, shapes and the exporter or kernel to add.
- Load speed is a feature; do not regress readiness through wasted load
  work (redundant reads, repacking, re-tiling, serial loads). Readiness may
  grow in proportion to work that pays at runtime, e.g. more resident expert
  bytes or startup graph capture. Judge such changes by load efficiency
  (seconds per resident GB, per phase) against the baseline, not wall time.
- Preserve graph pointer/shape/workspace lifetimes; drain queued work before
  publishing or releasing storage; weight and workspace admission precede
  allocation; zero steady-state graph captures per request.
- Rust: typed errors inside crates, `anyhow` at edges, `tracing`, no
  `unsafe` outside FFI and verbs layers, each block with a safety comment.
- Keep weights, build artifacts, benchmark runs and local config out of Git
  (published reports under `benchmarks/` are the exception).

## Results publishing

- The root README holds the only exhaustive table: the basic benchmark
  profile for every family and quant in the three card columns (5090,
  1× RTX, 2× RTX; 0 Sparks where the GPUs hold the model).
  Re-run a family's rows after changes that target that family's code (or a
  shared hot path that plausibly moves it); skip irrelevant changes,
  staleness is fine.
- Other profiles run only when specifically requested: their exports
  (`report.svg` + `report.json` from `cuteafd bench`, the same runner as
  the dashboard) go to `benchmarks/<family>/<date>-<profile>-<hardware>/`
  and get a line in `benchmarks/README.md` (per family, newest first: date,
  profile, hardware, build). Commit them straight on top of `main` or the
  working branch; no release or branch needed.
- `cuteafd bench --url URL --profile NAME` runs a profile on a launched
  server and writes the exports there; `cuteafd bench publish` rebuilds
  both tables from what is placed. Release prep: `cuteafd bench smoke
  --matrix FILE` (format: `scripts/bench/release-smoke.example.json`)
  launches each entry with `./run.sh`, runs Release smoke, exports, tears
  down, resumes per build and runs entries on disjoint hardware side by
  side under the lock files.

## Releases

- Tag phase boundaries during development as `p0`, `p1`, …; an official
  release is tagged `vN.N.N` once its basic benchmark profile has run
  across the full family × quant × reference-config matrix.
- The README results table is generated by `cuteafd bench publish` from
  Release smoke exports, not hand-edited; re-run the smoke matrix before
  cutting a release if any family's rows are stale against the code being
  released.
- A family's `docs/models/<family>.md` changelog gets a new row only when a
  change that affects that model's output or performance triggered a fresh
  basic eval for it — not for every release, and not for changes that
  don't touch that family's path.

## Git

- Work on `main`. Small commits, push after every green step. Tag phase
  boundaries `p0`, `p1`, … Imperative subjects; performance commits carry
  the measurement table. No attribution trailers.

## This cluster

The rules above apply to any deployment of this engine. This section
records the specific hardware, network and storage facts for the cluster
this repository is developed against; a different deployment will have
different hosts, addresses and paths, but needs the same categories of
facts recorded somewhere for agents working on it.

### Hosts

- `raptor`: coordinator, x86-64, 2× RTX PRO 6000 Blackwell 96 GB (SM120),
  power-capped at 325 W during unattended runs. Fabric 10.55.0.22 /
  10.55.1.22.
- Sparks (GB10, SM121, ARM64, 121 GiB unified; `nvidia-smi` memory reads
  N/A, measure with CUDA): ostrich, dodo, emu, kiwi, rhea, moa at
  10.55.0.1–6. Ranks 0–5 in that order. All six are the pool. TP4 on the
  first four is the qualified V4.1 default until a six-rank layout wins.
- Fabric: Sparks have two RoCE ports at 200 Gb/s each on separate
  subnets (rail A 10.55.0.x, rail B 10.55.1.x); raptor has one 400 Gb/s
  port carrying both rail subnets. The switch is being raised from 100G to
  200G. Read rates and states at startup (`rdma link`, sysfs `rate`); do
  not assume them. Dual rail at 100G caused head-of-line blocking against
  200+ Gb/s PCIe ingress, so rail use is a measured decision.
- Passwordless SSH by hostname. Fan-out: `scripts/launch/run-on-hosts.sh`.
- Root: `agent-sudo --agent-context "why" CMD` (remote human approval).

### Storage

- Every host mounts sparknest at `/mnt/sparknest`; `HF_HOME` is
  `/mnt/sparknest/hf-home`. Sealed local copies read at NVMe speed; files
  without a local copy stream over RoCE at ~5 GB/s. Reads never replicate.
- Don't replicate for loading: serving and golden references stream
  checkpoint shards from wherever sparknest holds them (one copy spread
  across hosts is enough; ~5 GB/s over RoCE, slower readiness is
  acceptable). The exception is a model's Engram tables (V4.1): they are
  read at random every token, so keep them local on the coordinator
  (sparknest rules `v41-engram-*` pin them to raptor). Watch readiness
  loosely for traffic changes rather than replicating to speed it up.
- `nest where hf:ORG/MODEL` shows copies; `nest replicate SEL --hosts
  ... --wait` places them; `nest evict` removes (never the last copy);
  `nest plan --free` when space is tight. Manage space.
- Each host's `~/.cache/huggingface/hub` is a symlink into `/mnt/sparknest`;
  containers must mount the resolved hub (`run.sh` does) or `/mnt/sparknest`.
- `/mnt/scratch` and `/mnt/models` are slow archive stores (150 MB/s
  write, 500 MB/s read). Never build on `/mnt/scratch` (NTFS kernel bug).
