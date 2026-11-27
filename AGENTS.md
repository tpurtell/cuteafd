# Working on CuteAFD

CuteAFD is developed by a multi-vendor **PeerSwarm** of native CLI coordinators (Google Antigravity `agy` running Gemini Neon High — with Neon XHigh as premium tier — as master coordinator, Anthropic `claude` Opus 5.5 across dual accounts `tpurtell`+`vimoin`, and OpenAI `codex` GPT Sol 6.2 High).

- **Swarm coordination & tooling**: Read [peerswarm/PEERSWARM.md](peerswarm/PEERSWARM.md) and [peerswarm.toml](peerswarm.toml). Run `peerswarm --help` (binary at `peerswarm/target/release/peerswarm` or on `PATH`). The read-only Mission Control dashboard runs automatically on LAN (`http://<host>:7788`) whenever the swarm is launched.
- **Activating, scheduling & recovery**: `peerswarm launch` automatically kicks off `"activate peerswarm"` once each CLI reaches idle and starts the watchdog + read-only LAN Mission Control dashboard (`http://raptor:7788`). When activated (or after a watchdog auto-restart / `peerswarm restart --peer <id>` for a CLI upgrade), run `peerswarm activate --peer "$PEERSWARM_PEER"` and `peerswarm job list --all --peer "$PEERSWARM_PEER"`, reattach to surviving durable systemd jobs, and resume or respawn any interrupted subagents from their worktrees/briefs. Master consults `peerswarm schedule --kind <kind> --tier <tier>` before assigning new tasks (honoring `peerswarm peer drain/pause` states, GPU utilization, merge-queue rebase risk, and the 20% last-account rule across Claude `tpurtell`+`vimoin`), and waits reactively via `peerswarm wait --peer "$PEERSWARM_PEER" --merges`. Never busy-poll GPUs or mail; never hammer a peer whose status is `quota_exhausted` or `breaker=OPEN`.
- **Architecture & briefs**: Porting a family or editing shared serve/prefill/exchange/admission code: read [docs/architecture/shared-engine.md](docs/architecture/shared-engine.md). Writing a quantization brief: start from [docs/quantization/brief-template.md](docs/quantization/brief-template.md) and append only a model addendum.
- **Operator interaction**: Any coordinator may take operator requests directly, create local items (`peerswarm item create`), and notify master (`peerswarm mail send --to master`). When operator decisions remain open, end replies with a short open-decisions list and recommendation.
- **Code is king**: Keep documentation minimal (`AGENTS.md`, `README.md`, `docs/options.md`, `benchmarks/` index, and `peerswarm/PEERSWARM.md`). Track day-to-day issues, milestones, and reviews in local SQLite (`peerswarm item/milestone/merge`), replicating active milestones, major `feature` items, and significant merges to GitHub via `peerswarm gh` (never blocking if GitHub is offline).

## Standing ML & Engineering Principles

- **Honor checkpoint numerics with native kernels**: Never settle for dequantizing/converting native formats (NVFP4 expert math is W4A4 at every row count; never pad or widen stored weights to work around a kernel staging limit—fix the kernel). Native paths become default once they measure at least as good.
- **Standard HF checkpoints**: Derive quantization layout (EXL3 trellis, ModelOpt NVFP4, compressed-tensors, FP8 block scales) directly from `config.json`, `quantization_config`, and safetensors headers—no required side files, no runtime weight overlays.
- **Multi-core (MC) checkpoints**: All MC families support uniform `original`, `k8-compact`, `k6-compact`, and `fp8-compact` views (`CORE_VIEW`). Engine JSON readers ignore unknown fields; writers stay strict; use `required_features` for semantics.
- **Exact prefix-cache restores**: Deepest cached prefix snapshot must produce byte-identical state. Exception: DeepSeek V4.1 Encoder SWA Bounded Replay (paper §3.2.2)—label approximate in metrics/traces; fewest re-prefilled tokens always wins selection.
- **Strict lossless speculation (`spec == decode` bitwise)**: Keep strict bitwise invariance despite perf cost. Every kernel and graph route decision must depend ONLY on per-row context length and static device attributes (`cudaDevAttr`), NEVER on batch row count.
- **Attention precision = quality tier**: `ATTENTION_PRECISION=bf16` means BF16-tier quality (F32-oracle error ≤ strict BF16); take the fastest kernel meeting that bar.
- **Packed weight arenas & exact memory admission**: Static weights live in one preplanned arena per physical device per independently retired weight set (optional drafters, onboarding, and native media towers get their own arenas; never allocate per tensor). Every device allocation has one named, measured owner charged once—no `"other"` ledger scope and no arbitrary conservative margins. Measured memory beats formula sums.
- **Memory, KV & placement defaults**: 2M KV target pool, KV pool first, FP8/quantized KV with scales default everywhere (unless paired quality fails, e.g., Muse BF16 default). RTX expert onboarding is always default-on (including TP2); balanced TP placement only; Spark attic carveout fills expert arena first; `VISION=auto` places official bundled vision towers on a Spark first (never bolt third-party encoders onto text checkpoints).
- **CUDA graphs & donors**: Zero steady-state graph captures per request; drain queued work before publishing/releasing storage; `GRAPH_PREWARM=off` uniformly (spanning sites prewarm). Soft cap = `max(1.2·W1, W1+100 MiB, W1+2·step)` with `W1` in shipping config; donors activate only at soft cap, never lag 0; heap pages are 2 MiB and never shrink (eligibility = last GPU use completed). Donor `cudaGraphExecUpdate` counts are informational (never a merge blocker, but DO land finished donor-efficiency PRs).
- **Speculation & stream ordering**: Prefer DFlash2/dSpark by delivered emitted tok/s (not raw acceptance); use uniform calibrated confidence/history drafting across families (no per-family policy sprawl); all `copy_d2h` calls must be stream-ordered against non-blocking engine streams.
- **Device-driven exchange & hand-rolled collectives**: GPU-to-GPU and GPU-to-NIC transfers with no host hop or spin-wait CPU burn. Collectives run on our own verbs/RDMA and P2P layers—never NCCL in serving paths.
- **One SM120 build**: Serve both RTX PRO 6000 and RTX 5090 from one SM120 build; query SM count and L2 size via `cudaDevAttr` at runtime.
- **Clean code, no compat shims**: Delete removed keys and obsolete paths cleanly—no deprecated-key shims. Rust uses typed errors inside crates, `anyhow` at edges, `tracing`, and `unsafe` only in FFI/verbs with safety comments. Never invent synthetic reference fixtures (`np.zeros` forbidden; use repo generators or the fidelity store).

## Measurement, Fidelity & Kernel Duels

- **Fidelity gates generated rows only (`fidelity.rs` `verdict()`)**: Never cite all-rows or context-row KL/top-1 as a pass, failure, or drift. Exact reference compares run vendor PyTorch built against our CUDA toolkit.
- **Measure-first, no proof padding**: State ceiling and stop bar before spending GPU time. When a root cause is already known, state it and fix directly; only run repros/goldens where behavior is unclear or changed. Exclude one untimed warm-up batch per concurrency level.
- **Matched A/B discipline**: Same launcher in both arms; report commits, settings diff, identical inputs, time gap, host load, `/v1/thermal` slowdown/clock/temp windows, and power cap (else mark confounded). One launch per arm unless borderline; judge noisy C8/C1 variance on 3 interleaved alternating pairs using the paired median.
- **Timed runs proceed under build load**: Use pressure-only grant checks (CPU PSI avg10 ≤10, mem ≤5); do not add build-hold gates.
- **Kernel row curves & duels**: Measure warmed time per routed row across `8192/k` (`k=2..4`) plus real tails (`--capacity`, `--routes`, `--compare-routes-exact`, `--check-prefix-rows`). Attention duels gate layer outputs (`uv`/LSE) plus full-model golden. Cap duels at 3 rounds and let entrants finish rounds uninterrupted.
- **Cards, options & publishing**:
  - `benchmarks/current-matrix.json` is test input—run `scripts/tests/test_example_configs.py` and `scripts/tests/test_options_doc.py` before merging card/matrix updates.
  - Card columns: `1× RTX` and `2× RTX` (0 Sparks when GPUs hold the model; MC main card = K8; retiring a card never disables serving).
  - Any launch key, `./run.sh` option, default, or default-off perf lever updates `docs/options.md` (and [Optional performance levers](docs/options.md#optional-performance-levers)) in the same PR.
  - Releases tag `vN.N.N` on `main` (fast-forwarded from `dev`) only with operator approval; release notes list features only.

## Build, Resource Scheduling & Storage Hygiene

- **Keep everything inside `peerswarm.state/`**: Peer worktrees live in `peerswarm.state/peers/<peer>/`, build caches and task builds in `peerswarm.state/caches/` (on `/home/tj/Developer` 7.3 TB NVMe), SQLite DB and dated artifacts in `peerswarm.state/work/`. Always check build paths with `scripts/build/assert-build-filesystem.py` (never build on `/mnt/scratch` NTFS).
- **Unified resource scheduling (`peerswarm resource`)**: All GPU, Spark, build-seed, and host compile-slot locks go through `peerswarm resource` (`exec`, `hold`, `pick`, `status`, `reap`), replacing legacy `hwlock.py`, `build-slot.sh`, and `build-resource.py`.
  - Canonical order (enforced automatically): `sparks -> gpu0 -> gpu1 -> rhea -> moa -> kakapo -> cassowary -> maxq -> build -> cpu-gate`.
  - Never nest non-gate locks (refused with exit `77`). Hold GPU/Spark locks only during hardware execution (never compiles or downloads); stop all started containers before releasing locks; leave hardware idle after runs.
  - Key commands:
    - Hardware run: `peerswarm resource exec --locks sparks,gpu1 --max-run 1800 --preflight '<cmd>' -- <CMD>`
    - CPU compile slot: `peerswarm resource exec --locks build --nice 19 -- cargo test --workspace --locked --no-run`
    - CPU merge gate: `scripts/dev/cpu-gate --task <name> [--cargo] [--scripts] [--skip-live-planners]` (`--full --slow` for release cuts only)
    - Card A/B: `python3 scripts/bench/wip-cards.py --kit gate --interleave --repeats 1 ...`
    - Family goldens / GPU tests: `python3 scripts/bench/golden-run.py --task <name> ...`
    - Iterate / release: `./wip.sh --slot S --role both`, `./run.sh --wip S --restart`, `./build.sh`
- **Durable detached jobs (`peerswarm job run`)**: Launch all long-running gates, builds, goldens, benchmarks, and quant runs via `peerswarm job run --name <slug> --peer "$PEERSWARM_PEER" [--host maxq] [--locks ...] -- <CMD>` (wraps `systemd-run --user --collect --expand-environment=no`, forwards `PATH`/`HOME`, keeps `KillMode=control-group` and `TimeoutStopSec=90s`, logs under `peerswarm.state/work/data/`, tracks in SQLite for crash/restart recovery & dashboard viewing, and sends high-priority failure alerts so silent failures never idle GPUs). Use `peerswarm job stop` to cancel cleanly (never `pkill -f` or `kill -9`) and `peerswarm job prune` to clear stale failed units.
- **Compiler & AOT caches**: Never bypass or clear `kache`/`sccache`/`jit`/`cargo-home`/`export-cache` except on operator order, release step, or disk emergency. Never mix `kache` and plain Cargo builds in the same `CARGO_TARGET_DIR` (kache restores read-only hardlinks). Build container binaries inside dev images (host glibc 2.43 is newer than container glibc).
- **AOT exports & batched pin bumps**: CuTe-DSL/Triton AOT exports run CPU-only (`CUDA_VISIBLE_DEVICES=""`) without GPU locks. Submodule code lands on `~/Developer/sparkinfer-glmrt` `master` or `~/Developer/GPTQModel` `main` first, then bumps the pin + `third_party/*.lock.json`. Batch queued placement-input publications into one integration branch (never serial per-PR 78-minute publications); check `scripts/dev/cpu-gate` against the exact hashed placement set.
- **No live host state in tests**: Unit/planner tests inject fixed memory budgets and never depend on live host RAM/GPU availability.
- **Disk visibility & safe cleanup**: Run `peerswarm disk --remotes` to inspect `peerswarm.state/`, Rust targets, worktrees, and remote build hosts. Run `peerswarm gc --targets --merged-worktrees --apply` to clean safely (live-checked by PIDs, Docker mounts, and open work items—never delete by name regex). Remove your task worktree and Cargo `target*` dir immediately after merge.

## Cluster & Storage Facts

Host tuning details: [docs/host-tuning.md](docs/host-tuning.md). Passwordless SSH works by hostname; privileged commands run directly via `agent-sudo --agent-context "<why>" <CMD>` (never wrap `sudo` inside a script).

- **`raptor` (coordinator, uid 1000)**: x86-64 (32c/64t), 2× RTX PRO 6000 Blackwell 96 GB (SM120), governed by `gpu-powerd` (850–900 W shared budget, max 600 W/card, 80 °C target; never alter limits). Fabric `10.55.0.22` / `10.55.1.22`. `/home/tj/Developer` is a 7.3 TB NVMe.
- **Serving Sparks (GB10, SM121, ARM64, 121 GiB unified, uid 1001)**: `ostrich`, `dodo`, `emu`, `kiwi`, `rhea`, `moa` (`10.55.0.1–6`, dual RoCE rails `10.55.0.x` / `10.55.1.x`). Admission uses `MemAvailable` (CUDA reclaims Linux page cache). Set `USER`/`LOGNAME`/`HOME`/`TORCHINDUCTOR_CACHE_DIR`/`TRITON_CACHE_DIR` in uid containers.
- **LAN-only build seeds (10 GbE, build only, never serve)**: `kakapo` (`172.22.2.7`) and `cassowary` (`172.22.2.8`), 3 counting build slots each via `peerswarm resource pick --pool lan_seeds`.
- **`maxq`**: WSL2 x86-64, RTX PRO 6000 Max-Q 96 GB (`ssh maxq`). Quants, CPU gates, single-GPU correctness only (no perf A/B, cards, serving, or `ncu`). Never touch `~/Developer/cuteafd/.venv` on `maxq` (use `~/.cache/cuteafd/gate-venv`).
- **Switch**: `agent@switch.200gb` is read-only unless operator explicitly approves a change.
- **SparkNest (`/mnt/sparknest`, `HF_HOME=/mnt/sparknest/hf-home`)**: Stream checkpoint shards over RoCE (~5 GB/s)—don't replicate except Engram/PLE tables pinned to `raptor`. Check `nest ls .../blobs` before claiming where spread shards live. Quant uploads use HTTP LFS (`HF_HUB_DISABLE_XET=1`, parallel 16 workers), then no-copy `nest hf import` and delete intermediates.

## Git Rules

- `dev` is the integration branch; `main` moves only at tagged releases (`vN.N.N`).
- Keep `~/Developer/cuteafd` fast-forwarded to `dev` with pinned submodules after merges; do feature work in `peerswarm.state/peers/<peer>/tasks/<slug>` worktrees on `work/<peer>/<task>` branches.
- Small commits with imperative subjects, before → after tables on perf commits, no attribution trailers. Always initialize submodules locally via `scripts/dev/submodule-init-local`.
