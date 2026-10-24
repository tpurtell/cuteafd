# WIP hardware cards

Use `scripts/bench/wip-cards.py` instead of copying an RC kit driver. It reads
`cfg/*.config` and `matrix-*.json` from the kit (default: release-v2-rc2), writes
private one-card matrices and a 0600 API key, and runs `cuteafd bench smoke`
under `setsid` with numeric exit files. Smoke alone owns serving locks.
Stopping a run (`systemctl --user stop UNIT`, Ctrl-C, SIGTERM/SIGHUP) makes
smoke tear down its coordinator and Spark workers before it exits; wip-cards
and `hwlock.py` wait for it, so the locks are released on clean hardware (see
the module help for the systemd-run flags). Use a
current CLI containing the opt-in observer/precheck hooks for probes.

Fabric attribution reads each serving host's SparkNest 0.4.1 schema-1 stream.
The launcher mounts only its dedicated `/srv/sparknest/run` directory read-only,
when that directory already exists, and connects to `run/rdma-counters.sock`
inside it. The stable directory mount lets the observer reconnect after the
storage daemon replaces its socket on restart (ADR-052); it does not expose
the parent state directory or create missing directories. Missing, restarting
or stale sources remain unknown until fresh samples establish a new interval.

```sh
python3 scripts/bench/wip-cards.py --interleave --repeats 3 \
  --cards v41-flash-min --arm baseline=my-base:slot --arm candidate=my-new:slot \
  --task v41-ab --probe v41-flash-min:image,memory,console \
  --expect-pool v41-flash-min=2097152 --dry-run
python3 scripts/bench/wip-cards.py --matrix --parallel 2 \
  --cards v41-flash-sim5090 glm53f-exl3-sim5090 \
  --arm left=my-left:slot --arm right=my-right:slot \
  --card-arm v41-flash-sim5090=left --card-arm glm53f-exl3-sim5090=right --dry-run
```

Remove `--dry-run` to execute. `--set CARD:KEY=VALUE` overrides serving keys.
Parallel mode accepts only simulated or explicitly correctness-only cards
(`kind`: `sim-5090`, `fidelity`, `cache`, `image`, `no-fit`, `packed-check`),
refuses shared GPUs/Sparks/ports/instances and published/compared cards, and
withholds incidental performance numbers. Shared-arm parallel cards use one
smoke scheduler; separately bound arms use independent smoke processes with
disjoint lock sets. Simulated timings never enter
medians or paired deltas. Results include JSON, a markdown table, hook logs,
raw/corrected CUDA 50ms peaks (NVML sampler-context subtraction). The memory probe holds about 550 MiB of CUDA context on each sampled GPU for the whole run, and the server's admission sees it: a card with `--probe memory` admits a smaller pool than a real launch (V4 Flash max 1.18M where the same plan gives more without the probe). Leave the memory probe off for pool and admission comparisons, and never compare pools across arms that differ in probing, worker ring
ledger peaks and authenticated console stage events. Explicit probes fail
rather than silently claim missing evidence. Pool admission and red-square
image checks run after readiness, before the benchmark. Explicit `panels:`
selections are validated before launch and checked against the finished report.
Serving instance names use a short hash suffix to stay within run.sh's 41-byte
limit; WIP arm identities remain unchanged. `--interleave --matched-prompts`
replays the same `CUTEAFD_BENCH_NONCE_SEED` sequence across fresh arm servers,
with a new recorded seed per pair. Basic/decode-content requests remain greedy
with their fixed token limits. Authenticated console rounds report per-request
emitted tok/s (first-token-to-retirement decode time), tokens/step and ms/step
(round service time); round-service tok/s is reported separately. Missing or different first-request token hashes fail the pair. This
requires arm images with opt-in prompt-hash telemetry; release cards without
the flag keep their existing prompts.

Optional `--build baseline=REV --build candidate=REV` freezes separate source
worktrees, uses identical family scopes, stages sealed slots to the union of
selected Spark hosts, and checks every card's local/remote seals before launch.
Builds use `build.lock`, nice 19 and 16 CPU jobs. Sources with opt-in export
locking take the specifically mounted raptor GPU lock only during native/AOT
work. Legacy frozen sources hold that lock for the full build instead; the
mode is logged and recorded per arm. No scripts are overlaid into frozen arms.
A busy Spark seed is
refused using a raptor-side lock check (not an atomic reservation; operators
must still keep the seed idle throughout export). `--cleanup` verifies and
removes only task-owned build roots/containers, overlays and registrations on
raptor and all six Sparks; existing arms without this task's ownership marker
are deliberately refused, never adopted. Keep the same `--task`, `--state`
and arms for cleanup. Source worktrees are retained for inspection.

## Subsets, traces, profiles, preflight and resume

- `--kit gate` is the in-repo kit (`scripts/bench/kits/gate`): mimo26-flash-max,
  qwen38-exl3-min, glmf-exl3-max (2 RTX, 0 Sparks, EXL3) and v41-flash-max on
  `examples/configs`, with ports of their own.
- `--panels LIST` replaces every card's profile (`baseline` alone for the basic
  card and quick quality, `none` for no server profile). `--levels 1,16` adds a
  client-side code-decode sweep after it (one warm and one measured wave per
  level); `--post-load-c1` measures C1 again after the widest level. The sweep
  runs in the smoke binary, so arms with different server builds measure the
  same workload; pass `--binary` built from a checkout that has it (wip-cards
  checks).
- `--panels baseline,spec_lossless_set` adds the lossless merge gate: every
  case of `scripts/bench/fixtures/spec-lossless-set-v1.json` (or
  `--lossless-set FILE`, `[{id, prompt, nonce}]` with the full nonce string)
  runs cold with drafts on and off for 128 greedy tokens and must match in
  token ids and every row hash, with no tie tolerance. The row fails as
  `quality` naming the failing cases; each case's probe pair (prompt ids,
  generated ids, row hashes, top two) stays in `report.json` under the pass's
  `evidence`. The baseline's single `spec_lossless` check is unchanged.
- `--trace` gives every job `SPECULATION_TRACE=<job>/speculation-trace.jsonl`.
  The observer notes it after warm-up; a speculating arm that wrote nothing
  fails its row. `--set CARD:PROFILE=census,transport-timing,expert-timing`
  turns profiling on in the coordinator and every Spark worker
  (`CUTEAFD_PROFILE_HOST` names the physical host); the per-job `profile/`
  directory collects every host's files. `PROFILE=expert-routes` logs every
  serving Spark wave's top-k expert ids (`expert-routes-HOST.jsonl`, plus GLM
  5.3 MoE input samples capped by `PROFILE_EXPERT_INPUT_SAMPLE`); a row whose
  log is missing or empty fails. Score it with
  `scripts/bench/expert-route-balance.py LOG --tp 4,6`.
- `--set CARD:PROFILE_SPARK=nsys|ncu` wraps Spark expert workers in a profiler
  in real serving, on the ranks in `PROFILE_SPARK_RANKS` (default `0`),
  independently of the coordinator's `PROFILE` (the keys are documented at the
  PROFILE comment in `scripts/launch/run-family.sh`). With nsys, the sweep
  window opens and closes every wrapped worker's session together with the
  coordinator's (pass `--levels`). With ncu, set `PROFILE_SPARK_NCU_KERNEL` and
  usually `PROFILE_SPARK_NCU_SKIP` to get past readiness launches;
  `PROFILE_NCU_SKIP` does the same for the coordinator. Reports come back as
  `profile/<tool>-HOST-rankR.<nsys-rep|ncu-rep>`, and a selected rank without
  one fails the row. Needs a smoke `--binary` that stops profiled workers
  gracefully (wip-cards checks).
- Preflight runs before any queue ticket and again at grant (smoke's prelaunch
  hook): the arm is sealed on every host, the slot holds the expert packages and
  audio entry the card needs (the same derivation `--build` uses), and
  trace/profile directories are writable. Container absence is checked only at
  grant, so a job may queue behind a running task that still holds a container
  of the same name. `--preflight-only` runs the ticket-free checks alone, e.g.
  as `hwlock.py --preflight`.
- `--binary` defaults to `cuteafd` on PATH, else the arm's own sealed slot
  binary (`<WIP_ROOT>/slots/<slot>/coordinator/workspace/.cuteafd-wip/cuteafd`),
  resolved per arm, so a PATH without cuteafd needs no manual flag; a missing
  slot binary names the path to build.
- A state directory holds one run. `--resume` keeps passed jobs and reruns the
  failed or missing ones, moving the earlier attempt to `attempt-N/`. Without
  `--state`/`--task`, each run gets a fresh time-stamped state.
- `summary.md`: before -> after per card and metric with ratios and paired
  medians, then each job's spec/cache/trace/profile status and its first real
  error (the launcher's own line first). Failures are classified as preflight,
  launch, quality or measurement.
- `--build` writes a staging config: the first card's model with its
  `--set`/`--arm-set` overrides, no serving topology (export TP and families
  come from every card's scopes), and dev images tagged with the revision's
  toolchain identity.
