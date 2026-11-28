# WIP hardware cards

Every launch conditions the shared draft policy before timing with the server's
cached warm-prelude receipt: two fixed code/prose/JSON triples at the sweep
output length, identical nonces across arms. Both report and sweep retain the
receipt; an older server without this endpoint is refused. Per-point warm waves
remain, and Nsight windows exclude the prelude. This normalizes preceding
benchmark traffic rather than claiming widths have converged.

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
ledger peaks and authenticated console stage events. Spark memory is sampled
from host `/proc/meminfo` at the same 50 ms cadence, without a CUDA context:
`memory-spark-rankN.jsonl` retains byte-valued fields and sampled phase labels
(refreshed once per second; console snapshots can be up to two seconds old,
so these are not exact wave boundaries). Additional host samples at readiness
and after each sweep go to `memory-spark-rankN-ready.jsonl` and
`memory-spark-rankN-post-sweep.jsonl`, retaining their boundary provenance.
Each result's `spark_memory` reports per-rank and per-phase minimum MemAvailable
and field peaks; the text summary also lists minima. Streams end with the card,
on observer EOF, or after a lost heartbeat. Optional `--spark-memory-guard`
stops only this card's mount-verified WIP worker below 512 MiB MemAvailable;
it never stops a foreign container. Explicit probes fail
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
the flag keep their existing prompts. For cross-version pairs, use a current
shared `--binary` and `--panels decode_content,concurrency,spec_lossless_set`
(or either speed panel alone, without the basic card). The host runs the same
six untimed conditioning requests through the server endpoint when present,
or replays them as chat on a legacy server. It records `server-v1` or
`client-v1` per arm and compares admitted prompt token hashes per timed wave;
prechecks, conditioning, the closed-batch warm waves and arm-local lossless
probes are recorded but not compared. An unsupported panel combination uses
the ordinary whole-run hash contract and still needs the server endpoint.

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

- `--kit gate` is the in-repo kit (`scripts/bench/kits/gate`): e.g. mimo26-flash-max,
  qwen38-exl3-min, glmf-mc-original-max (2 RTX, 0 Sparks, EXL3 K4-MC) and
  v41-flash-max on `examples/configs`, with ports of their own. Every card loads a
  checkpoint that is a live `benchmarks/current-matrix.json` cell
  (`test_gate_kit_cards_load_live_matrix_checkpoints`).
- `--panels LIST` replaces every card's profile (`baseline` alone for the basic
  card and quick quality, `none` for no server profile). `--levels 1,16` adds a
  client-side code-decode sweep after it (one warm and one measured wave per
  level); `--post-load-c1` measures C1 again after the widest level. The sweep
  runs in the smoke binary, so arms with different server builds measure the
  same workload; pass `--binary` built from a checkout that has it (wip-cards
  checks). Add `--panel-config 'concurrency={"sustained":true,"multiplier":4}'`
  for separate continuous-refill rows at those levels (or in the server's
  `concurrency` panel). The steady window excludes initial warm-up and final
  drain; output-only tok/s includes the time spent on refill prefills. See
  [benchmark panel configuration](options.md#benchmark-panel-configuration)
  for window metadata and the explicit uniform-emission overlap estimate.
- `--panels none --verify-shapes '4x8;5,4x7;8x8;4x8@64'` replaces the decode
  sweep with warmed verify-only exact-shape timing (#2460): each `;`-separated
  shape lists per-request verify rows (`ROWS` or `ROWSxCOUNT`, so `5,4x7` is C8
  at R=33), optionally `@B` to pad to a forced bucket B. B must be a bucket
  serving itself pads to (a fixed point of the family's bucket map: the padded
  ladder, or every row count under exact-row graphs) and at least R's natural
  bucket, so same-R/different-B isolates padding cost without new graph
  geometry. The server runs the shapes through its production verify step
  (captured graphs, real padding) on cold scratch sequences of
  `--verify-context` (default 1024) prefilled tokens each, one untimed pass,
  then `--verify-repeats` (default 20) timed steps per shape in one load. Only
  an idle server takes the probe, whose scratch pages must fit without
  evicting retained prefixes. Sequences are reset after every step and
  nothing is committed. A greedy reference decode before the probes,
  restored from the prefix cache after them, must reproduce every token and
  row hash, or the card fails `measurement`. `sweep.json` `verify` and the
  summary's `verify_shapes` give per shape R, B, natural B, contexts,
  synchronized verify and selection medians, and graph captures/misses/eager
  runs during the timed steps (`graphed` is true only with none). With
  `--set CARD:PROFILE=nsys` the window frames the timed pass, which carries NVTX
  ranges per sample (`verify SHAPE R= B=`), per layer segment (`L<i>
  attention`, `L<i> attention+dense`, `head`), MoE (`L<i> moe R/B`, `router`,
  `experts`, `exchange wait`; GLM 5.3 `L<i> experts exchange R/B`) and `select`.
  They are host ranges around graph launches: with node tracing, a captured
  segment's kernels fall in its range, so attention/dense versus MoE/exchange
  split per layer, and kernel names split inside a segment. Under a head
  split, the second GPU's segments fall only in the sample range: split them
  by device. Families: GLM
  Flash and GLM 5.3; another family adopts it by implementing
  `shared::probe::verify_timing::VerifyTarget` over its serving verify step.
- `--panels baseline,spec_lossless_set` adds the lossless merge gate: every
  case of `scripts/bench/fixtures/spec-lossless-set-v1.json` (or
  `--lossless-set FILE`, `[{id, prompt, nonce}]` with the full nonce string)
  runs cold with drafts on and off for 128 greedy tokens and must match in
  token ids and every row hash, with no tie tolerance. The row fails as
  `quality` naming the failing cases; each case's probe pair (prompt ids,
  generated ids, row hashes, top two) stays in `report.json` under the pass's
  `evidence`. The baseline's single `spec_lossless` check is unchanged.
  `--lossless-concurrency N` (default 1) runs the drafts-on cases N at a time,
  then the drafts-off cases N at a time (never together), so the shared draft
  row budget is exercised; the pass records `concurrency` (N, per-wave max
  in-flight).
- `--panels none --client 'ARGV'` runs a client workload (for example
  `bench-agentic-session.py record`) against the live server once the card is
  ready, inside the card's turn and before teardown. It is smoke's precheck
  hook: it runs in `JOB/client/` with `CUTEAFD_SMOKE_URL`, `CUTEAFD_API_KEY_FILE`
  and `CUTEAFD_CLIENT_OUT` exported, and a failing client fails the card.
  `--client-timeout` (default 3600 s, at most 7200 s with any precheck) bounds
  it. It excludes `--exact-state` and `--cold-window`, and `--dry-run` prints the hook.
  Greedy byte-exactness: `--client 'scripts/bench/greedy-exactness.py client scripts/bench/fixtures/spec-lossless-set-v1.json'`
  (levels 1,4,10; `--levels` overrides) writes `greedy.json`; `greedy-exactness.py compare A.json B.json`
  diffs two arms per level and prompt, reports cross-level matches, and exits nonzero on any mismatch.
- `--sweep-passes N` (default 1) repeats the measurement N times against one
  server launch, for graph-cache convergence: pass 1 captures shapes for the
  first time, later passes should show no new captures. The client sweep
  (`--levels`, `--prefill-lengths`, `--verify-shapes`) runs N full passes,
  warm-ups included, and `context_load` runs N server-side passes. `--repeats`
  instead relaunches the server per repeat. `sweep.json` keeps every pass under
  `passes` (each with its points, prefill, phase thermals and graph bands, the
  #2649 deltas with #2670 sites and thermals); its top level is the final pass.
  `summary.json` lists every pass in `passes` (and per row in `rows[].passes`),
  the Graph Captures table gains a Pass column, and headline speed is the
  final pass, labelled so in `summary.md`. A measurement window (nsys, cold
  purge, exact state) opens before pass 1 and closes after the last pass.
  `run_timeout_s` bounds each pass. A runner that records fewer passes than
  asked fails the row as `measurement`. This is a bench option, not a serving
  key.
- `--prefill-idle-seconds N` waits after each ordinary prefill warm-up before
  the timed request (default 0). Each prefill point records `idle_seconds` in
  `sweep.json`; the wait is outside TTFT and the timed thermal window.
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

Prefill sizing uses the served tokenizer, including the chat template, to
reach the exact requested total prompt-token count. Warm-ups have the same
length; a timed usage mismatch fails the run, never silently clamps it.
`--prefill-lengths 8192` also records **8K w/tail** (`3 * served unit rows + 40`,
8,296 at 2,752 rows/unit) in `sweep.json` and the development summary. The basic
profile records both in `report.json`, but only exactly 8,192 appears on card
images and README tables. Points record their execution-unit shape beside
actual prompt tokens. Earlier approximate-8K results are not directly comparable.
