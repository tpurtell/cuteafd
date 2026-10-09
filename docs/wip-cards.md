# WIP hardware cards

Use `scripts/bench/wip-cards.py` instead of copying an RC kit driver. It reads
`cfg/*.config` and `matrix-*.json` from the kit (default: release-v2-rc2), writes
private one-card matrices and a 0600 API key, and runs `cuteafd bench smoke`
under `setsid` with numeric exit files. Smoke alone owns serving locks. Use a
current CLI containing the opt-in observer/precheck hooks for probes.

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
raw/corrected CUDA 50ms peaks (NVML sampler-context subtraction), worker ring
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
