# Bounded graph cache family rollout

The shared controller is `shared/decode_graph/device.rs`; typed executable
ownership stays in `shared/decode_graph/bank.rs`. GLM 5.3 is the reference
integration in `families/glm5/engine.rs`. Do not copy its model geometry into
shared policy.

Families using `LayerGraphs`, `RowGraphs`, or `BindingGraphs` opt into the same
controller via `attach_device` in `shared/decode_graph/{native,binding}.rs`.
Unattached wrappers keep
their existing decisions, budgets and statistics unchanged. After attachment,
use `cache_ticket` before beginning capture (None means eager), then
`cache_insert` with native topology before `cached_launch`. A ticket cancels
itself on every failed path, including begin/end-capture errors. Attached
wrappers must not call legacy `insert`: it rejects unbudgeted insertion.
The family traverses every wrapper with `cache_cycle` and, only after all
ranks have queued and drained the complete step, calls `cache_measured` once
per physical device and `cache_drain` on every bank. `cache_tick` flushes
pinned-hit counts at boundaries without graph growth. No wrapper synchronizes
an individual capture or invents model geometry.

`CacheAttached` is the object-safe attachment/cycle/measurement/drain surface
for heterogeneous family traversal; direct banks and all three adapters
implement it. A family visitor supplies the physical device, stable site name,
and explicit peer-span flag alongside `&mut dyn CacheAttached`. Attach before
any capture, and pass the same device reserve and basis for every visited bank.
The visitor does not drain or measure on its own; the top-level completed-step
boundary supplies those callbacks after every consumer has drained.

## Family inputs

- Supply a `Site` name, default `Cycle`, `Preference`, bootstrap byte estimate,
  device-wide `Basis`, and explicit `spans_devices`. CUDA memcpy inspection
  cannot discover peer pointers embedded in arbitrary kernel arguments.
- Attach every bank on one physical device to the same reserve and basis.
  Node rates must match across banks. Use calibrated architecture/family node
  and executable coefficients only for their measured route; otherwise select
  bytes for the entire device and warn. An unknown node count runs eagerly;
  never mix byte estimates into a node pool.
- Before each decode/verify/prefill/wave boundary, call `cache_cycle` with the
  actual cycle. Pinned hits accumulate locally without controller locking or
  ARC-list mutation; changing cycles flushes the previous cycle's hits.
- Supply an exact typed key and retained executable/resource owner. On a miss,
  use `cache_miss`; Eager means execute normally, never reject the request.
  Cancel a ticket on every failed capture path. After successful instantiate,
  report native nodes/width/span and insert the owner before launching it.
- Supply a checked-in versioned prewarm list with stable geometry and explicit
  pin flags, plus `reserve_base_bytes` and `reserve_extra_bytes`. Prewarm is
  best effort, not a seal. First-touch eagerly before sampling capture growth.
  Attribute startup activity to Other rather than decode/verify frequency.
- Admission owns the full configured graph reserve once, before KV sizing.
  Measured warmup does not replace or shrink that budget. GPU_RESERVE cannot
  remove it. A scalar GPU_GRAPH_RESERVE applies independently to each GPU;
  gpuN selectors override individual coordinator ranks. Zero runs eagerly.

## Measurement and lifetime rules

Queue the entire multi-rank step before synchronizing any rank. Peer segments
may wait for work not yet queued: never synchronize around an individual
capture, instantiate or segment. Sample each physical device separately at the
completed-step boundary, then report only that device's observed growth.
Drain every consumer before destroying a selected executable or its resources.
Error paths use the same submission-drain lifetime proof.

Historical node/executable peaks advance only for successful surviving owners
at a completed measurement boundary, never when reserving a ticket. Cancelled
captures retain pending growth until sampling because a failed instantiate can
still grow driver storage. Destruction usually frees zero visible bytes; the
measured graph charge is permanent high-water, while ARC frees live capacity.
Chain reuse discounts apply only to qualified single-device chain captures.
Non-chain and explicitly peer-spanning sites reserve each capture separately.

Estimates select capture or eager and victims; measured growth decides charge.
Underestimation logs an overshoot and prevents additional growth, not request
refusal. Do not claim an arbitrary native allocation is hard-bounded by its
estimate. Check both warmed residency and reserve plus explicit capture
headroom on the family's qualification run.

## Trace and prewarm refresh

Launch the normal card plus concurrency-variance sweep with GRAPH_TRACE set
to an absolute host path whose parent exists. The launcher writes one
`<stem>.device-N.json` snapshot per physical device. Preserve the trace beside
the card results. It contains schema_version, cache charge/reserve/basis/pools,
site measurement classes, and identities with count, per-cycle counts,
first-seen time, native maximum ready-frontier width and device-span flag.
Schema 2 adds hits, misses, captures, evictions, recaptures (successful captures
after eviction), ghost_hits (misses found in ARC history), last_hit_ms and a
rolling hit_histogram: at most 60 intervals of 10 seconds, oldest first, with
start_ms and interval_ms. The final interval is partial. Lifetime counters are
not windowed; telemetry lives only as long as its resident/ghost ARC record.
Forgotten ghosts lose their history. `/v1/stats` graph_cache includes top_hits
and top_churn (evictions + recaptures), up to 16 identities each per device,
with the same fields even without tracing. telemetry_bytes reports the owned
per-key statistics payload, including histogram rings, not allocator overhead.
Pinned replay is lock-free: per-key counts flush at cycle/boundary with that
boundary's time. ARC hits use the clock sampled at cycle entry or boundary,
not a clock read per launch. Timestamps/windows are cycle/boundary-attributed,
not launch-exact; sampled_at_ms identifies the device snapshot time.
Miss identities expose bank-local key_id and are interned by exact typed-key equality; hash collisions and
identical display strings cannot merge keys. Display strings are built once
per interned key, not per miss. Interned keys are pruned with forgotten ARC
records at boundaries and cleared at teardown.

Each row includes capture_bytes (the latest proportional share of measured
step-boundary arena growth, null before capture and possibly zero on reuse),
nodes (native count when inspected, otherwise node-model count or null),
executables, basis, and permanent_bytes (the
unpacked node/exec model estimate or byte-based ARC size). These are not extra
physical charges: shared 2 MiB driver pools remain charged once per device.

Generate a proposal with:

```sh
python3 scripts/dev/graph-prewarm-from-trace.py --family glm5 \
  --trace /absolute/trace.device-0.json --output /absolute/proposal.json
```

The generator aggregates segment identities into geometry frequency, excludes
startup Other counts when per-cycle counts exist, takes the maximum across
overlapping snapshots/devices, and selects requested frequency coverage.
Schema 2 proposes pins when at least 60% of at least three complete observed
intervals have hits, or when a key has at least two evictions and two recaptures;
churning candidates survive the frequency cutoff. Inspect rolling heat and
request-cycle counts together: startup or an early burst alone is not evidence
of sustained request demand. Schema 1 retains its legacy hot minimum-parameter
heuristic because it has no interval/churn data. Each proposed pin prints a
per-device observed permanent-cost lower bound, observed reserve-fraction lower
bound, measured capture share and model basis to stdout (unknown for legacy
traces). A geometry pin covers every layer/residual/head variant, while bounded
traces may omit those variants: the full permanent_bytes and reserve_fraction
are explicitly unknown, never presented as the observed subset's sum.
Heat and costs use only the latest complete snapshot's identity set per device;
forgotten identities cannot keep old heat alive. The output list keeps its
consumer schema.
The base is the maximum measured physical
graph high-water, rounded to 2 MiB; extra is one largest measured capture,
separately rounded. Review the proposal, update the family's checked-in list,
and relaunch before accepting it. Extend the parser/consumer for each new
family rather than pretending its keys are GLM keys.

The GLM reference list comes from the trace6 ordinary EXL3 K4 1-RTX/4-Spark
baseline plus C1/2/4/8/16 sweep: 28 geometries cover every one of 151,522
request-cycle graph events (startup Other excluded), with 1x16 pinned. The
provisional 96-geometry arena was 1,189,085,184 bytes. All 7,742 inspected
executables were width-one and single-device. The final fixed-DRAFT_LENGTH=4,
1M-pool dev/candidate C1/C16 relaunch qualified the 28-shape list: its measured
prewarm arena is 335,544,320 bytes, with 855,638,016 bytes retained for runtime
lazy captures under ARC (including wider long-context tables). The total stays
1,191,182,336 bytes. There were no runtime captures, eager runs, evictions,
refusals or OOMs. C1/C16 ratios were 0.988/0.997 with identical verify widths
and no recorded throttling; KL/top-1 and byte-exact prefix/speculation checks
passed. The measured route used the byte fallback because it onboarded local
experts; this is not a qualification of the remote-only node-rate route.

## Delete the old family policy

Remove startup sealing, declared-key request refusals, per-bank lazy budgets,
and old census replacement of admission graph owners from the integrated
family. Retain any declared-key census used solely for provenance/bootstrap,
clearly separate from the fixed reserve. Shared legacy APIs remain only for
families not yet migrated; do not remove another family's working path in a
mechanical rollout.

## Reconciliation and Spark status

GLM publishes memory_reconciliation schema_version=1 with Ready and
first_request stage objects. Each device row carries measured_bytes,
charged_bytes excluding release padding, padding_bytes, owners, provenance,
measurement_method and unavailable_reason, plus three signed byte fields:

- `delta_vs_admitted`: observer-excluded CUDA used minus all admitted serving
  Growth owners (padding excluded), including lazy Workspace/Staging,
  allowances and derived tails. Negative values can be unused admitted capacity.
- `graph_reserve_slack`: admitted graph reserve minus measured graph arena
  high-water; negative means the arena exceeded its reserve. The component
  values are `graph_reserve_bytes` and `graph_arena_high_water_bytes`.
- `delta_vs_measured`: delta_vs_admitted plus graph_reserve_slack, replacing the
  admitted graph budget with actual measured high-water. This is the
  accounting-error signal, not the admitted-owner delta.

For RTX, `drift_flag_eligible` is true and `beyond_half_padding` is true only
when abs(delta_vs_measured) exceeds half padding_bytes. A negative admitted delta exactly equal to unused graph reserve
is not drift. Missing physical samples yield null deltas/drift. Missing arena
telemetry for a nonzero graph owner yields null measured delta/slack/drift and
an explicit graph_measurement_unavailable_reason, never an assumed zero.
Devices with no graph owner and no controller have zero arena/slack.
RTX rows retain raw_used_bytes for audit and report observer_bytes plus
observer_provenance (the measured sampler context). The same observer exclusion
as cuda_allocatable_memory_info applies exactly once before either delta; absent
observers contribute zero. Device-wide CUDA use is not process attribution.
Ready still includes unused admitted Growth owners, so a negative Ready delta
need not be an accounting defect. GB10 admission MemAvailable/allocatable samples
remain separately labelled.

Spark Ready aggregation uses the reconciled worker resident/workspace and UMA
host-owner ledger, not the coordinator's empty placeholder plan. Device/pinned
detail aliases are not charged twice. Its drift basis is launch-relative:
allocatable free at the post-media/pre-expert retained-ledger baseline minus
allocatable free at the admission snapshot, less owners allocated after that
baseline. Preexisting system/page-cache occupancy cancels; changes during loading
can still perturb this system-wide witness. Signed consumption can be negative.
The admission snapshot precedes endpoint-ring creation: admitted future ring bytes
are reported separately and excluded from this snapshot's charge. Missing baseline
or Ready samples yield null deltas/drift with a reason, never an absolute fallback.
Raw CUDA used and absolute allocatable used remain diagnostic fields only (#1568).
Launch-relative Spark residuals also remain diagnostic until #1568 resolves
ownership and the system witness: the final four ranks measured +1.84, +1.98,
+3.87 and +2.44 GB. Spark rows set `drift_flag_eligible=false`,
`beyond_half_padding=null` and `drift_flag_unavailable_reason` linking #1568;
retain signed deltas but do not drift-flag them in recorders. This does not
change Spark graph behavior or substitute absolute occupancy for a missing baseline.
Baseline phase and sample provenance live in the optional memory_reconciliation
object. New coordinators accept missing and unknown optional detail fields;
coordinator and Spark release images are a compatible
pair, not arbitrary old/new binary combinations. Post-request Spark samples
at the coordinator are null with the #1631 reason until control-plane telemetry
exists. Worker first-request logs include their own raw CUDA sample and a
separately labelled ledger-only residual, not a fabricated planner delta.
Recorder integration for cards/sweep.json is owned by a separate task.

The live native Spark worker routes through service/backend.rs Execution
launch/flush paths, not the dormant capture/replay methods in experts/execution.rs.
There are no live worker graph sites to attach today. SPARK_GRAPH_RESERVE is
forwarded but has no active worker graph owner to enforce; a family enabling
worker capture must add admission ownership and this controller together.

## Qualification checklist

Run controller, bank ownership, capture cancellation, basis mismatch, cycle
attribution, optional Spark detail and signed-delta tests; workspace all-target
check; CUDA-ON all-target native build; default CPU gate. Measure a real card's
no-refusal/no-OOM behavior, eager/eviction counters, measured graph high-water
versus reserve/headroom, Ready/first-request deltas, and same-day warmed C1/C16
speed with thermal conditions. Report missing telemetry explicitly. Preserve
logs/results and stop every task-owned container before releasing its ticket.
