# The shared engine

There is one engine. A model family supplies facts and hooks: its geometry,
attention, router, experts, head, speculator taps, a lane entry with its
reason, and a prefill estimate. The shared code owns the serve loop, the decode
lanes, the prefill schedule, the Spark exchange, the Spark compute owners and
admission. This page maps those shared parts to the code. Read it before you
port a family (MiniMax Flash is next) or change shared code. Why the engine is
built this way: [PLAN "One engine, shared structure"](../../PLAN.md#one-engine-shared-structure-tj-2026-10-25).
Working rules: [AGENTS.md](../../AGENTS.md).

Line-numbered pointers are at dev `5cd3295a1` (2026-10-29); the graph-cache
section uses file/symbol pointers without fixed line numbers. `dev` is the
integration branch; `main` moves only at tagged releases. Lines drift, so each
one also names a symbol you can grep for. Short prefixes used below:

| Prefix | Path |
|---|---|
| `daemon/` | `rust/crates/cuteafd-daemon/src/` |
| `loader/` | `rust/crates/cuteafd-loader/src/` |
| `core/` | `rust/crates/cuteafd-core/src/` |
| `transport/` | `rust/crates/cuteafd-transport/src/` |

## 1. One engine, families supply facts

### The family descriptor

`Family` ([loader/plan/family.rs:113](../../rust/crates/cuteafd-loader/src/plan/family.rs#L113))
is a static, CPU-only descriptor. It is registered in `REGISTRY` (`:162`).
`cuteafd plan`, serve and golden all read the same descriptor. It states:

- **`decode_lanes() -> LaneSupport`** ([loader/plan/decode_lanes.rs:19](../../rust/crates/cuteafd-loader/src/plan/decode_lanes.rs#L19)).
  The three variants:
  - `One { reason }`: the reason is a measurement or a predicted cost, with
    its issue number. The planner prints it when `--decode-lanes 2` is
    refused (`resolve`, `:40`).
  - `OneOrTwo { default }`: resolves to one lane at concurrency 1 (`:58`).
  - `FixedTwo`: V4.1 only.

  The trait default, `DRAFTER` (`:15`), is for drafter checkpoints only. Two
  tests enforce this: `one_lane_families_record_their_reason` (`:103`) and
  `only_drafters_keep_the_default_reason` (`:116`).
- **`spark_decode() -> Option<SparkDecode { rows }>`** (`family.rs:128`): the
  rows of one decode/verify request to a Spark. `None` means the family has
  no Spark experts.
- **`prefill_limits() -> PrefillLimits`**
  ([loader/serving_capacity/prefill_waves.rs:58](../../rust/crates/cuteafd-loader/src/serving_capacity/prefill_waves.rs#L58)).
  Three parts:
  - `estimate: PrefillEstimate` (`:22`), which is one of:
    - `Timing { spark_ms, turnaround_ms, reference_ranks, reference_unit_rows, source }`:
      measured S and T.
    - `MeasuredWaves { waves, reference_ranks, larger_topology_waves, source }`:
      a measured best k, used when there is no independent S/T.
    - `Unmeasured`: two waves.
  - `initial_unit_rows`: a measured starting unit (Step-5: 2048).
  - Hard executor caps: `max_waves` (Qwen4: 2, because its mapped-PLE ring
    has two slots), `max_unit_rows`, `capacities`.
- **`open() -> FamilyModel`** (`family.rs:55`), which carries the tensor
  contract (`accepts`, `experts`) and `placement()` (`:70`). `placement()` is
  the family's adapter over the shared solver (section 5).

Worked examples:

- GLM 5.3 / GLM Flash: [loader/plan/families/glm.rs:28](../../rust/crates/cuteafd-loader/src/plan/families/glm.rs#L28)
  (`GLM5`: `Timing`; `GLM5_FLASH`: `MeasuredWaves`).
- DeepSeek: [deepseek.rs:25](../../rust/crates/cuteafd-loader/src/plan/families/deepseek.rs#L25).
- Step-5: [step5.rs:339](../../rust/crates/cuteafd-loader/src/plan/families/step5.rs#L339).
- Kimi: [kimi/mod.rs:254](../../rust/crates/cuteafd-loader/src/plan/families/kimi/mod.rs#L254).

Every `source` string links the measurement it came from. Never put a family
name, Spark-count threshold or per-family row cap in shared code: add a
descriptor field instead.

Current entries:

| Family | LaneSupport | Prefill estimate |
|---|---|---|
| V4.1 | FixedTwo (C8 1.17×, C16 1.19×, #723) | MeasuredWaves k2 (`max_waves` 2: one target pass per slot), initial unit 2048 |
| V4 | One (V4 Flash two-lane A/B pending) | MeasuredWaves k3 @ 6 |
| GLM 5.3 | One (second graph set ≈3 GiB/GPU) | Timing S 13 ms, T 19.5 ms @ 6, 2752 rows |
| GLM Flash | One (min-card two-lane A/B pending) | MeasuredWaves k2 @ 4, k3 above |
| MiMo | One (measured 0.66–0.93×) | legacy (private prefill, below) |
| Kimi K3 | One (predicted loss) | MeasuredWaves k3 @ 6 |
| Step-5 | One (needs its own decode link) | MeasuredWaves k4 @ 6, initial unit 2048 |
| Qwen4, dense Qwen | One (no Spark round trip) | Unmeasured, `max_waves: 2` |

### What a family must not do

- **Run a private serve loop.** Decode rounds go through `LaneSet`
  (section 2), and prefill goes through `run_prefill_units` (section 3). Each
  port adds a source-gate test that pins this; copy the pattern in
  `decode_runs_as_lane_zero_of_a_shared_lane_set`
  ([daemon/families/deepseek_v4/serve.rs:1036](../../rust/crates/cuteafd-daemon/src/families/deepseek_v4/serve.rs#L1036)).
  MiMo's prefill (`prefill_lane_count`, [daemon/families/mimo_v2/engine.rs:136](../../rust/crates/cuteafd-daemon/src/families/mimo_v2/engine.rs#L136))
  is the one remaining private loop, and its port is open (#744, #671). Do not
  copy it.
- **Own a placement policy.** A family builds a `PlacementRequest` from its
  geometry and calls the shared solver. It does not pick pool sizes, onboard
  counts or wave depths itself.
- **Leave compat shims.** A port deletes the family's old loop, workspaces and
  launch keys in the same change. It leaves no dormant path, removed-key
  diagnostic or alias. If a feature truly cannot fit the shared structure, name
  it and the reason on #744 instead of forking the path.

Qwen-shaped families (Kimi, Step-5, Qwen4, dense Qwen) go further. They
implement `Backend` ([daemon/shared/qwen/scheduler.rs:93](../../rust/crates/cuteafd-daemon/src/shared/qwen/scheduler.rs#L93))
and use the shared scheduler `schedule()` (`:618`) unchanged: admission,
prefix snapshots, drafting, sampling and the console. A new family should start
there unless its request lifecycle truly differs.

## 2. Decode: lanes, commit, pacing, readiness

All of this is in [daemon/shared/serve/lanes.rs](../../rust/crates/cuteafd-daemon/src/shared/serve/lanes.rs).

- **`LaneSet`** (`:19`) holds one stop flag (`Cell<bool>`) for its lanes. All
  lanes are polled on the engine thread. Lanes share request banks through
  `RefCell`s, borrowed only for synchronous work and never across an await
  (`:3-8`).
- **`LaneSet::run_one(lane)`** (`:53`) runs the only lane of a
  `LaneSupport::One` family as lane 0. It uses the same lease, failure stop and
  pacing as a pair. Every family except V4.1 calls it once per round from its
  scheduler, for example:
  - [qwen/scheduler.rs:1039](../../rust/crates/cuteafd-daemon/src/shared/qwen/scheduler.rs#L1039)
  - [glm5/serve.rs:772](../../rust/crates/cuteafd-daemon/src/families/glm5/serve.rs#L772)
  - [mimo_v2/serve.rs:934](../../rust/crates/cuteafd-daemon/src/families/mimo_v2/serve.rs#L934)
- **`LaneSet::run(first, second)`** (`:42`) runs two lane loops through
  `join` (`:62`), which uses `join!`, not `try_join!`. A failing lane sets the
  stop flag (`LaneLease::run`, `:113`), and its peer returns at its next
  completed boundary. A peer is never dropped mid-round, because it may own
  queued device or transport work. The first lane's error wins. V4.1 is the
  only two-lane user ([deepseek_v41/v41_native_serve/scheduler/independent.rs:17](../../rust/crates/cuteafd-daemon/src/families/deepseek_v41/v41_native_serve/scheduler/independent.rs#L17)).
  The scheduler also uses `stop()` to pause both lanes for admission. After
  that, `rebalance` (`:90`) and `decode_lanes::assign` (`decode_lanes.rs:66`)
  keep the lanes within one request of each other.
- **`try_map(setup)`** (`:36`) constructs each lane's owners with the lease
  that later runs the lane, so lane index = owner identity end to end. For
  example, V4.1 opens its Spark endpoint per lane with `lane.index()`
  ([v41_native_serve.rs:453](../../rust/crates/cuteafd-daemon/src/families/deepseek_v41/v41_native_serve.rs#L453)).
  The Spark worker then routes decode rows of endpoint lane *n* to decode owner
  *n* (`execution_owner`, [transport/worker_selection.rs:84](../../rust/crates/cuteafd-transport/src/worker_selection.rs#L84)).
  A second lane's bytes are a planner charge made before allocation and
  labelled per lane: see the `lane {lane} …` demands in
  [loader/placement/families/deepseek_v41/owners.rs:179](../../rust/crates/cuteafd-loader/src/placement/families/deepseek_v41/owners.rs#L179).
- **`LaneCommit` + `lanes::commit`** (`:122`, `:136`) end every round. A
  round's commit runs in four steps:
  1. `begin` enqueues both halves: the target's cache or state commit, and the
     drafter's context update.
  2. `ready` is polled, yielding to peer lanes, until both halves have drained.
  3. `publish` makes the commit visible.
  4. On any failure, `abort_and_drain` runs before storage is revoked, and the
     original error is returned.

  The four implementations show the range: V4 commits by position inside the
  round, so its commit enqueues nothing; GLM Flash replays KDA kept rows; Qwen
  accepts; V4.1 is the two-lane case. Look them up with
  `grep -n 'impl.*LaneCommit for' -r daemon/`.
- **Pacing.** `paced` (`:78`) wraps every lane future in a `Backoff`
  ([transport/wait_policy.rs:115](../../rust/crates/cuteafd-transport/src/wait_policy.rs#L115)).
  It works as follows:
  - Under `--host-wait block` (the default), a pending stretch spins for
    `BLOCK_SPIN` = 10 ms (`:92`), then sleeps `BLOCK_SLEEP` = 20 µs per poll
    (`:94`). A completed poll restarts the spin.
  - Why 10 ms: every per-layer decode wait ends inside that window, so decode
    never pays a wake-up. The earlier 250 µs window cost Kimi BF16 max C8
    0.91× (#731).
  - `spin` never sleeps.
  - Both modes park after 30 s idle (`:1-21`).

  Do not add a family-private spin or sleep: use `Backoff` or `spin_window`
  (`:103`).
- **Readiness.** Signal ready only after every fallible init step, including
  prefix-cache construction and capacity publication. MiMo once signalled ready
  before its prefix cache existed, and its first card recorded pool = 0 (#791).
  Where the rule lives:
  - Shared scheduler: `initialize(init, ready)` ([qwen/scheduler.rs:476](../../rust/crates/cuteafd-daemon/src/shared/qwen/scheduler.rs#L476)).
  - Own schedulers: a single `ready();` after the last `?` and before
    `loop {` ([glm5/serve.rs:491](../../rust/crates/cuteafd-daemon/src/families/glm5/serve.rs#L491)).
  - The tests are in `readiness_tests` ([daemon/shared/serve.rs:6](../../rust/crates/cuteafd-daemon/src/shared/serve.rs#L6)).
    Add your family to them.

## 3. Prefill: continuous row units

### The executor

[daemon/shared/prefill_pipeline.rs](../../rust/crates/cuteafd-daemon/src/shared/prefill_pipeline.rs)
owns the schedule. A family implements `PrefillHooks` (`:170`). Each hook
enqueues work and returns:

- `route(layer)`: `Local` (dense or RTX-resident experts, finished inside one
  hook) or `Spark` (a wave sent in one hook and landed in a later one).
- `begin_unit`, `attention`, `local_ffn`.
- `send_experts -> Ticket`, then `land_experts(ticket)`.
- `end_unit`, then `commit_unit`, called in unit order.

The family then calls `run_prefill_units(hooks, units, waves)` (`:339`). The
`prefill_schedule` (`:264`) is pure, and the CPU tests check it. It enforces:

- unit k+1 attends layer L only after unit k attended L;
- waves land in dispatch order, and units commit in order;
- a newly admitted unit is served before older units land again, so the Sparks
  never drain between units.

Each in-flight unit owns its slot's slice (`unit.slot`) of every buffer that
lives across a hook. Hook-local scratch is shared. `prefill_units` (`:216`)
cuts the span into equal, 64-row-aligned units in multiples of `waves`. A span
that fits one unit splits only if each wave keeps `MIN_SPLIT_ROWS` = 1024
(`:206`).

Reference ports:

- Kimi (the first): `UnitHooks` at [kimi_k3/layers.rs:708](../../rust/crates/cuteafd-daemon/src/families/kimi_k3/layers.rs#L708).
- GLM 5.3: [glm5/engine.rs:2402](../../rust/crates/cuteafd-daemon/src/families/glm5/engine.rs#L2402).
- V4: [deepseek_v4/engine/prefill.rs:128](../../rust/crates/cuteafd-daemon/src/families/deepseek_v4/engine/prefill.rs#L128).
- V4.1 (async hooks, `run_prefill_units_async`): one target pass and Spark
  link per slot, every hook in the slot's device-ordered stage chain, at
  [deepseek_v41/v41_target_pass/encoder_units.rs](../../rust/crates/cuteafd-daemon/src/families/deepseek_v41/v41_target_pass/encoder_units.rs)
  and its two-RTX form in `distributed/encoder_units.rs`. The model's causal
  encoder/decoder (CED) stays family arithmetic: encoder layers 0–19 prefill
  the prompt as the executor's units, and the decoder layers 20–39 replay the
  last ≤128 rows once per prompt. The shared executor schedules both; there
  is no private V4.1 prefill loop.
- Step-5, Qwen4 and GLM Flash also implement it.

The ordering, row-ownership and head rules are in
[PLAN "Prefill contract"](../../PLAN.md#prefill-contract-tj-2026-10-23).

### Never block the issuing thread

The executor walks its steps on one thread, one hook at a time, so a hook that
waits on the host for its own unit's work stalls every other unit. On a
prefill or unit path, a wait for Spark replies (collect/land), a route sync, a
shared-expert or mHC finish, or a stream sync must be a device-side wait
(stream/event ordering: the stage chain,
[shared/memory/chain.rs](../../rust/crates/cuteafd-daemon/src/shared/memory/chain.rs),
`Waves::land`'s value wait) or happen once per unit at its end. It must never
be a host spin, sleep or `cuda_stream_synchronize`; in a sequential walk a
cooperative yield is no better, because nothing else is runnable.

Why: in #815, 5326add93 made V4.1's 1-RTX collect spin on the host until the
reply landed. That serialised units, and V4.1 prefill fell to about 0.5×, with
the Sparks idle half the time. The fix was #817, then V4.1's units in
device-ordered chains, linked across slots with `chain::record_into`/`depend`.

How to spot it: a `PROFILE=transport-timing,expert-timing` run where, per
Spark rank, CUDA compute is about equal to the GPU idle gaps between units,
staging and post per wave are small (≈0.1 ms / ≈1 ms), and posted→all-replies
is large (≈10 ms at 1024 rows).

### Waves and the 8192-row reserve

This is all in [loader/serving_capacity/prefill_waves.rs](../../rust/crates/cuteafd-loader/src/serving_capacity/prefill_waves.rs).

- **The reserve and units:** one fixed `RESERVE_ROWS` = 8192 for every
  family, or the planner's fit cap on a card that cannot hold it (the largest
  64-row multiple that fits; never shrunk for KV or onboarded layers). The
  depth is a fixed `DEPTH` = 3 (TJ 2026-10-28): units of roundup64(reserve / 3)
  rows, 3 × 2752 = 8256 in the standard reserve (the `RESERVE_ROWS + 64` cap).
  Zero Sparks use one unit; a family's hard executor capacity
  (`PrefillLimits::max_waves`, e.g. Qwen 4's two-slot PLE ring) caps it.
- **Pins:** `PREFILL_WAVES`/`PREFILL_ROWS` are plain config for measurement
  arms; a pin either fits or fails with a typed `PrefillError`.
- **Short prompts:** `waves_used` uses fewer waves when a prompt cannot keep
  each unit at least `MIN_FILL` = 50% full. Storage is not reallocated and
  graphs are not recaptured.
- **Why fixed:** the #762 adaptive controller (calibration, serving
  re-tests, per-call re-cut, four admitted slots) was removed after every
  measured layout ran best at three and the controller itself cost 5–7%
  (PLAN "Prefill contract").

**What waves assume of kernels.** A unit's Spark time must scale roughly
linearly with its rows across the whole 2048–4096 band. If a kernel has a
per-row cliff, larger units become silently slower, and that shows up as a wave
regression rather than a kernel problem.

The example that exposed this was Step-5's BF16-input FP8 kernel. It switched
GEMV→GEMM above 2048 rows, costing 33–54% more per row on GB10, so 3 × 2752
measured slower than 3 × 2048 ([#762 audit](https://github.com/tpurtell/cuteafd/issues/762#issuecomment-6413296191)).
The guard is the kernel row-curve rule in
[AGENTS.md "Build and run"](../../AGENTS.md#build-and-run). If a kernel keeps a
cliff, add a descriptor preference such as `initial_unit_rows` that avoids the
bad unit.

### Packing short prompts into one unit

[daemon/shared/prefill_packing.rs](../../rust/crates/cuteafd-daemon/src/shared/prefill_packing.rs)
(#847, #855, #858). When several short prompts are already queued, the
coordinator lays them out as adjacent segments of one unit (`plan`) instead
of one unit each. It never waits for more work to arrive.

- Attention runs per segment; rowwise and expert work run once on the packed
  rows. Each member's result and failure stay its own (`member_outcomes`).
- A drafter's tap buffer caps a pack (`Limits::taps`, the drafter's window:
  1024 rows for MiMo DFlash, 2048 for the shared DFlash2 ring).
- `PREFILL_PACKING=unrestricted` (default) packs across arithmetic route
  boundaries; `exact` caps a pack at the loaded kernels' route boundary (the
  fork publishes it through `cuteafd_fp8moe_arithmetic_route`) so a packed
  row's arithmetic equals running it alone. Use `exact` for byte-exactness
  validation.
- It is on by default wherever a family implements it, and every family
  should: the family's prefill hooks must run attention per segment. MiMo and
  GLM Flash do today; the other families are #847 phases 2–5.
- Not the same thing as Spark wave merging (below): packing joins short
  prompts on the coordinator; wave merging joins full units on a Spark.

### Deployment contract: units and span

`cuteafd plan --deployment` publishes:

- `prefill_unit_rows`, `prefill_waves`, `prefill_span_rows`,
  `prefill_reserve_rows` and `decode_lanes` (`DeploymentSpec`,
  [loader/plan/deployment.rs](../../rust/crates/cuteafd-loader/src/plan/deployment.rs));
- typed topology, roles and admission (the same struct).

The launcher reads those fields back in `release_prefill_shape`
([scripts/launch/run-family.sh](../../scripts/launch/run-family.sh)) and
refuses a plan that names no positive unit or wave count. The family's mode
comes from `family_prefill_mode`:

| Mode | Families | Serve flags |
|---|---|---|
| `units` | step5, kimi_k3, qwen4, deepseek_v4, deepseek_v41 | `--prefill-rows` (V4.1 `--prefill-batch-tokens`), `--prefill-waves` (V4.1 also `--prefill-reserve-rows`) |
| `units-span` | glm5, glm5_flash, mimo_v2 | the above plus `--prefill-span-rows`: unit × waves |

A new family opts in here once its serve CLI accepts the shape. Each slot's
storage is a view at `slot * unit` inside per-field arenas spanning the
slots' units; hook-local scratch is shared by the slots on a GPU, and the
planner charges exactly that. One prefill call is one window cycle of
`waves * unit` rows.

## 4. Spark exchange and typed owners

### Coordinator side: one exchange per link

[daemon/shared/spark_waves.rs](../../rust/crates/cuteafd-daemon/src/shared/spark_waves.rs)
(#731) gives each link one persistent transport worker. The worker owns the
link's QPs and carries every wave, decode and prefill alike. The GPU-issuing
thread never waits on it. The two meet through monotone u32 counters
(`staged`, `landed`, `consumed`) in pinned, device-mapped memory:

- the stream writes `staged` after its copies;
- `Waves::land` (`:816`) queues a stream wait on `landed`;
- the reduce writes `consumed`.

Wave k uses slot k mod waves, and a slot's next wave waits for its previous
one to land. All staging copies around these waits are SM copies through the
mapped alias, never `cudaMemcpyAsync` (the module doc, `:1-64`, explains why).

On a transport error, the worker records a terminal health failure and
releases every queued wait. Every consumer checks `healthy()` (`:125`) after
its sampling sync.

**Landing mode.** Ranks land in GPU memory (dma-buf) or pinned host memory,
which the probe decides per link (`CUTEAFD_SPARK_INTAKE`). The #815 regression
showed on both, so test any exchange change on both:
`v41-flash-sim5090` (TP4, pinned) and the TP3 5090 shape (`v41-flash-min` at
31.8 GiB, 1024-row units, dma-buf).

Links are `SparkLink::new_waves(..., waves, wave_rows)`
([daemon/shared/spark_intake.rs:1474](../../rust/crates/cuteafd-daemon/src/shared/spark_intake.rs#L1474)).
Every request and reply is one registered RC SEND, with no chunking and no
credit beyond ring depth ([PLAN, "Frames"](../../PLAN.md#prefill-executor-api-shared-2026-10-23)).

### Spark side: typed compute owners

`SparkExecutions` ([core/spark_executions.rs:44](../../rust/crates/cuteafd-core/src/spark_executions.rs#L44))
holds one small **decode owner per lane** (`DecodeExecution`, `:16`) and
exactly **one prefill owner per Spark** (`PrefillExecution`, `:25`):

- **Decode owners** are sized from `spark_decode().rows` at the covering export.
- **The prefill owner** has scratch at the export covering the admitted unit:
  - `buffered_units` = k retains each unit's own I/O, but scratch and the
    stream are not multiplied;
  - `MAX_BUFFERED_PREFILL_UNITS` = 4 (`:10`).

Plan, serve and golden all build owners through `admit_family`
([loader/plan/spark_executions.rs:113](../../rust/crates/cuteafd-loader/src/plan/spark_executions.rs#L113)).
The worker's backend export ladder (`SparkBackend::exports`, `:81`) fixes the
covering export. An uncovered capacity is a typed `ExecutionAdmissionError`,
not a fallback.

**FIFO units and credit.** On the worker
([daemon/shared/experts/service/local.rs](../../rust/crates/cuteafd-daemon/src/shared/experts/service/local.rs)):

- `execution_owner` (`:719`) maps every request to one owner. Prefill and
  benchmark rows go to the prefill owner; decode and MTP-verify rows go to
  `Decode(lane)`. A request that mixes owners is rejected.
- `OwnerQueues` (`:739`) publishes GPU completion in owner FIFO order.
- A completed unit keeps its I/O and its admission credit until its reply is
  released (`release`, `:783`).
- Decode lanes never borrow prefill credit, or each other's (tests at `:975`,
  `:998`, `:1036`).

**Completion: decode stream vs prefill ordering events.** `PendingUnits::for_owner`
([execution/timing.rs:49](../../rust/crates/cuteafd-daemon/src/shared/experts/execution/timing.rs#L49))
does the following:

- A decode owner holds one unit (`SingleFlight`), so a stream query is its
  completion fence.
- The prefill owner records one ordering-only event per queued unit
  (`Event::ordering`, `:18`, created without timing). An earlier unit's reply
  therefore never waits for later work on the same stream.
- Timing-enabled events exist only where profiling publishes elapsed times.

**Packed route staging.**

- Each queued unit packs its route ids and weights into its own pinned staging
  slot, which is uploaded with one contiguous H2D
  ([service/routes.rs](../../rust/crates/cuteafd-daemon/src/shared/experts/service/routes.rs),
  `pack` `:34`, `upload` `:51`).
- The slot stays owned until that unit's fence.
- Do not copy routes as strided D2D copies out of mapped request records: on
  GB10 that cost 1062 µs per 2752 × top-8 unit, against 7 µs packed
  (`f2f05dfa0`).

**Wave merging on the Spark (FP8/MXFP4, #844).** Units that arrive for the
same layer while the GPU is busy run as one expert launch:

- Each received unit is copied from the RDMA ring into per-slot device memory
  on a copy stream; compute waits on the copy's event. GB10 has no GPUDirect
  RDMA, and reading the mapped ring in place made FC1 4.9× slower.
- Staged units flush when the compute stream is idle, the layer changes,
  intake is blocked, or two units are staged: a merged wave is at most
  2 × unit rows ([fp8/worker.rs](../../rust/crates/cuteafd-daemon/src/shared/experts/fp8/worker.rs) `stage`,
  [service/local.rs](../../rust/crates/cuteafd-daemon/src/shared/experts/service/local.rs) `flush_after_sweep`).
- Row-address kernels gather FC1 rows and store combine rows per member
  address, so merged output is byte-identical to singleton launches.
- MiMo Flash 8K prefill 1.11×, MiMo Pro 1.04×, with expert launches 93 → 65
  per request on a Spark rank. A new FP8/MXFP4 family gets this from the
  shared worker with no family code.

**Ingress dtype.** Each endpoint has a typed `SparkIngress`
([core/expert_geometry.rs:57](../../rust/crates/cuteafd-core/src/expert_geometry.rs#L57)):
`Fp8K32`, `Bf16`, `Bf16Decode { rows }` (BF16 decode rows and FP8 K32
prefill on one QP), or `Native` (each layer's package input dtype). Two checks
keep the wire and the kernels in agreement:

- The worker checks the installed packages against the endpoint before serving
  (`validate_ingress`, [service/backend.rs:132](../../rust/crates/cuteafd-daemon/src/shared/experts/service/backend.rs#L132);
  `supports_ingress`, `:498`).
- Each request's header dtype must match the bound kernel's native input
  (`require_input_dtype`, [transport/expert.rs:322](../../rust/crates/cuteafd-transport/src/expert.rs#L322)).

`SparkIngress::Native` lets each layer use its package's native input, with
BF16-max ring and intake capacity and fail-closed package validation. Step FP8
uses FP8 K32 on layers 3–87 and native BF16 on layers 88–90; Step NVFP4 keeps
BF16 wire. Send the checkpoint's native activation format; don't convert on
the coordinator to fit an endpoint
([AGENTS.md "Engineering principles"](../../AGENTS.md#engineering-principles)).

## 5. Admission and the planner

- **Plan == runtime.** One CPU-only solver serves every family
  ([loader/placement/mod.rs:1](../../rust/crates/cuteafd-loader/src/placement/mod.rs#L1),
  `solve` in `solve.rs:85`). The planner (`FamilyModel::placement`,
  `family.rs:70`) and the runtime admission build the same `PlacementRequest`
  from the same family inputs. They differ only in the GPU `Baseline`
  (`mod.rs:145`): `Planned` charges context and loaded weights, and `Measured`
  is one CUDA sample. Spark owners use the same `admit_family`. A planner that
  disagrees with the runtime is a bug in one of them; don't add a margin.
- **Typed charges.**
  - Every demand has a `Category` ([core/memory_layout.rs:15](../../rust/crates/cuteafd-core/src/memory_layout.rs#L15)):
    `Weights`, `Kv`, `Workspace`, `Transport` and so on, plus a group label
    (for example `lane 1 backbone`).
  - The runtime allocation ledger uses the same labels, so a prediction can be
    compared with what was allocated (`compare`, `:327`). The ready report
    logs it (`memory_report::log("ready")`).
  - Weight and workspace admission happen before any allocation.
- **One memory policy.** `MemoryPolicy` ([core/memory_policy.rs](../../rust/crates/cuteafd-core/src/memory_policy.rs))
  is installed once per process from the global flags (`GPU_MEMORY_CAP`,
  `SPARK_MEMORY_CAP`, `GPU_RESERVE`, `SPARK_RESERVE`, `FAMILY_RESERVE`,
  `KV_TARGET_TOKENS`, `KV_MIN_TOKENS`, `KV_MAX_TOKENS`; [docs/options.md](../options.md)).
  - Each device has one cap: weights, KV, workspaces and the reserve all
    live inside it.
  - The reserve is the sum of named, measured components (CUDA context and
    cuBLAS, loaded code, graphs, runtime growth, the Spark runtime residual)
    plus the family's `FAMILY_RESERVE` constant (default 0; constants and the
    shared breakdown/log helpers live in [loader/placement/reserve.rs](../../rust/crates/cuteafd-loader/src/placement/reserve.rs)). `GPU_RESERVE` /
    `SPARK_RESERVE` override the generic reserve, not the separate graph owner
    ([Graph cache](#8-graph-cache)). No family keeps private headroom,
    an occupancy percentage or a graph-budget key.
  - Startup and `cuteafd plan` print one line per device: cap, weights,
    workspaces, every reserve component, and the KV target, min, max and pool.
- **Pool first.** `PoolPolicy` ([loader/placement/pool.rs:11](../../rust/crates/cuteafd-loader/src/placement/pool.rs#L11))
  takes the KV target (2M tokens on cards above 32 GiB, 1M at 32 GiB or less),
  min and max from the policy. The solver's order (`mod.rs:7-10`) is:
  1. fixed demands (graphs, workspaces) and mandatory movables;
  2. the pool target;
  3. RTX expert layers in what is left;
  4. with the layout fixed, the pool grows into what is left, up to the KV
     max; below the KV min the launch is refused.
- **Onboarding.** `Onboard::Auto` (`mod.rs:425`) is the default:
  - onboarding is on in every shipped config;
  - explicit `all` (every routed-expert layer of the target model) and `N` are
    pins; a pin that does not fit is refused with its count, never shrunk;
  - a zero-onboard config is only a labelled measurement arm, never a shipped
    default.
- **The 30 s budget.** Every full-plan test runs through `BudgetedPlanInput`
  ([loader/plan/testing.rs:12](../../rust/crates/cuteafd-loader/src/plan/testing.rs#L12)).
  It allows 30 s per plan (`:31`) and prints phase timings, so a planner that
  grows slow fails a CPU test, not a launch.
- **`cuteafd plan --deployment`** (cli flags at [daemon/cli.rs:244](../../rust/crates/cuteafd-daemon/src/cli.rs#L244))
  emits the typed launch decisions the launcher consumes: shapes, lanes,
  topology, roles and the memory layout. `cuteafd plan MODEL` without it tells
  you which tensors, formats and kernels a new checkpoint still needs.

## 6. Gates for a new family

These are the gates; the tiers and bars are in
[AGENTS.md "Engineering rules"](../../AGENTS.md#engineering-rules).

1. **CPU gate.** Run `scripts/dev/cpu-gate --task NAME` while iterating, and
   `--full` before the merge into dev.
   - `--full` adds the installed deployment sweep,
     `installed_example_deployments_publish_executable_prefill_shapes`
     ([daemon/tests/budget_cli.rs:220](../../rust/crates/cuteafd-daemon/tests/budget_cli.rs#L220)).
     It plans every `examples/configs/*.config` (42 today) and runs it through
     `release_prefill_shape`.
   - Add your family's min and max configs there. Link them from the family
     doc and from `examples/configs/README.md`;
     `scripts/tests/test_example_configs.py` checks both.
   - Add your family to the lane and readiness source gates (sections 1 and 2).
2. **Golden byte-exactness.** Run your family's golden subcommand through
   `scripts/bench/golden-run.py`. Unit N=1 vs N must be byte-identical (logits
   and KV). Families whose Spark exports use FP32 atomics gate on KL instead
   ([PLAN "Prefill contract", gates](../../PLAN.md#prefill-contract-tj-2026-10-23)).
3. **Exact prefix cache.**
   - The bench's `cache_exact` check ([rust/crates/cuteafd-bench/src/baseline.rs:319](../../rust/crates/cuteafd-bench/src/baseline.rs#L319))
     covers prompt-end and turn-end restores, which must be byte-identical.
   - `scripts/qualify/qualify-prefix-cache.py` covers multi-turn traffic, the
     torture phase and leak accounting.
4. **Lossless speculation.** The `spec_lossless_set` panel
   ([cuteafd-bench/src/panels/lossless.rs](../../rust/crates/cuteafd-bench/src/panels/lossless.rs))
   must pass 10/10: identical ids and row hashes with drafts on and off, with
   no tie tolerance.
5. **One wip-cards pair.** One launch per arm, candidate against dev,
   through `scripts/bench/wip-cards.py` ([docs/wip-cards.md](../wip-cards.md)).
   An exchange or prefill-scheduling change also runs the two landing-mode
   cards: `v41-flash-sim5090` (pinned) and the TP3 5090 shape (dma-buf):
   - panels `baseline` and `spec_lossless_set`;
   - C1/C8/C16;
   - 8K and 64K prefill.

   Speed ratios must be ≥ 0.95, and the port must also match the family's
   absolute prefill reference
   ([PLAN "Prefill contract"](../../PLAN.md#prefill-contract-tj-2026-10-23)).
6. **New kernels.** These need three checks:
   - `scripts/build/compare-sm-exports.py --sms 188,170` (CPU-only; review
     object and cubin differences);
   - the row-curve check across the 2048/2752/4096 band (section 3);
   - for kernel code, landing on `sparkinfer-glmrt` master first, then a pin
     bump.
7. **One SM120 build for PRO 6000 and 5090.** Don't bake in an SM count or L2
   size. Query them at runtime with `NativeLibrary::sm_count` and
   `l2_cache_bytes` ([cuteafd-ffi/src/shared/l2_prefetch.rs:8](../../rust/crates/cuteafd-ffi/src/shared/l2_prefetch.rs#L8))
   and size grids from them, as `Fp8QuantizeGrid` does
   ([daemon/shared/launch_grid.rs:18](../../rust/crates/cuteafd-daemon/src/shared/launch_grid.rs#L18)).
   The SM188/170 export comparison is the CPU-side check that a build doesn't
   depend on one card.
8. **Graphs.** Run the [graph-cache checklist and qualification](#8-graph-cache):
   controller lifetime/cancellation tests, per-layout traces, joined memory
   reconciliation and trace-free adaptive speed evidence.

## 7. Porting a new family: checklist

1. Run `cuteafd plan MODEL` on the checkpoint. Read what it says is missing
   (tensors, formats, kernels) before you write kernel code.
2. Add the descriptor (`loader/plan/families/<family>.rs`) and register it in
   `REGISTRY`. It needs:
   - `detect` and `open` from the checkpoint's own `config.json` and headers;
   - `decode_lanes`: `One { reason }` until a two-lane A/B wins, and the reason
     must cite the issue;
   - `spark_decode` rows;
   - `prefill_limits`: `MeasuredWaves`, `Timing` or `Unmeasured`, with
     `source`;
3. Write the placement adapter: a `PlacementRequest` builder over the shared
   solver (`loader/placement/families/`). Charge every owner with a category
   and group before allocation, and don't run a private pool or onboard policy.
4. Write the coordinator programs and expert packages. Use the AOT exports at
   the exact capacities (1/16/80/256/1024/2048/2752/4096 for EXL3), and the
   checkpoint's native numerics and ingress (FP8/MXFP4 checkpoints use the
   shared FP8 worker; EXL3 is two-tier only, warp-specialized for prefill).
5. Prefill: implement `PrefillHooks` and call `run_prefill_units`. Each unit
   owns its slot slices. Opt the family into `family_prefill_mode` in
   `run-family.sh`.
6. Decode:
   - implement `Backend` for the shared Qwen scheduler, or call
     `LaneSet::run_one(decode_round)` from your scheduler;
   - end every round with `lanes::commit` on a `LaneCommit`;
   - use `Backoff` for all waits;
   - signal readiness after all fallible init;
   - integrate graph owners and prewarm through the [graph-cache checklist](#porting-a-family).
7. Spark experts: use the shared exchange (`SparkLink::new_waves`). Owners come
   from `admit_family`, and route staging uses the shared packed path.
8. Add source gates (only `run_one` and `lanes::commit` in the round; readiness
   ordering), min/max example configs, and family-doc links.
9. Run the gates: `cpu-gate --full`, golden N=1 vs N, exact cache, lossless set,
   one wip-cards pair, and, for kernels, SM188/170 plus row curves.
10. Delete what the shared path replaced. If something cannot fit the shared
    structure, write the reason on #744 rather than forking.

## 8. Graph cache

### Model

One ARC controller per physical device owns capture admission and eviction
([daemon/shared/decode_graph/device.rs](../../rust/crates/cuteafd-daemon/src/shared/decode_graph/device.rs));
typed banks retain executables and their resources
([bank.rs](../../rust/crates/cuteafd-daemon/src/shared/decode_graph/bank.rs)).
Prewarm is best effort, with trace-qualified hot geometries pinned. A miss
captures within budget or runs eagerly; it never refuses a request. Estimates
choose capture and victims, while completed-boundary measurements decide charge.
An underestimate logs overshoot and blocks further growth, not serving.

Admission charges one canonical `graph reserve` per device before KV sizing.
The default total is the trace-derived working-set base plus its 2 MiB tail,
plus shared runtime headroom: 500 MB capped at 100% of that base by default.
`GPU_GRAPH_RESERVE` replaces the total (zero means eager); a scalar applies to
each device independently, and `gpuN` selectors override ranks.
`GPU_GRAPH_RESERVE_EXTRA` and `GPU_GRAPH_RESERVE_EXTRA_CAP` configure headroom;
an explicit extra without an explicit cap is uncapped. Neither may accompany
an explicit total for the same rank. `GPU_RESERVE` cannot erase graph ownership
([loader/placement/graph_reserve.rs](../../rust/crates/cuteafd-loader/src/placement/graph_reserve.rs),
[launch keys](../options.md)). Measured warmup never shrinks admission's reserve.

Driver graph storage is a monotone measured high-water: destruction can release
ARC live capacity without freeing visible physical bytes. Cancelled captures
still require observation, since failed instantiation can grow driver storage.
Qualified single-device chains reuse node/executable pools; fan-out and explicit
peer-spanning captures do not receive chain discounts. Native width and span
matter, not just node count (#1626). Currently reconciliation substitutes this
high-water for the admitted reserve; it is not a current-residency measurement.
Current-level reconciliation, while retaining high-water for admission, is
pending [#1702](https://github.com/tpurtell/cuteafd/pull/1702).

### Single capture path

Every graph-using family is attached: V4.1 through its heterogeneous bank
visitor, GLM/ML4, GLM Flash, V4 and MiMo through their rank banks, Kimi and
Step-5 through their section owners, and Muse/dense Qwen at construction.
MiniMax currently executes eagerly and owns no decode graphs. There is no
unattached insertion, per-bank budget, fixed-startup seal or legacy retirement
path. Empty owners may be constructed before attachment, but cannot capture or
populate a bank until the device controller grants a cancellable ticket.
`begin_capture` requires a live ticket, and production CUDA capture/replay
entry points live in `shared/decode_graph`; the source contract test allows
raw calls only in test/oracle code (`scripts/tests/test_decode_graph_cache_contract.py`).

### Porting a family

Paths below use the prefixes defined at the top of this page; checklist items
name the implementing code or regression-test location.

1. Attach every bank before capture through `CacheAttached` or `attach_device`, even with empty prewarm, with stable site names, cycle, preference, bootstrap bytes and explicit peer span (`daemon/shared/decode_graph.rs`, `decode_graph/native.rs`, `decode_graph/binding.rs`, `daemon/families/kimi_k3/engine.rs`).
2. Use the same reserve and basis on each device: calibrated node rates only on their measured family/architecture/route, otherwise device-wide bytes with a warning; unknown node counts stay eager (`daemon/shared/decode_graph/device.rs`).
3. Retain a strong controller per device across temporary startup banks until persistent banks attach; the registry is Weak and seeds the observer only on creation (`daemon/families/kimi_k3/startup.rs`, `daemon/shared/decode_graph/device.rs`).
4. Key captures by exact typed geometry and retained resource owners; take a cancellable ticket before capture, report native nodes/width/span before insertion, and commit only through that ticket (`daemon/shared/decode_graph/bank.rs`, `decode_graph/native.rs`).
5. Set `cache_cycle` at decode/verify/prefill/wave entry, attribute startup to Other, and first-touch eagerly before measuring capture growth (`daemon/families/glm5/engine.rs`, `daemon/shared/decode_graph/bank.rs`).
6. Queue all ranks before any sync; call `observe_completed` once at the outer completed-step boundary, measure each device once, and drain all consumers before dropping executables/resources, including errors (`daemon/shared/decode_graph/device.rs`, `decode_graph/bank.rs`).
7. Remove sealing, declared-key refusals, per-bank lazy budgets, census replacement of admission and family graph margins; charge exactly one `graph reserve` per device (`loader/placement/graph_reserve.rs`, `daemon/shared/memory_report/reconcile.rs` tests).
8. Version prewarm/reserves by resolved capture layout, including RTX count, peer rank and Spark/onboarding state; never borrow a different layout's trace, and bootstrap missing layouts unpinned with a warning (`loader/serving_capacity/qwen_graphs.rs`, `glmf_graphs.rs`).
9. Extend the trace generator for the family's real site/key schema; select whole reachable geometries with at least one replay, charge zero-hit siblings within selected geometries, and leave wholly zero-replay geometries lazy (`scripts/dev/graph-prewarm-from-trace.py`, `scripts/tests/test_graph_prewarm_from_trace.py`).
10. Prewarm the sustained decode/verify working set, not just startup rows; pin sustained rolling heat/churn only, sort pinned-first then lifetime replay hits descending, and test exact base/tail regeneration and no untraced pins (`scripts/tests/test_graph_prewarm_from_trace.py`, `loader/serving_capacity/kimi_prewarm.rs`).
11. Delete or fold single-node graphs into adjacent sections between real host/exchange joins; inspect every <=2-node key and test that no new key commits fewer than two native nodes (`daemon/families/kimi_k3/graphs.rs`, `capture_worthwhile`).
12. Preserve existing pinned full-chunk prefill coverage, with tails eager; adding prefill graphing is separate work, not permission to capture peer joins incoherently (`daemon/families/deepseek_v41/graph_policy.rs`, `daemon/families/deepseek_v4/engine/prefill.rs`, #1638).
13. Publish final post-pool-adoption provenance through `fault::set_inputs`; Once-guard `first_request` after request retirement and the joined `observe_completed`, never inside a live lane (`daemon/shared/memory_report/reconcile.rs`, `daemon/families/deepseek_v41/v41_native_serve/scheduler/independent.rs`).
14. Keep graph/mapped-table/reconciliation diagnostics on the central request-time stats hook, never per-round `graph_capture_stats`; test provenance, canonical owner readback and diagnostic keys (`daemon/shared/probe.rs`, `daemon/shared/memory_report/reconcile.rs`).
15. Gate controller/bank ownership, cancellation, basis/cycle/span, startup lifetime and signed/null reconciliation; run workspace check, default CPU gate, native build if changed, golden/cache/speculation checks and the family's card (`daemon/shared/decode_graph/{device,bank,native}.rs`, `scripts/tests/test_graph_prewarm_from_trace.py`).

### Measuring

- Trace working sets, churn and memory only: use `GRAPH_TRACE` with an absolute
  host path and existing parent; preserve per-device schema-2 snapshots beside
  results. Trace each supported capture layout with the adaptive standard card,
  C1/2/4/8/16 and a 32K point. Keep KV target 2M / max 100M to retain onboarding.
- Generate proposals with `scripts/dev/graph-prewarm-from-trace.py --family FAMILY
  --trace /absolute/trace.device-0.json --output /absolute/proposal.json`; review
  whole-geometry cost/heat, update the checked-in list and relaunch. Hits are
  cycle/boundary-attributed, not launch-exact; forgotten ARC ghosts lose history.
  Per-key capture shares/model costs are attribution, not additional device charges.
- Speed requires both `GRAPH_TRACE` and `SPECULATION_TRACE` unset, even after
  trace I/O offload (#1696). Use same-day, matched-pool, balanced-order warmed
  pairs; fixed draft length is a step-parity control, not default-selection evidence.
- Judge graphs on/off on the adaptive standard mixed-content card and C1-C16
  sweep, with per-arm verify-width distributions; wins or ties stay on by default.
  Report TTFT and emitted tok/s, plus per-round wall minus full-step time to expose
  host costs. Verify exact 8,192-token prefill with the serving tokenizer and report
  prompt tokens/unit geometry (#1672, #1703); different shapes are not comparable.
- Read thermal/power-cap seconds, busy clocks, temperature and T.Limit headroom;
  qualify no refusals/OOMs, eager/eviction/recapture counters, arena versus reserve
  and explicit headroom, and report KV/onboarding changes owner by owner, not an
  inferred payoff from a smaller reserve (see [engineering rules](../../AGENTS.md#engineering-rules)).

Reconciliation ([daemon/shared/memory_report/reconcile.rs](../../rust/crates/cuteafd-daemon/src/shared/memory_report/reconcile.rs))
compares observer-excluded CUDA use with admitted Growth owners, excluding
release padding. `delta_vs_measured = delta_vs_admitted + graph_reserve_slack`;
slack is reserve minus the observed current graph-arena level, sampled again
from the settled device sample at Ready and joined first_request rather than
reused from the last capture boundary, which can catch a transient
instantiate/upload peak that has already reversed. The refreshed level is
bounded by the admitted high-water, so a non-graph untracked allocation live at
the sample can only reduce the figure below the charge, never inflate it; the
path releases peak credit and never raises the charge. Negative slack means the
arena exceeded its reserve. Component values are `graph_reserve_bytes`,
`graph_arena_bytes` (the level used) and `graph_arena_high_water_bytes` (the
monotone charge admission and eviction keep, which still covers a transient
capture peak); a family whose controller never samples reports the high-water as
both. At Ready and joined first_request,
require nonnegative slack and RTX `abs(delta_vs_measured) <= padding_bytes / 2`.
Unused admitted Growth capacity can make Ready negative; explain it, do not hide
it. Exclude the measured observer context once; missing provenance, physical
samples or nonzero-owner arena telemetry is unavailable/null, never a pass.
Spark Ready uses launch-relative allocatable consumption and the reconciled UMA
owner ledger without alias double charges or future endpoint rings. Its deltas
remain diagnostic, not drift-flagged (#1568); coordinator post-request Spark
physical telemetry is unavailable (#1631). The live native worker has no graph
sites today: `SPARK_GRAPH_RESERVE` forwarding is not active ownership; add worker
admission and controller attachment together when enabling capture
(`daemon/shared/experts/service/backend.rs`).

The observer's untracked residual subtracts the ledger's reported granularity
tail (the SM120 2 MiB backing beyond every live tracked request) as well as the
tracked total. `used - tracked` otherwise contains the page padding of every
tracked owner, so a cache that builds many small device buffers would be read as
graph growth and fund the graph reserve. Padding belongs to the owner that
allocated it; only bytes no ledger row claims and no rounding explains count as
graph arena. Measured owners with no allocation request behind them are
subtracted too: a mapped pinned host pool reserves device page-table bytes in
every context it is mapped into (SM120: one byte per 512 bytes mapped, so a
32 GiB host tier costs 64 MiB per GPU), which admission charges as driver page
tables. The pool measures the real cost once, before its first chunk and after
its last, and publishes it as `pinned host mapping: N chunks` in `/v1/stats`
(`unclaimed`/`unclaimed_bytes`). That source set is closed: only a vetted,
measured driver-side cost may publish an unclaimed row. The attach baseline is
itself an untracked residual, so an owner that predates the controller cancels
in it and never lowers the level.

### Pitfalls

- Warmup must first-touch eagerly and upload synthetic tables before launch; all-masked rows must still satisfy MoE guards, and empty prewarm must still attach the lazy controller (#1663, #1645).
- Family-named or duplicate graph owners break reconciliation even when bytes were charged: use exactly `graph reserve` (#1663, #1645).
- Single-node executable overhead can dominate an arena: fold or delete it, never treat it as a graph win (#1673); legacy Qwen/V4.1 sites remain follow-up work (#1658, #1644).
- Zero-replay prewarm wastes startup and storage; select whole replayed geometries, account siblings, and order by heat so reduced reserves keep the hot prefix (#1663, #1666).
- Untraced TP2/onboarding layouts cannot inherit pins or another layout's reserve/list (#1658, #1645).
- A lane-local first-request sample races its peer; missing provenance or noncanonical owners also fabricates apparent accounting coverage (#1644, #1663).
- Weak registry expiry during startup resamples against already captured storage: retain a controller until persistent lane attachment (#1644, #1663).
- Per-round `graph_capture_stats` scans/serializes model-sized inventories: leave it to request-time diagnostics (#1697).
- Trace-enabled speed includes observer overhead even with bounded background writers; use traces only for inventory and trace-free adaptive cards for decisions (#1694, #1696, #1660).
- Boundary observation still samples CUDA, walks the allocation ledger and visits banks: do not multiply per-step observer calls; optimization is pending #1706.
- A graphs-on speed loss with equal GPU step time is host bookkeeping (stats snapshot, observer), not replay: measure per-round wall minus step before blaming graphs (#1697, #1706).
- High-water can retain transient capture peaks, and untracked driver memory (e.g. pinned host mapping page tables) can land in the graph residual: do not label it current resident graph memory (pending #1702).
