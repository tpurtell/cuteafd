# `/v1/stats` shared serving counters

Every serving family installs `ApiPolicy::serve` (`shared/api.rs`), which runs
`shared::probe::graph_capture_stats` on `/v1/stats` fetches. Cards and dashboards
read one schema across all 15 serving families (`deepseek_v4`, `deepseek_v41`,
`glm5`, `glm5_flash`, `kimi_k3`, `qwen4`, `qwen3_5`, `mimo_v2`, `minimax_m3`,
`step5`, `hy4_flash`, `inkling2`, `inkling2_mini`, `muse_spark`,
`mistral_large4`).

All cumulative counters are monotonic since process startup; clients subtract
across their own warmed interval.

## Shared schema

### Graph counters

| Field | Unit | Meaning | Families |
|---|---|---|---|
| `graph_cache` | array (per device) | ARC graph-cache controller snapshots (`charged`, soft/hard caps, hits, evictions, update sites) | All 15 |
| `graph_captures` | count | Cumulative CUDA graph captures recorded in `cuteafd-ffi` | All 15 |
| `graph_capture_sites` | `[site, count][]` | Cumulative graph captures per capture site | All 15 |
| `graph_capture_site_deltas` | `[site, delta][]` | Per-site capture delta since the previous `/v1/stats` fetch | All 15 |
| `graph_missing_variants` | `{family, shape, runs}[]` | Shapes that ran eagerly because no sealed graph covered them | All 15 |
| `graph_instantiate_ns` | ns | Cumulative host time inside `cudaGraphInstantiate` + `cudaGraphUpload` in the shared `GraphBank` (failed instantiations included) | All 15 |

### Serve-loop phase counters (`phases`)

The three time counters `prefill_round`, `decode_call`, and `idle_wait` are
mutually exclusive; `decode_call_with_prefill_waiting` is a subset of
`decode_call`.

| Field | Unit | Meaning | Families |
|---|---|---|---|
| `phases.phase_ns_total.prefill_round` | ns | Cumulative charged wall time of `PrefillQueue` rounds (excludes non-prefill GPU-turn waits; includes Spark exchange) | All 15 |
| `phases.phase_ns_total.decode_call` | ns | Cumulative wall time of decode calls (see batching note below) | All 15 |
| `phases.phase_ns_total.decode_call_with_prefill_waiting` | ns | Subset of `decode_call` that ran while prompts waited in `PrefillQueue` (time paid to `--decode-share`) | All 15 |
| `phases.phase_ns_total.idle_wait` | ns | Cumulative wall time blocked waiting for the next request with nothing admitted, queued, or encoding | All 15 |
| `phases.rounds_total.prefill` | rounds | Cumulative `PrefillQueue` rounds executed | All 15 |
| `phases.rounds_total.decode_calls` | calls | Cumulative decode calls executed | All 15 |
| `phases.decode_call_members_total` | member-calls | Sum of active decoding requests participating in each decode call (one per request per call, not draft verify rows) | All 15 |
| `phases.prefill_prompt_rounds_total` | prompt-rounds | Sum of prompts that ran at least one chunk in each prefill round (a prompt spanning N rounds counts N times) | All 15 |
| `phases.prefill_queue_wait_ns_total` | ns | Cumulative wait from a prompt entering `PrefillQueue` to starting its first prefill chunk (charged once per prompt) | All 15 |
| `phases.active_decode` | requests | Live decoding requests at the last scheduler stats publish (mirrors top-level `active`; drops to `0` before entering `idle_wait`) | All 15 |

**Decode-call batching difference:**
- **Single-lane families** (`deepseek_v4`, `glm5`, `mistral_large4`, `glm5_flash`, `mimo_v2`, `kimi_k3`, `qwen4`, `qwen3_5`, `minimax_m3`, `step5`, `hy4_flash`, `inkling2`, `inkling2_mini`, `muse_spark`): one `decode_call` is one `lanes.run_one(|lease| decode_round(..))` invocation (a single decode/verify step across `active.len()` active sequences).
- **`deepseek_v41`**: one `decode_call` is one `P::decode_round` invocation, which on two-lane and distributed paths executes a stretch of lane rounds until a wake, admission, or prefill-due boundary across `members.iter().map(Vec::len).sum()` active requests.

All 15 families share `PrefillQueue`, `LaneSet` decode calls, blocking idle receive, and live `active` publication, so every family records every phase counter above (no phase is omitted or zero-faked).

### Spark exchange counters (`exchange`)

Recorded by `shared::spark_waves::run_worker` on every landed wave (one layer's coordinator-to-Spark round trip).

| Field | Unit | Meaning | Families |
|---|---|---|---|
| `exchange.wave_ns_total.staging` | ns | Cumulative worker host time from wave enqueue until staged copies complete | All 15* |
| `exchange.wave_ns_total.post` | ns | Cumulative worker host time building and posting the RDMA request | All 15* |
| `exchange.wave_ns_total.wait` | ns | Cumulative worker host time from post until the last Spark rank's reply lands | All 15* |
| `exchange.waves_total` | waves | Cumulative landed Spark exchange waves | All 15* |
| `exchange.wave_rows_total` | rows | Cumulative rows carried by landed waves (counted once per wave, not per rank) | All 15* |
| `exchange.reply_plane_bytes_total` | bytes | Cumulative single-rank reply-plane bytes across landed waves | All 15* |
| `exchange.expert_prefill_units.busy_ns_total` | `[ns; MAX_SPARK_WORLD]` | Per-rank sum of Spark prefill-unit busy time reported in reply headers (decode units report no pace; merged launches overlap so sum can exceed wall time) | All 15* |
| `exchange.expert_prefill_units.idle_ns_total` | `[ns; MAX_SPARK_WORLD]` | Per-rank sum of inter-unit idle time on the Spark prefill owner stream | All 15* |
| `exchange.expert_prefill_units.units_total` | `[units; MAX_SPARK_WORLD]` | Per-rank count of prefill units whose reply carried a pace | All 15* |

\* Every family emits the `exchange` group so readers see one schema. Non-Spark families (`qwen3_5`, `inkling2_mini`) and 0-Spark deployments of Spark-capable families never start `run_worker`, so their `exchange` counters remain `0`.

## Hot-path cost by call site

Every recording point uses relaxed atomic `fetch_add` only — no allocation, no lock, and no log line:

- `shared/decode_graph/bank.rs` (`CaptureTicket::complete_source`): 1 relaxed add on cold capture instantiation only; zero cost on steady-state graph replay.
- `shared/spark_waves.rs` (`run_worker`): 6 relaxed adds per landed wave plus 3 per reporting Spark rank; reuses the worker's existing `WaveTimes` clocks (zero new clock reads).
- `shared/prefill_share.rs` (`PrefillQueue::round_*`): 3 relaxed adds per prefill round; reuses `started` / `started.elapsed()` already taken for `last_round` (zero new clock reads).
- Single-lane serve loops (`shared/qwen/scheduler.rs`, `families/glm5/serve.rs`, `families/glm5_flash/serve.rs`, `families/mimo_v2/serve.rs`, `families/deepseek_v4/serve.rs`):
  - `record_decode_call`: 3–4 relaxed adds per decode call; reuses the existing `cycle = Instant::now()` and `cycle.elapsed()` passed to `prefills.stepped(...)` (zero new clock reads on the hot path).
  - `record_first_chunk_wait`: 1 relaxed add once per prompt (`chunks == 0`); reuses `p.started` and the chunk's existing `timer = Instant::now()` (zero new clock reads).
  - `time_idle_wait`: runs only when blocked idle (`!busy && media.is_empty()`), never on the active path.
- `families/deepseek_v41/v41_native_serve/scheduler.rs` + `queue.rs`:
  - `record_decode_call`: 3–4 relaxed adds per `P::decode_round` call; reuses `cycle` when prefills are waiting (takes one `Instant::now()` per multi-round `P::decode_round` call when no prefill is waiting).
  - `record_prefill_queue_wait`: 1 relaxed add once per admitted prompt at its first chunk.
  - `time_idle_wait`: runs only when `busy_slots == 0` and blocked in `receive.blocking_recv()`.
