# Reply incast reference

Reference audit and measured experiments, 2026-10-25; source base
`974d235de` on `work/p0`. The reference audit uses existing receipts;
new hardware measurements are explicitly labeled under Experiments.
No kernel or network settings are changed by these experiments.
Related evidence: #693, #714, #715, #718, #729 and the corrected probe history
in #497. Rates use decimal GB/s and Gb/s; frame capacities use bytes.

## Verdict

Reply incast is plausible and visible in directional error counters, but
neither Spark count alone nor the startup landing probe proves the bottleneck.
All traffic-bearing Spark ports in the selected receipts report **100 Gb/s**;
raptor reports **400 Gb/s**, not six active 200G senders. Six synchronized
100G writers can nevertheless offer 600G to a 400G receiver. Actual switch
queue size, PFC/ECN settings and pause behavior are not established here;
"no PFC" below is the experiment assumption, not a discovered configuration.

The strongest same-build GLM comparison has nearly clean four-rank replies
and substantially more six-rank errors at 64K, but also changes onboarding
and placement. V4.1 supplies an important counterexample: six ranks can be
clean when the large encoder is local and remote replay waves are small.
MiMo has no lower-rank control in this selected set. Do not turn these
observations into a universal four-versus-six threshold or a causal fit.

## Bytes and the general hardware model

Expert TP returns a full-hidden BF16 partial from every participating rank;
the output width does not shrink with TP degree. For a full row unit with
hidden width H, routed rows R, top-k K and N ranks:

```
request/rank = 96 + R * (H + H/32 + 40 + 12*K)  # FP8 K32 values/scales + metadata
reply payload/rank = R * H * 2                 # BF16 partials
aggregate reply = N * reply payload/rank
```

These are full-row-unit estimates, not reconstructed actual routed-row
counts for every layer. Empty ranks, tails, local experts and encoder/replay
splits change actual traffic. A contiguous reply adds a 96-byte header;
admitted reply-frame capacity also reserves row indices, which need not be
transmitted on the contiguous path. Wire headers/ACKs are additional.

| Shape | H / K / full R | Request/rank | Reply payload/rank | Admitted reply capacity/rank |
| --- | --- | ---: | ---: | ---: |
| G: GLM K4, EXLR8, MiMo Pro | 6144 / 8 / 4096 | 26,509,408 | 50,331,648 | 50,348,128 |
| V2: V4.1 unit capacity | 5120 / 6 / 2048 | 11,042,912 | 20,971,520 | 20,979,808 |
| V4: V4.1 enlarged unit capacity | 5120 / 6 / 4096 | 22,085,728 | 41,943,040 | 41,959,520 |
| Vr: V4.1 max replay upper row unit | 5120 / 6 / 128 | 690,272 | 1,310,720 | 1,311,328 |

Frame constants are in `rust/crates/cuteafd-transport/src/protocol_v2.rs:23`;
V4.1 top-six FP8-K32 geometry is also asserted by
`rust/crates/cuteafd-daemon/src/families/deepseek_v41/v41_experts/tp2/reference_tests.rs:23`.
Thus G replies are 1.899x request bytes, not exactly 2x after request
metadata; both scale linearly in N at fixed R/H. EXLR8 A8 versus A16 names
expert activation arithmetic, **not FP8 wire replies**.

Let L_i be sender link rate, P_i reply production rate averaged over its
compute cycle, C coordinator ingress link rate, and G effective direct-GPU
landing service rate. Distinguish two regimes:

```
peak synchronized burst ratio = sum(L_i) / min(C, G)
compute-averaged offered rate = sum(min(L_i, P_i))
sustained service ratio = compute-averaged offered rate / min(C, G)
```

Ratios above one indicate queue growth if maintained; ratios below one do
not preclude a short synchronized burst. Transport duty cycle, staggering,
backpressure and buffer size determine overflow. For pinned intake, use host
landing capacity for NIC ingress and model the subsequent H2D queue
separately: the harmonic host+H2D rate is **not** the host ingress ceiling.
These are single-active-rail tables; dual rails require summing actual
independent paths and checking their shared NIC/PCIe sink.

| N | Offered peak at 100G/Spark | At 200G/Spark | NIC-only ratio: C200/L100 | C400/L100 | C200/L200 | C400/L200 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 200 | 400 | 1.0 | 0.5 | 2.0 | 1.0 |
| 3 | 300 | 600 | 1.5 | 0.75 | 3.0 | 1.5 |
| 4 | 400 | 800 | 2.0 | 1.0 | 4.0 | 2.0 |
| 6 | 600 | 1200 | 3.0 | 1.5 | 6.0 | 3.0 |
| 8 | 800 | 1600 | 4.0 | 2.0 | 8.0 | 4.0 |

A **hypothetical** effective GPU sink equal to the 19.6 GB/s local probe
is 156.8G, below either coordinator NIC. Its burst ratios for N=2/3/4/6/8
are 1.28/1.91/2.55/3.83/5.10 at 100G, or twice those at 200G.
This is a sensitivity calculation, not measured remote-incast bandwidth.
At pinned host probe 52.4 GB/s (419.2G), the C200/C400 port would constrain
initial ingress before that probe rate; host buffering can decouple ingress
from the slower full store-and-forward completion.

## Probe validity and automatic selection

At this document's base, `rust/crates/cuteafd-daemon/src/shared/spark_intake.rs:125`
requests a 64 MiB, eight-sample probe. Native implementation is
`native/shared/src/cuteafd_native.cc:3939`: a local RC QP pair on the same
NIC, pinned-host sender, one outstanding SEND, 96-byte host header plus
host/GPU payload receive SGEs, waiting for both completions. Each path gets
one untimed warm round and eight timed rounds; the median is used, then GPU
payload bytes are checked. Host and GPU paths are tested sequentially.

This tests registration, byte correctness and a local transaction rate.
It is not a switched four/six-writer incast test, does not establish SEND
versus WRITE equivalence under load, and does not certify an uncontended
NIC/GPU. The host rate above 400G itself warns against treating this local
loopback rate as a switched-port throughput measurement.

The pinned estimate is `1/(1/host + 1/H2D)`
(`spark_intake.rs:93`). `auto` accepts GPU at >=0.5 of that estimate
(`spark_intake.rs:88`), a policy based on historical end-to-end overlap wins,
not "select the fastest isolated GB/s." It intentionally permits a lower
standalone GPU rate to avoid host staging, copies and resource contention.
`mode_for` at line 114 simply divides bytes by the probe; startup "302 MB
waves land in 15.4 ms" is an **estimate**, not observed serving latency.

An executor's `IntakePreference::Pinned` changes only `auto` to pinned
(`spark_intake.rs:146`); explicit settings still win. Startup may log a
generic GPU transport and then the prefill executor's pinned transport.
Use the actual executor intake, not the first GPU line. The inventory below
records the final logged prefill choice; V4.1 can log multiple instances.

#497 originally attributed 3.6 GB/s to idle Gen1. Its later controlled
comment disproves Gen1 as a sufficient explanation: GPU0 entered an
unmodified cold probe at Gen1 x16 and measured 19.9 GB/s; warming did not
materially improve it. Historical contention was not captured, so the old
3.6 GB/s cause remains unresolved. No warm-up or ordering fix follows from
that title. Diagnostics/logging changed; selection thresholds did not.

## What the recorded comparisons establish

Measured 8K / 64K values, worker physical TX / engine logical TX amplification
(A), worker retransmit events (T), coordinator OOS / worker packet-sequence
events (reply direction):

| Receipt comparison | A (8K / 64K) | T (8K / 64K) | Interpretation |
| --- | --- | --- | --- |
| GLM 09db 2 RTX + 4 Sparks, pinned | 1.0577 / 1.0190 | 0 / 0 | OOS 4/28, seq 4/28; 2929/2817 tok/s |
| GLM 09db 2 RTX + 6, zero onboarding, pinned | 1.0127 / 1.0991 | 19 / 1318 | OOS 777/34224, seq 749/34264; 1982/2001 tok/s |
| GLM 09db 1 RTX + 4, pinned | 1.0438 / 1.0108 | 2 / 1 | OOS 0/38, seq 0/38; 2926/2750 tok/s |
| GLM combined base -> candidate, 6 | 1.0920/1.1988 -> 1.1195/1.1502 | 122/1835 -> 55/1294 | Several transport/overlap/intake changes, not rank isolation |
| MiMo combined base -> candidate, 6 | 1.2834/1.1801 -> 1.1393/1.1527 | 1638/7197 -> 806/4417 | Reply pressure persists; no selected smaller-N control |
| EXLR8 combined base -> candidate, 6 | 1.1236/1.1785 -> 1.2022/1.1827 | 90/796 -> 1028/9321 | Pinned -> GPU plus combined changes; more events without proportionate A increase |
| EXLR8 up1 A16 min4 -> max6 | invalid/1.0159 -> 1.1570/1.2273 | 0/0 -> 219/2853 | Supports rank/burst hypothesis, but layout/compute differ |
| V4.1 combined max6, both arms | see inventory | 0 / 0 | Reply OOS/seq also zero; large encoder local, replay <=128 rows |

The GLM four-rank topology control has six onboard layers; sixzero has none.
Changed kernel slices, placement and KV budgets forbid a rank-only causal
claim. V4.1 min3/min4 has some request-direction errors and cannot support
an invariant "smaller N is clean" rule. Cards are receipts, not a regression
fit across unequal configurations.

### Matched GLM profile

`transport-combined-glmtrace/` holds clean `09db857a8` versus combined
`0f888ffd9`: same prompts, 2 RTX + 6 Sparks, five onboard layers,
2,097,152-token pool, one warm request per length. At measured 64K:

- Worker amplification 1.01762 -> 1.14432; worker retransmits 23 -> 1727.
  Coordinator OOS 0 -> 48353; worker packet-sequence errors 0 -> 47015.
  Baseline request errors instead occur on workers/coordinator respectively.
- Aggregate whole-prefill worker physical/logical rates are approximately
  71.1/69.8 -> 97.1/84.8 Gb/s. These averages are below both port and probe
  proxy, but cannot rule out short 600G incast bursts. Actual receive-window
  start/end times are unavailable: a serving-wave offered or landed rate
  cannot honestly be recovered from these averages.
- GPU0 H2D 338.2307 GB / 6.29314 s gives 53.75 GB/s during copying, **not**
  end-to-end pinned landing. Candidate almost eliminates those copies.
  Simultaneous no-traced-CUDA time remains 19.474 -> 19.068 s; GPUDirect NIC
  writes are not CUPTI memcpy operations. That interval is not measured
  direct landing time or proof that all of it is network wait.
- Profiled 64K client rate 1720 -> 2073 tok/s is informative, not a final
  performance claim: baseline emits client timing records, candidate does
  not. Candidate 8K fabric is absent, not zero. Baseline 8K A=0.9706 is
  sampling-skew-invalid. No paired fine-grained post/receive/land split.

Physical NIC counters include headers, ACKs, retries and unrelated traffic;
logical counters and approximately one-second fabric snapshots can have
boundary skew. Values below one and V4.1 inconsistent counter windows must
not be interpreted as negative overhead or genuine 2x replays. Raw
`roce_adp_retrans`, OOS and packet_seq are event deltas, not packet-loss
percentages. Coordinator reply RX/logical RX is deliberately excluded:
WRITE-tail accounting double-counts bytes (#714/#718 audit).

## GLM after the compact-WS kernel fix: conditional ceiling

For H=6144, R=4096, N=6, one full BF16 reply wave is 301.990 MB.
The proposed 13 ms/layer Spark compute time is an input assumption supplied
for the new S1 compact-WS kernel, **not measured by the old receipts**.

| Component | Estimated full-wave time | Qualification |
| --- | ---: | --- |
| 400G coordinator serialization | 6.04 ms | Ideal payload-only, no retries |
| 200G coordinator serialization | 12.08 ms | Hypothetical NIC variant |
| Six 100G sender burst duration | 4.03 ms/rank | Up to 600G aggregate offered during overlap |
| GPU using 19.6 GB/s probe proxy | 15.41 ms | Must validate with remote serving A/B |
| Pinned using 27.4 GB/s harmonic proxy | 11.02 ms | Store-and-forward estimate, not NIC ingress |
| Spark compute assumption | 13.00 ms | Per full row unit/layer |

Compute-averaged production is 30.97 Gb/s per Spark, 185.84G at six ranks
or 123.89G at four. Each sender is below 100G on average. At the GPU proxy,
sustained ratio is 1.185 for six ranks versus 0.790 for four; at 400G NIC
alone it is only 0.465 versus 0.310. Thus direct landing **could** become
the six-rank critical path after the kernel fix if this proxy holds; the
400G port is only a peak-burst constraint in that calculation. At four
ranks the analogous GPU estimate is 10.27 ms. No observed serving-wave rate
yet distinguishes actual landing service, buffering, synchronization and
Spark compute as the binding limit. The new frozen-batch experiment below
establishes that GLM's multiwave prefill actually retains **pinned** intake;
the direct-GPU branch of this hypothetical model does not describe that
measurement. Its logged staged-path proxy is 27.4 GB/s, not 19.6 GB/s.

## Remedy bounds, not promised speedups

- **Chunk size:** no ideal payload-byte saving. A 48 MiB reply needs about
  192/48/12 payload WRs at 256 KiB/1 MiB/4 MiB (plus framing/notifications).
  Larger chunks reduce posting/completion work, but can increase contiguous
  bursts and replay work; splitting alone does not cap aggregate send rate.
  Request and reply chunking must be independently controlled to isolate this.
- **Pacing:** can remove avoidable queue overflow/replays, but does not raise
  the physical sink rate and can hurt when overconservative. At six ranks,
  fair shares are C400/N=66.7G, C200/N=33.3G, or Gproxy/N=26.1G. A fair-rate
  cap is not interchangeable with staggered starts. For matched A=1.14432,
  eliminating *all* excess wire bytes has at most 2.22 ms GPU-proxy service
  equivalent per G wave (about 1.99 ms after a nominal 1.5% header allowance).
  The #714-only A=1.244 gives 3.76 ms. These are byte-service bounds, **not**
  measured replay latency bounds: timeout stalls can change latency without
  proportional extra bytes, and not all excess bytes are retransmissions.
- **FP8 replies:** ideal half-byte payload transport saves at most 7.70 ms
  against the 15.41 ms GPU proxy, before scales, quantize/dequantize and
  reduction costs. With complete compute/transport overlap, cycle time is
  max(13,15.41)=15.41 -> max(13,7.70)=13 ms: only 2.41 ms, or 1.185x
  throughput ceiling for this simplified stage. Unoverlapped wave savings
  can be larger. FP8 halves sustained production bytes and burst duration,
  not instantaneous line-rate incast. It changes numerics and needs an
  explicit quality gate; A8 expert kernels are not evidence for it.
- **PFC/ECN:** lossless flow control or congestion control may replace drops
  with backpressure; neither adds sink capacity. Shared queues can introduce
  head-of-line blocking. Do not infer their status from OOS alone or change
  switch policy as part of a launcher-only experiment.

## Bounded experiment plan (status updated by Experiments)

Existing setting: `SPARK_INTAKE=gpu|pinned|auto|host` in the shared launcher
(`scripts/launch/run-family.sh:613`), mapped to `CUTEAFD_SPARK_INTAKE` at
line 633. Use explicit `gpu` versus `pinned`, and check the prefill executor's
final startup line. `host` also lands in host memory but uses the original
staging/compute-stream upload path; it is not an alias for pinned intake's
large-wave receive-slot retention and copy-stream uploads.

At inspected base `974d235de`, no runtime reply-chunk cutoff setting exists;
the prior combined seal has a hard-coded 256 KiB request/reply WRITE chunk
constant (`request_write.rs:17` in that frozen source), while this base lacks
that module. The inspected `work/shared-decode-exchange` seal `0f888ffd9`
also does not supply independent launcher cutoff controls. **Do not invent
a config key or call off/1 MiB/4 MiB currently runnable.** Pacing needs implementation. FP8 row-scaled response protocol support already
exists (`protocol_v2/request.rs:237` and protocol tests), but no daemon
call site selecting it was found at this base: end-to-end serving integration,
launcher controls and correctness/quality gates are still needed.
Any subsequently landed controls must be checked against their own revision.

The measured explicit-GPU attempt below exposes a feasibility limitation:
this base's multiwave path requires pinned intake (`spark_intake.rs:428`).
The setting exists, but a GPU/pinned pair is not runnable for that executor
without implementing per-wave GPU landing ranges or changing execution.
The latter would confound a landing-only comparison. The matrix below is
therefore conditional, not a claim that every arm currently launches.

1. **Conditional: landing x ranks.** GLM K4 max, then EXLR8 max, fixed
   source/checkpoint/prompt hashes, 4096-row units and identical onboarding
   and KV policy wherever both layouts fit. GPU/pinned x 4/6 Sparks is four
   launches/model. If a fixed placement cannot fit both, mark the rank
   comparison confounded, but keep each same-rank landing pair valid. One
   untimed 8K and 64K warm batch then one measured pair per launch. Existing
   runner only; budget <=5 min/config excluding builds (20 min/model,
   40 min total), explicit load/run timeout; readiness can exceed budget
   when streaming, so report load separately rather than hiding it.
   Ceiling: conditional 15.41 vs 11.02 ms six-rank service estimates, not a
   guaranteed pinned end-to-end win. A started bounded smoke finishes and
   tears down naturally; mark failed arms invalid and schedule no follow-up
   on correctness/memory failure. Do not pursue a clearly >=5% slower arm
   without a new hypothesis.
   One launch/arm is the screen, repeats only for genuinely borderline wins.
2. **Once independent chunk controls exist:** same two landing modes x
   reply cutoff off/256 KiB/1 MiB/4 MiB x 4/6 Sparks = 16 configurations
   per model (32 total; <=160 min at five min each, excluding builds/load
   overruns). Reuse stage-1 off controls only if source and configuration
   remain identical; otherwise rerun controls. Keep request transport and
   request chunk size fixed. First screen GLM, then EXLR8; stop a cutoff
   branch at a clear <0.95 throughput ratio or correctness failure. Retain
   only lower reply errors/amplification with non-regressing warmed TTFT
   and C1/C16 decode, or a clear end-to-end win. Ceiling: posting overhead
   and replay relief only, not fewer ideal payload bytes; do not chase a
   smaller raw event counter when end-to-end performance does not improve.
3. **Pacing only after that screen:** one winning unpaced control and one
   paced arm at each affected rank/landing path; start from measured sink
   capacity, not unvalidated 19.6 GB/s. Record offered-rate cap/stagger
   semantics. Ceiling: excess-byte service estimates above, with unknown
   timeout benefit. Budget <=10 min per A/B path after implementation;
   stop at correctness failure or clear >5% slowdown. No broad rate sweep
   unless the first pair is borderline and supplies a new hypothesis.
4. **FP8 replies last:** first an explicit native/quality check, then one
   BF16/FP8 A/B on the worst still-limited case (<=10 min after build/load).
   Ceiling: 7.70 ms transport-only or 2.41 ms overlapped-cycle saving under
   the stated assumptions. Stop on failed quality or no delivered win after
   conversion/reduction costs. Only expand to the other model if it wins.
   Optional V4.1 min adds a negative/control case after GLM/EXLR8, not a
   prerequisite; max-local encoder cannot stress large remote reply waves.

Every hardware arm should save negotiated traffic-bearing port rates,
actual executor mode and all four probe rates, per-rank logical/physical
bytes, directional retrans/OOS/sequence deltas over the exact same request
window, prompt hashes, row units, local expert placement, pool budget,
ready time, warmed TTFT/prefill rate, C1/C16 decode and memory headroom.
Prefer true receive/post/completion timings if existing instrumentation can
supply them; mark missing data instead of using CUDA idle as a surrogate.
Hold hardware locks only around the run and tear down before releasing.
No new driver or hardware queue is created by this reference.

## Experiments

### Measured: compact-WS kernel batch, frozen 974d235de

One launch per arm, baseline `12d743f7136e92f98a1898f41d10a9a6bd275f65`
to candidate `974d235def25603ce42de3140c845d28011f5d86` (SparkInfer
`080bd9255a01062aa0cc0653810a50aeffad7095`). GLM K4, 2 RTX + 6 Sparks,
DFlash2, context attention, 2,097,152-token pool and identical five TP2
onboarded expert layers (3..8). Both arms have transport timing enabled;
one untimed warm request per length precedes each timed request. These
are paired profiled measurements, not an unprofiled release card.

| Measured metric | Baseline | Candidate |
| --- | ---: | ---: |
| Warm 8K / 64K prefill tok/s | 2051.75 / 2032.29 | 2709.46 / 2604.22 |
| C1 / C16 aggregate emitted tok/s | 54.040 / 147.139 | 53.679 / 150.457 |
| C1 / C16 request-step-weighted ms/step | 57.523 / 207.093 | 56.985 / 210.433 |
| C1 / C16 emitted tokens/step | 3.13115 / 2.57023 | 3.08065 / 2.57023 |
| Worker execute median ms, 8K / 64K | 25.9045 / 25.918 | 14.059 / 13.953 |
| Worker inter-service gap median ms, 8K / 64K | 0.015 / 0.015 | 5.425 / 5.813 |
| Worker inter-service gap p95 ms, 8K / 64K | 4.750 / 4.437 | 7.496 / 10.026 |
| Reply worker retransmits, 8K / 64K | 16 / 3078 | 7 / 1298 |
| Reply coordinator OOS, 8K / 64K | 1395 / 98923 | 4578 / 70172 |
| Reply worker sequence events, 8K / 64K | 627 / 98918 | 4583 / 73276 |
| Worker TX amplification, 8K / 64K | 0.9859! / 1.2675 | 1.1328 / 1.1923 |

Spark execute drops 45.7% / 46.2%, while end-to-end prefill improves
32.1% / 28.1%. The candidate remains below the historical 3168/3087 tok/s
8K/64K anchors (0.855x/0.844x); those are absolute targets, not a matched
same-session comparison. C1/C16 ratios are 0.9933x/1.0226x, clearing the
requested 0.98x decode floor. Per-rank summed execute is approximately 3.622 -> 1.975 s
at 8K and 29.243 -> 15.835 s at 64K, against request prefill wall spans
3.962 -> 2.998 s and 32.376 -> 25.285 s. This exposes more non-service
spacing after the compute improvement. The inter-service gap is successive
worker end timestamps minus the next service total; it is not coordinator
CPU idle or a measurement of landing alone. It includes reply landing and
reduction, next-request arrival, coordinator GPU work and scheduling.
Console handoffs between 16K chunks remain 7-12 us. On ostrich's timed
8K request, candidate same-layer row-unit gaps have a 5.700 ms median
(70 pairs), versus 4.873 ms between layers (69 pairs); both baseline
medians are 0.015 ms. Thus the gap is not solely next-layer attention.
Full-4096-row 64K per-rank mean execute ranges 25.365-26.171 ms before
and 13.891-14.266 ms after, confirming the gain across all six ranks.
Approximate epoch alignment clips a few 64K boundary records (1189-1190
per worker); sums are approximate. The 8K amplification below one is
sampling-skew-invalid.

These counters support residual reply contention/replay, not a causal split
between incast, landing and coordinator work. Traffic-bearing links remain
100G on Sparks and 400G on raptor. Crucially, the candidate's actual prefill
intake is **pinned**, not GPU: its coordinator startup line 70 logs
`mode="pinned"` with `reason=executor retains pinned uploads`; the earlier
line 55 selects GPU only for the generic/decode transport. Thus the measured
2709/2604 tok/s prefill did not use direct-GPU landing. The direct-GPU
19.6 GB/s (157G) proxy and 15.41 ms estimate cannot explain this prefill's
critical path. The relevant staged-path proxy is the logged harmonic
27.4 GB/s (219G, about 11.02 ms per full reply wave), still not measured
remote-wave throughput. Host ingress and subsequent H2D remain distinct
queues; that harmonic estimate is not a NIC ingress ceiling. The prefill
landing-mode axis is moot on this frozen multiwave build.

Both arms pass prompt/turn-end byte-exact cache restore and cold recompute,
128-token spec/no-spec equality and the strict 10/10 lossless set. Strict
set success is not universal row-width-invariance proof. Quick KL changes
0.036301322 -> 0.037048855, so the unchanged-KL condition does not pass.
Object/source audit identifies a changed K4 indexer arithmetic route:
shared FP8 GEMV shape `(4096,2048)` selects two warps rather than the
former four, also affecting `_IndexProducerW8` rather than only Flash's
shared-down projection. Only `glm_index_producer_m64` changes among 90
hashed GLM coordinator objects; six of eight fidelity windows are identical.
A separate fix is outside this frozen measurement.

Receipts: `kernel-batch-k4/summary.json` and the two
`kernel-batch-k4/glm53-exl3-max-{base,cand}-r1/result.json` files under
`~/.cache/cuteafd/builds/`. Detailed tables and conditions are on #729 and
#703. No newer source is mixed into the measurements.

### Null: EXLR8 up1 candidate correctness failure

The same frozen baseline/candidate max-layout screen does not yield an
accepted EXLR8 comparison. Candidate aborts during the C16 warmup with
`ExpertProtocolV2 route gate_weight must be finite; queued streams drained;
engine cannot be reused` (2026-10-24T21:49:25Z). Candidate 8K, KL, cache,
spec and matched decode sweep are therefore unavailable, not regressions
quantified by a valid ratio. Baseline alone passes quick KL 0.028109469,
exact cache/spec and strict 10/10, with C1/C16 59.043/141.878 tok/s and
8K 2299.867 tok/s. Earlier partial baseline-panel decode numbers are not
accepted paired results. Both arms use a 2,849,024-token pool and zero
onboarded layers. The run finishes naturally; no source fix or retry is
mixed into this measurement. Detailed failure receipts are on #729/#703.

### Measured/null: explicit GPU versus pinned, same frozen seal

One launch per setting, same candidate seal `974d235de`, GLM K4,
2 RTX + 6 Sparks, context attention, five TP2 onboard layers and the
2,097,152-token pool. Transport timing enabled; one untimed warm request
per prefill length. The runner completed both attempts naturally.

| Metric | Explicit GPU | Explicit pinned |
| --- | --- | ---: |
| Launch / paired validity | Failed before readiness / invalid | Completed / unmatched |
| Readiness s | NA | 50.749 |
| Warm 8K / 64K prefill tok/s | NA | 2662.270 / 2537.890 |
| 8K / 64K TTFT s | NA | 3.059 / 25.999 |
| C1 / C16 aggregate emitted tok/s | NA | 57.103 / 134.200 |
| Strict lossless set | Not run | 10/10 pass |
| Reply worker retransmits, 8K / 64K | NA | 39 / 1798 |
| Reply worker sequence events, 8K / 64K | NA | 5359 / 80640 |
| Reply coordinator OOS, 8K / 64K | NA | 5359 / 80655 |
| Worker TX amplification, 8K / 64K | NA | 1.11649 / 1.21274 |

The explicit-GPU arm fails with `GPU landing has one landing range per rank`.
`SparkIntake::with_waves` rejects GPU mode when `waves > 1`
(`rust/crates/cuteafd-daemon/src/shared/spark_intake.rs:428`);
`SparkLink::new_waves` otherwise chooses a pinned preference at line 1440.
An explicit GPU setting overrides that preference and reaches the rejection.
The earlier successful 19.63 GB/s local probe and generic GPU startup lines
do not establish that the multiwave prefill executor can use GPU landing.

There is **no valid GPU/pinned throughput ratio or causal landing result**.
The pinned arm supplies standalone measurements only; its paired verdict
fails because the GPU arm has no prompt hashes. KL, cache-restore and the
separate spec/no-spec panel were not requested in this landing screen and
are absent, not passed. The strict set checks its ten cases, not universal
row-width invariance. Root ports remain 100G on Sparks and 400G on raptor.
Four-Spark measurement was skipped: this frozen export seal lacks its
width-512 slices and another build was not justified at 93% root usage.

Receipts: `kernel-incast-tp6/summary.json` and
`kernel-incast-tp6/glm53-exl3-max-{gpu,pinned}-r1/result.json` under
`~/.cache/cuteafd/builds/`. No source or network changes, no repeat.

### Measured: candidate-only coordinator Nsight prefill trace

The initial forced-GPU capture fails at the same multiwave guard and writes
no report. One authorized retry leaves intake at `auto`, uses the same frozen
candidate, five onboard layers, pool and attention policy, and captures only
prefill with `PROFILE=nsys,transport-timing`. It completes successfully with
an 83,776,679-byte coordinator report. No Spark GPU trace is collected.
The first 8K warm request is outside capture; 64K warmup is inside capture
but excluded from the timed windows below. Nsight perturbs execution:
client rates are 2100.01/2126.24 tok/s, not a replacement speed card or a
paired regression. No quality/cache/spec panels run in this diagnostic.

Console prefill events identify timed requests 4 and 6. Their process epoch
is aligned to `TARGET_INFO_SESSION_START_TIME.utcEpochNs` in the exported
SQLite trace. Durations below use clipped interval unions across both RTX
GPUs, not sums of overlapping kernels. H2D bytes count completed copies
inside the window; boundary clipping can affect a few records.

| Coordinator trace metric | Timed 8K | Timed 64K |
| --- | ---: | ---: |
| Console prefill wall s | 3.4310 | 30.9772 |
| Any traced CUDA activity union s | 2.1325 | 18.6096 |
| Neither GPU has traced CUDA activity s | 1.2985 | 12.3676 |
| H2D bytes GB / active copy union s | 42.026 / 0.7731 | 338.232 / 6.2268 |
| H2D bytes / summed copy duration GB/s | 54.36 | 54.32 |
| Peer push/flag-wait kernel union s | 1.2689 | 10.4803 |
| Other kernel union s | 0.9070 | 8.7201 |
| Compact reducer summed kernel s | 0.0362 | 0.2922 |

Peer push includes data movement and flag publication, not just idle wait.
Kernel categories overlap each other and H2D, so these rows must not be
added into a serial critical-path budget. `cudaEventSynchronize` summed
host-thread durations are 1.904/16.866 s; they can overlap device work and
each other and are not CPU execution time. No NVTX ranges appear in this
export despite the trace selection, limiting phase attribution.

Approximate epoch alignment to ostrich's worker timings gives a more direct
view of inter-service spacing in this **profiled** run. There are 140/1188
worker records inside the timed windows; a few 64K boundaries are clipped.
Execute medians remain 14.232/14.221 ms, but gap medians grow to
8.317/10.436 ms under profiling, versus 5.425/5.813 ms in the original
six-worker untraced-CUDA A/B. Do not transfer the following fractions to
those original gaps as if profiling were free.

| Coordinator activity inside ostrich inter-service gaps | 8K | 64K |
| --- | ---: | ---: |
| Gap count / total gap s | 139 / 1.2956 | 1187 / 14.7113 |
| H2D union s | 0.6657 | 4.2495 |
| Peer push/flag-wait union s | 0.6223 | 4.2691 |
| Other kernel union s | 0.0991 | 1.7157 |
| Any traced CUDA union s | 0.7870 | 6.3049 |
| No traced CUDA on either RTX s | 0.5086 | 8.4064 |

Again, the activity rows overlap: they are coverage, not additive causal
shares. The trace establishes substantial pinned uploads and peer exchange
within the gaps, with the compact reducer a much smaller summed GPU cost.
It also leaves 39.3%/57.1% of those profiled gaps without traced CUDA.
That remainder can include remote reply/network wait, host scheduling and
coordination; it is not proof of CPU idle, incast latency or an isolated
landing stall. H2D reaches about 54.3 GB/s while copying, so the 27.4 GB/s
sequential host-plus-H2D probe model is not the measured copy-engine rate.
The residual path is mixed transfer/synchronization/coordination; exact
receive/post/land critical-path attribution remains unavailable.

Receipts: `kernel-incast-nsys-auto/glm53-exl3-max-cand-r1/result.json`,
`console-frames.jsonl`, `smoke/logs/*.ostrich.log` and
`profile/{nsys-raptor.nsys-rep,nsys-raptor.sqlite,interval-analysis.json}`
under `~/.cache/cuteafd/builds/`. Export uses the matching dev image's
Nsight 2026.2 CPU-only, as host Nsight 2026.1 cannot read the newer report.
No source/network changes or further hardware repeat.

## Receipt inventory

All paths below are relative to `/home/tj/.cache/cuteafd/builds/` and name
`result.json`; associated startup evidence is under each receipt's
`smoke/logs/*coordinator.log`. Every selected active worker port reports
100G and coordinator 400G. N is Spark rank count (expert-TP layouts here),
not coordinator GPU count. G/V2/V4 are the full-capacity byte shapes above;
V4.1 max's actual remote replay can be Vr despite V2 allocation capacity.
GPU/pinned/host/H2D rate columns are GB/s; pinned is the logged harmonic
estimate, not independently measured ingress. The effective mode is the
final logged prefill choice, not the earlier generic transport choice.


| ID | Receipt path | N / shape | Intake | GPU / pinned / host / H2D |
| --- | --- | --- | --- | --- |
| 1 | `transport-combined-exlr8max/glm53-exlr8-max-base-r1/result.json` | 6 / G | pinned | 19.6 / 27.5 / 52.4 / 57.8 |
| 2 | `transport-combined-exlr8max/glm53-exlr8-max-cand-r1/result.json` | 6 / G | gpu | 19.6 / 27.5 / 52.4 / 57.7 |
| 3 | `transport-combined-glmmax/glm53-exl3-max-base-r1/result.json` | 6 / G | pinned | 19.6 / 27.5 / 52.4 / 57.8 |
| 4 | `transport-combined-glmmax/glm53-exl3-max-cand-r1/result.json` | 6 / G | gpu | 19.6 / 27.5 / 52.4 / 57.8 |
| 5 | `transport-combined-glmtrace/glm53-exl3-max-base-r1/result.json` | 6 / G | pinned | 19.6 / 27.5 / 52.4 / 57.8 |
| 6 | `transport-combined-glmtrace/glm53-exl3-max-cand-r1/result.json` | 6 / G | gpu | 19.6 / 27.4 / 52.4 / 57.5 |
| 7 | `transport-combined-mimomax/mimo26-pro-max-base-r1/result.json` | 6 / G | gpu | 19.6 / 27.5 / 52.4 / 57.8 |
| 8 | `transport-combined-mimomax/mimo26-pro-max-cand-r1/result.json` | 6 / G | gpu | 20.0 / 27.2 / 52.1 / 56.9 |
| 9 | `transport-combined-v41max/v41-flash-max-base-r1/result.json` | 6 / V2 | pinned | 18.9 / 27.3 / 52.4 / 57.0 |
| 10 | `transport-combined-v41max/v41-flash-max-cand-r1/result.json` | 6 / V2 | pinned | 18.9 / 27.5 / 52.4 / 57.8 |
| 11 | `land-714-2b936-glm/glm53-exl3-max-cand-r1/result.json` | 6 / G | gpu | 19.5 / 27.5 / 52.4 / 57.8 |
| 12 | `transport-thread-glmmax/glm53-exl3-max-base-r1/result.json` | 6 / G | pinned | 19.7 / 27.5 / 52.4 / 57.8 |
| 13 | `transport-thread-glmmax/glm53-exl3-max-cand-r1/result.json` | 6 / G | pinned | 19.9 / 27.4 / 52.2 / 57.7 |
| 14 | `measure408-f4e1/glm53-exl3-max-current-r1/result.json` | 6 / G | pinned | 19.6 / 27.5 / 52.4 / 57.8 |
| 15 | `measure408-f4e1/glm53-exlr8-max-current-r1/result.json` | 6 / G | pinned | 19.9 / 27.3 / 52.4 / 57.0 |
| 16 | `glmmin-mimopro-09db/glm53-exl3-min-base-r1/result.json` | 4 / G | pinned | 19.0 / 27.5 / 52.4 / 57.8 |
| 17 | `glmmin-mimopro-09db/mimo26-pro-max-base-r1/result.json` | 6 / G | gpu | 19.7 / 27.5 / 52.4 / 57.8 |
| 18 | `exlr8-a8-max-up1/glm53-exlr8-max-a16-r1/result.json` | 6 / G | pinned | 19.9 / 27.3 / 51.8 / 57.8 |
| 19 | `exlr8-a8-max-up1/glm53-exlr8-max-a8-r1/result.json` | 6 / G | pinned | 19.7 / 27.5 / 52.4 / 57.7 |
| 20 | `exlr8-a8-min-bit325/glm53-exl3-min-a16-r1/result.json` | 4 / G | pinned | 19.2 / 27.5 / 52.4 / 57.7 |
| 21 | `exlr8-a8-min-up1/glm53-exl3-min-a16-r1/result.json` | 4 / G | pinned | 19.2 / 27.5 / 52.4 / 57.8 |
| 22 | `exlr8-a8-min-up1/glm53-exl3-min-a8-r1/result.json` | 4 / G | pinned | 18.9 / 27.4 / 52.4 / 57.6 |
| 23 | `v41-async-max/v41-flash-max-base-r1/result.json` | 6 / V2 | pinned | 19.2 / 27.3 / 52.4 / 57.2 |
| 24 | `v41-async-max/v41-flash-max-c1-r1/result.json` | 6 / V2 | pinned | 18.9 / 27.5 / 52.3 / 57.8 |
| 25 | `v41-async-max/v41-flash-max-c2-r1/result.json` | 6 / V2 | gpu | 18.9 / 27.4 / 52.4 / 57.4 |
| 26 | `v41-async-min/v41-flash-min-base-r1/result.json` | 3 / V2 | pinned | 19.7 / 27.5 / 52.4 / 57.8 |
| 27 | `v41-async-min/v41-flash-min-c1-r1/result.json` | 3 / V2 | pinned | 19.9 / 27.5 / 52.4 / 57.7 |
| 28 | `v41-async-min/v41-flash-min-c2-r1/result.json` | 3 / V2 | gpu | 19.6 / 27.5 / 52.4 / 57.8 |
| 29 | `v41-before09db/v41-flash-max-base-r1/result.json` | 6 / V2 | pinned | 18.9 / 27.5 / 52.4 / 57.8 |
| 30 | `v41-ffn-graphs-ab1/v41-flash-max-base-r1/result.json` | 4 / V2 | pinned | 19.0 / 27.5 / 52.4 / 57.7 |
| 31 | `v41-ffn-graphs-ab1/v41-flash-max-cand-r1/result.json` | 4 / V2 | pinned | 19.1 / 27.4 / 52.4 / 57.3 |
| 32 | `v41-max6/cards-tp4-systemd/v41-flash-ref-tp4-cand-r1/result.json` | 4 / V2 | pinned | 19.1 / 27.5 / 52.4 / 57.7 |
| 33 | `v41-max6/cards-tp6-systemd/v41-flash-max-cand-r1/result.json` | 6 / V2 | pinned | 19.1 / 27.3 / 52.4 / 57.1 |
| 34 | `v41-units-ref-91fbc/v41-flash-min-base-r1/result.json` | 4 / V2 | pinned | 19.9 / 27.5 / 52.4 / 57.7 |
| 35 | `v41-units-ref-91fbc/v41-flash-min-cand-r1/result.json` | 4 / V4 | pinned | 19.6 / 27.5 / 52.4 / 57.8 |
| 36 | `glm53-topology09db-run/glm53-exl3-max-four-r1/result.json` | 4 / G | pinned | 19.7 / 27.4 / 52.4 / 57.5 |
| 37 | `glm53-topology09db-run/glm53-exl3-max-sixzero-r1/result.json` | 6 / G | pinned | 19.8 / 27.3 / 52.4 / 56.9 |

Measured results below are **8K / 64K**. Prefer `prefill/8192` over the
separate smoke `prefill/8k` window; where only smoke exists use that window.
NA means unavailable, not zero. Rate/TTFT columns are client end-to-end
prefill metrics from the receipt, not per-wave timings. `!` marks skewed or
inconsistent amplification unsuitable for a byte/replay inference. All raw
phase windows remain in the original results. Unavailable fine-grained wave
timings cannot be reconstructed from these tables.

| ID | Prefill tok/s | Client TTFT s | Worker TX A | Worker retrans | Reply OOS (coord) | Reply seq (workers) | Request OOS (workers) | Request seq (coord) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2297 / 2250 | 3.556 / 29.378 | 1.1236 / 1.1785 | 90 / 796 | 6049 / 74138 | 6046 / 75930 | 342 / 0 | 342 / 0 |
| 2 | 2284 / 2211 | 3.562 / 29.802 | 1.2022 / 1.1827 | 1028 / 9321 | 4594 / 26598 | 4593 / 26649 | 0 / 0 | 0 / 0 |
| 3 | 2066 / 1997 | 3.954 / 33.045 | 1.0920 / 1.1988 | 122 / 1835 | 3391 / 71193 | 3391 / 71683 | 174 / 73 | 174 / 72 |
| 4 | 2139 / 2104 | 3.803 / 31.315 | 1.1195 / 1.1502 | 55 / 1294 | 6959 / 54165 | 6165 / 53919 | 0 / 0 | 0 / 0 |
| 5 | 1965 / 1720 | 4.148 / 38.374 | 0.9706! / 1.0176 | 0 / 23 | 416 / 0 | 0 / 0 | 720 / 4882 | 872 / 4858 |
| 6 | 1867 / 2073 | 4.365 / 31.835 | NA / 1.1443 | NA / 1727 | NA / 48353 | NA / 47015 | NA / 0 | NA / 0 |
| 7 | 3437 / 2798 | 2.350 / 22.867 | 1.2834 / 1.1801 | 1638 / 7197 | 3833 / 20896 | 4407 / 20690 | 12 / 679 | 12 / 1096 |
| 8 | 3484 / 2816 | 2.328 / 22.834 | 1.1393 / 1.1527 | 806 / 4417 | 1818 / 24724 | 1817 / 23819 | 0 / 0 | 0 / 0 |
| 9 | 6603 / 8868 | 1.246 / 7.693 | 0.0000! / 1.0143 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| 10 | 8742 / 9183 | 0.941 / 7.429 | 1.0144 / 1.0144 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| 11 | 2080 / 1701 | 3.916 / 38.788 | 1.0869 / 1.2440 | 255 / 16390 | 2257 / 19925 | 1909 / 19921 | 0 / 0 | 0 / 0 |
| 12 | 1804 / 2021 | 4.544 / 32.758 | 1.0777 / 1.1763 | 77 / 2033 | 1710 / 59066 | 1709 / 60876 | 3239 / 35 | 3053 / 35 |
| 13 | 1991 / 2022 | 4.102 / 32.694 | 0.9981! / 1.0979 | 0 / 949 | 27 / 29416 | 27 / 30841 | 1818 / 770 | 1818 / 692 |
| 14 | 2060 / 2025 | 3.964 / 32.574 | 1.0778 / 1.1759 | 104 / 1694 | 2932 / 58993 | 2932 / 61815 | 1135 / 633 | 1135 / 598 |
| 15 | 2324 / 2148 | 3.515 / 30.713 | 1.1366 / 1.1881 | 120 / 1740 | 6561 / 71782 | 6561 / 73087 | 544 / 0 | 544 / 0 |
| 16 | 2926 / 2750 | 2.781 / 23.915 | 1.0438 / 1.0108 | 2 / 1 | 0 / 38 | 0 / 38 | 269 / 133 | 19 / 20 |
| 17 | 3439 / 2770 | 2.340 / 23.014 | 1.0804 / 1.1856 | 763 / 7328 | 1563 / 20443 | 1395 / 20299 | 27 / 918 | 28 / 916 |
| 18 | 2179 / 2128 | 3.739 / 31.007 | 1.1570 / 1.2273 | 219 / 2853 | 8668 / 83638 | 6488 / 85314 | 321 / 0 | 33 / 0 |
| 19 | 2332 / 1928 | 3.493 / 34.223 | 1.1439 / 1.2114 | 61 / 2339 | 8901 / 85841 | 7653 / 85840 | 295 / 4 | 203 / 4 |
| 20 | 2453 / 2330 | 3.321 / 28.321 | 1.0144 / 1.0140 | 0 / 0 | 0 / 28 | 0 / 28 | 35 / 51 | 35 / 51 |
| 21 | 2168 / 2067 | 3.757 / 31.923 | 0.9533! / 1.0159 | 0 / 0 | 0 / 0 | 0 / 0 | 113 / 570 | 172 / 475 |
| 22 | 2393 / 2226 | 3.404 / 29.643 | 1.0600 / 1.0209 | 0 / 0 | 0 / 19 | 0 / 19 | 85 / 51 | 58 / 42 |
| 23 | 6021 / NA | 1.363 / NA | 1.0143 / NA | 0 / NA | 0 / NA | 0 / NA | 0 / NA | 0 / NA |
| 24 | 8448 / NA | 0.971 / NA | 1.4453 / NA | 0 / NA | 0 / NA | 0 / NA | 0 / NA | 0 / NA |
| 25 | 8582 / NA | 0.956 / NA | 1.0143 / NA | 0 / NA | 0 / NA | 0 / NA | 0 / NA | 0 / NA |
| 26 | 5360 / NA | 1.531 / NA | 1.1069 / NA | 0 / NA | 0 / NA | 0 / NA | 1013 / NA | 829 / NA |
| 27 | 4392 / NA | 1.868 / NA | 0.8389! / NA | 6 / NA | 0 / NA | 0 / NA | 1416 / NA | 1949 / NA |
| 28 | 4930 / NA | 1.664 / NA | 1.1343 / NA | 1 / NA | 0 / NA | 0 / NA | 359 / NA | 275 / NA |
| 29 | 6254 / 8890 | 1.316 / 7.688 | NA / 2.0285! | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| 30 | NA / NA | NA / NA | 1.0656 / NA | 0 / NA | 0 / NA | 0 / NA | 492 / NA | 126 / NA |
| 31 | NA / NA | NA / NA | 1.0656 / NA | 0 / NA | 0 / NA | 0 / NA | 207 / NA | 166 / NA |
| 32 | NA / NA | NA / NA | 1.0654 / NA | 0 / NA | 0 / NA | 0 / NA | 426 / NA | 165 / NA |
| 33 | NA / NA | NA / NA | 1.0834 / NA | 0 / NA | 0 / NA | 0 / NA | 165 / NA | 73 / NA |
| 34 | 5030 / 6684 | 1.631 / 10.206 | 1.1372 / 1.0144 | 0 / 0 | 0 / 0 | 0 / 0 | 727 / 621 | 759 / 585 |
| 35 | 4618 / 5988 | 1.776 / 11.393 | 1.1962 / 1.0098 | 9 / 94 | 0 / 0 | 0 / 0 | 566 / 9719 | 654 / 10194 |
| 36 | 2929 / 2817 | 2.788 / 23.419 | 1.0577 / 1.0190 | 0 / 0 | 4 / 28 | 4 / 28 | 141 / 42 | 51 / 27 |
| 37 | 1982 / 2001 | 4.122 / 32.977 | 1.0127 / 1.0991 | 19 / 1318 | 777 / 34224 | 749 / 34264 | 1111 / 28 | 1111 / 6 |
