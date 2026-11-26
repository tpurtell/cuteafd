# Options

Every user-facing launch key and CLI option, grouped by area, with defaults and
the values each launch path accepts. The family columns in the last section are
tested against the launcher (`scripts/tests/test_options_doc.py`).

Release builds (`./build.sh`) and publication share the shipping matrix reported
by `scripts/build/release-scope.py default-matrix`. `CUTEAFD_RELEASE_MATRIX`
overrides that default; `--scope` selects families or cards for development builds,
while publication requires `all` and complete image inventories.

## Benchmark panel configuration

Before any timed profile or sweep, the server runs one untimed, authenticated
warm prelude per serving instance: code, prose, JSON, then code, prose, JSON,
at the sweep output-token count (192 for a direct profile). The fixed prompts
and nonces are identical across arms. Existing per-point warm waves remain.
The prelude receipt is cached by the server; subsequent calls do no inference.
Missing receipts or a different server-instance identity refuse timing. The
receipt is embedded in `report.json` and `sweep.json`, including each request's
cycles, selected mean width and final policy telemetry when exposed. Profiling
windows open afterward. This is deterministic conditioning, not convergence:
K3.1 traces show initial width drift over 64-128 cycles, and about 316 cycles
for one 256-token triple; two triples provide roughly 630 cycles of conditioning.
Timed request records retain a 120-character opening without classifying it;
unknown per-request cycle counts are explicitly unavailable, not overlapping
global-counter estimates.

`cuteafd bench --panels decode_content --panel-config
'decode_content={"contents":["code"],"temperature":1.0,"top_p":0.95,"seed":42}'`
runs sampled C1 code decode. `scripts/bench/wip-cards.py` accepts the same
repeatable `--panel-config PANEL=JSON` option; smoke matrix entries/defaults use
`"panel_config":{"decode_content":{"contents":["code"],"temperature":1.0,
"top_p":0.95,"seed":42}}`. Entry settings replace defaults for the same panel.
The in-server run API accepts this same `panel_config` map. The `decode_content`, `concurrency` and
`spec_lossless_set` panels accept configuration; unknown panels and fields fail before
running. This does not change the baseline or goldens.

`concurrency={"sustained":true,"multiplier":4}` adds continuous-refill rows after
the existing closed-batch concurrency rows (which retain their schema and workload).
`sustained` defaults to false; `multiplier` is an integer in 2..64, default 4.
The in-server concurrency panel uses its usual levels. For an external sweep use
`wip-cards.py --levels 8,16 --panel-config 'concurrency={"sustained":true}'`
(with the usual kit/arms), or set that same `panel_config` in the smoke matrix.
`--sweep-tokens` still controls the external sweep's per-request output budget.

At level C, C workers immediately refill from a pool of N = multiplier * C
requests. The steady window starts at the first completion and ends when the
last refill is dispatched, excluding initial warm-up and final drain. Requests
are indexed before dispatch, so scheduling cannot change the prompt/nonces across
arms. The pool uses `CUTEAFD_BENCH_NONCE_SEED` (or `sustained-v1`) and records
`pool_sha256`; use a fresh shared seed for independent pairs to avoid cache reuse.
The prior untimed width warm-ups and all closed-batch points run first.

Sustained `aggregate_tok_s` is an **estimate**: exact final usage token totals are
credited pro rata by each request's first-to-last-content-token overlap with the
window, assuming uniform per-request emission (SSE text chunks are not tokens).
Only output/decode tokens enter the numerator; refill prompt/prefill work consumes
time inside the window but contributes no prompt tokens to throughput. This is
the intended under-load decode rate, not prefill throughput.
This includes boundary-crossing requests without crediting their warm-up/drain
output. `sustained` records relative window bounds, duration, initial/total/refill/
completed and overlapping request counts, fractional token credits, the explicit
accounting assumption, and every request's send/completion times, token span and
exact usage total. Errors invalidate the sustained run instead of silently
reducing concurrency. Thermal/fabric/counter observations cover the full refill
run including its warm-up and drain, not just the steady window. External sweep
rows use `phase: "sustained"`; wip-cards reports `sustained C8`, `sustained C16`,
etc., separately from `sweep C8`/`sweep C16`.

`decode_content` fields are optional: `contents` selects unique names from
`code,prose,json,math,chat,translation,summary,table` (all by default),
`temperature` is 0..2 (default 0, greedy), `top_p` is (0,1] (omitted by default,
using the server's default), and `seed` is a signed 64-bit request seed (omitted
by default). Explicit settings are recorded per content row and displayed on
the report. A seed repeats the sampling RNG; prompt nonces and dynamic scheduling
still need matching conditions for fully reproducible comparisons. Unconfigured
requests retain exactly their historical greedy body.

`cuteafd bench --panels spec_lossless_set --panel-config 'spec_lossless_set={"warm":true}'`
opts into prefix-restore qualification (default `warm:false`, the unchanged cold
probe pair). Warm sets add deterministic shared reference context, with served-tokenizer
counts proving at least 1,024 total prompt tokens and twice the configured host-retention
minimum. For each case it first primes that full prompt with a cache-eligible
speculation-on request (`cold:false`), then measures the identical full prompt
with speculation on. The comparison oracle remains the case's cold speculation-off
arm, never the priming output. Every measured restore must cover the entire
prompt-end snapshot (`cached_tokens >= prompt_tokens`); partial or absent restores fail. Strict generated-token and row-hash
comparison still applies; streamed content/reasoning bytes and tool calls must
also equal the off arm. Warm rows report cached tokens, identical generated-token
prefix length, output equality and the first differing row. Existing row dumps
capture the measured on/off arms, not the primer. `lossless_concurrency` still
bounds cases per arm; each case's prime completes before its warm measurement.
The same map works through the run API, smoke `panel_config` and wip-cards;
unknown settings fail validation. This opt-in adds no launch key or env knob.

Smoke entries take `"sweep_verify":{"shapes":["4x8","5,4x7","4x8@64"],"context":1024,
"repeats":20}` (wip-cards `--verify-shapes`, [details](wip-cards.md#subsets-traces-profiles-preflight-and-resume))
for warmed verify-only exact-shape timing on GLM Flash and GLM 5.3; it runs instead
of `levels`/`prefill_lengths`. It adds no launch key: the server takes it as an
authenticated benchmark probe (`verify_timing`) and refuses it while requests are in flight.

Draft acceptance rate remains accepted draft tokens / proposed draft tokens.
Acceptance length is **1 + accepted draft tokens / draft verify passes**: one
bonus/target token per draft pass, including passes accepting no draft tokens.
`/v1/stats` exposes `totals.draft_verify_passes`, a lifetime host-side counter
incremented once per sequence per verification step that had a proposed or
verified draft, not once per batch; anchor-only steps are excluded. No device
synchronization is added. Content rows and cold-C1 sweep counters use interval
deltas; length is absent/null when the server lacks this counter or no draft
passes occurred. Never substitute `verification_rounds` (batch steps, including
anchor-only decode) as its denominator. Existing rates remain available on old
servers. Greedy and sampled figures must be compared separately.

## Optional performance levers

Optional levers that are off by default and trade precision or host resources
for speed or memory at serve time. Every entry starts from the checkpoint's
native format or the most conservative behaviour; the effect is measured on the
stated hardware or marked unmeasured. Debug, profiling and capacity knobs are
documented in their own sections, not here. A PR that adds an optional,
default-off performance lever adds its row here in the same PR (AGENTS.md).

| Lever | Families | Enable | Effect | Cost |
| --- | --- | --- | --- | --- |
| `DECODE_LANES` | V4.1 | `DECODE_LANES=1` or `DECODE_LANES=2` (unset = layout default: `1` on 1× RTX, `2` on 2× RTX; `1` at `CONCURRENCY=1`) | Independent decode lanes at `DECODE_ROWS=64` (default): one lane runs a single batch of up to `min(CONCURRENCY, 64)` requests and returns the second lane's workspaces, graphs, Spark link and decode execution to VRAM (KV pool or an extra local expert layer). Per-layout default chosen from matched V4.1 StridedPLE A/Bs (`10fa3a8e8c` & `a5aa26864b`, post-#2760 lane-scaled graph reserve, dSpark, warmed interleaved matched-prompt launches, 2026-11-25, 0.0 s power-cap/thermal slowdown, [#2460](https://github.com/tpurtell/cuteafd/issues/2460#issuecomment-7165698474), [#2460 (1×128 vs 1×64 vs 2×64)](https://github.com/tpurtell/cuteafd/issues/2460#issuecomment-7168830736)): **1× RTX min** (1× RTX PRO 6000 + 3 Sparks, 9 local layers, graph reserve 2,466→1,286 MiB) ties at C1–C2 and wins C4–C16 and 32K C8 with 1 lane under both policies — 1L/2L C1/C2/C4/C8/C16/32K C8 is 1.01/0.98/1.12/1.06/1.18/1.13× (130.8/151.4/237.0/308.9/440.4/246.3→131.7/148.3/266.3/328.4/519.8/277.4 tok/s) under `DRAFT_POLICY=history` and 1.02/1.00/1.22/1.07/1.05/1.16× (+4.6% to +21.5% across C4–C16/32K C8) at `DRAFT_LENGTH=7`, with +20.4% KV pool (7,545,856→9,084,928 tokens) and faster readiness (29.85→27.36 s); **2× RTX max** (2× RTX PRO 6000 + 6 Sparks, graph reserve 1,444/1,452→734/738 MiB) keeps 2 lanes by default because 1 lane onboards a 23rd local TP2 expert layer (22→23) that shifts KV pool 8,347,136→3,664,384 tokens while 2 lanes win C16 (1L/2L 0.91–0.93× on `10fa3a8e8c`, 0.888× clean on `a5aa26864b`) and retain the 2.28× larger KV pool and >32K context headroom. | No precision change; one lane splits the verify-row budget (default `64`; configurable via `DECODE_ROWS`) across all requests and caps total configured concurrency at 64; two lanes hold up to 32 each; dSpark drafts a confidence-selected subset when the 32-request native draft capacity would be exceeded. |
| `DECODE_ROWS` | V4.1, GLM Flash, MiniMax, Inkling, Kimi K3, Qwen4, Muse | `DECODE_ROWS=128` or `256` (default `auto` = `64`; GLM Flash up to `128` on 1 GPU; MiniMax, Inkling, Kimi K3, Qwen4 and Muse up to `128` on 1 or 2 GPUs; other families refuse `>64` pending exports) | Raises the per-lane decode/verify row budget up to `CONCURRENCY × (max_draft + 1)` within the family's exported decode programs so high-concurrency batches can verify deeper drafts in one pass; effect is workload/acceptance-dependent. Measured (2026-11-23, one warmed interleaved matched-prompt launch/arm, no thermal slowdown): GLM Flash 1× RTX PRO 6000 + 2 Sparks (`glmf-exl3-min`, K3.25-Core-FP8, DFlash2, KV pool 2,375,168→2,343,168, −1.35%): on shipping pin `e1d168c7` (`a711428996`, lower-acceptance-tail prompt nonce, `pwr_cap` 1→2 s / 0→1 s) C16 312.3→305.6 tok/s (−2.1%) under `DRAFT_POLICY=history` and 281.2→260.7 (−7.3%) at `DRAFT_LENGTH=7`; on prior pin `c29797e1` (`302a6bd3cc`, higher-acceptance-tail prompt nonce) C16 300.5→313.5 (+4.3%) under `history` and 293.9→299.6 (+1.9%) at `DRAFT_LENGTH=7` (C1 flat in both sessions). V4.1 2× RTX PRO 6000 + 6 Sparks (`v41-flash-max`, dSpark, prior pin `c29797e1` at `7505936344`, 0.0 s power-cap): `DECODE_LANES=1 DECODE_ROWS=128` vs default `2×64` is 0.79/0.88/0.84× at C4/C8/C16 (442.2/605.3/891.0→349.2/532.6/745.4) under `history` and 0.80/0.95/0.97× (429.6/570.7/828.0→344.1/539.3/805.4) at `DRAFT_LENGTH=7` (recovering from #2460's 0.73× at `1×64` C16; C1 flat; KV pool 7,520,768→7,347,712, −2.30%). MiniMax M3.1 Flash 2× RTX PRO 6000 + 6 Sparks (`minimax-flash-max`, dSpark, Heads, `DRAFT_POLICY=clamped`, 2026-11-24, 0.0 s power-cap/thermal slowdown, capture-free): `DECODE_ROWS=128` vs `64` is flat on short-context sweep C1/C4/C8/C16 (76.3/175.8/261.1/332.0→73.6/178.1/255.7/329.7 tok/s) and distinct-v1 C1–C16 (67.4/117.7/157.7/213.7→66.4/116.6/157.9/213.0), and **+8.7%** at 32K C16 (224.9→244.6 tok/s, median 15.2→16.7, 32K C1 65.0→65.9; KV pool 2,089,408→2,081,728, −0.37%; 1+3 min card 1,048,576→1,048,576, 0%). Inkling 2× RTX PRO 6000 + 6 Sparks (`inkling2-exl3-max`, K4-MC original, dSpark, `DRAFT_POLICY=clamped`, 2026-11-24, `b766e8ccab`, one interleaved matched-prompt launch/arm, 0.0 s thermal/power-cap on `concurrency` and `sweep`, 32K `sw_power_cap` 7.0/2.0→6.0/2.0 s, KV pool 2,408,896→2,379,136, −1.24%): `concurrency` C1/C2/C4/C8/C16 73.9/104.7/147.0/199.1/265.0→74.0/104.5/146.4/199.3/267.4 tok/s (−0.4% to +0.9%), 32K `context_load` C1/C8/C16 42.6/202.4/296.6→42.6/200.0/301.3 (−1.2% to +1.6%), `sweep` C1/C4/C8/C16 134.5/325.1/365.1/602.7→134.3/321.8/367.9/865.0 (+43.5% at C16), readiness 33.53→32.06 s, strict lossless 10/10 at C16 on both arms. Kimi K3.1 2× RTX PRO 6000 + 6 Sparks (`kimi-k31-mc-max`, K2-MC original, dSpark, `DRAFT_POLICY=clamped`, 2026-11-24, 0.0 s power-cap/thermal slowdown, 0 captures): `DECODE_ROWS=128` vs `64` sweep C1/C2/C4/C8/C16 is 51.1/61.2/122.3/158.8/187.2→39.6/79.7/117.2/127.1/233.2 tok/s (−22.6%/+30.3%/−4.2%/−20.0%/+24.6%; `d128` C1 recorded 372 misses/eager runs vs 0 on `d64`), and 32K C1/C8/C16 is 34.4/113.4/159.4→39.6/124.3/164.2 tok/s (+15.3%/+9.6%/+3.0%); KV pool 2,224,128→2,161,024 (−2.84%; planned 2+6 max 2,225,024→2,161,024, −2.88%; planned 1+6 min 1,793,408→1,735,872, −3.21%), strict lossless 10/10 at C16 on both arms. Qwen4 1× RTX PRO 6000 + 0 Sparks (`qwen38-exl3-min`, local EXL3 K4.25, MTP ×3, `CONCURRENCY=32`, `DRAFT_POLICY=clamped`, 2026-11-24, 0.0 s thermal slowdown, 3.94→9.97 s power-cap, capture-free): `DECODE_ROWS=128` vs `64` is C1/C2/C4/C8/C16 218.0/365.4/620.1/1032.3/1485.0→217.8/363.0/605.9/997.6/1475.8 tok/s, **+25.7%** at C32 (1450.1→1822.4 tok/s, 32×4 = 128 verify rows), and 32K C1/C8/C16 205.7/870.8/1221.8→200.6/835.3/1179.6 tok/s (planned KV pool 1,114,880→1,102,336, −1.13%; live pool 1,016,064→1,003,776, −1.21%; strict lossless 10/10 on both arms; C8 `nsys` pair confirms identical GPU kernel count/geometry and −0.63% GPU kernel time). Muse Spark 2× RTX PRO 6000 + 6 Sparks (`muse-spark-exl3-k4-mc-max`, DFlash2, `DRAFT_POLICY=clamped`, 2026-11-24, one warmed interleaved matched-prompt launch/arm on `d225a2311b`, 0 s thermal slowdown, 32K `sw_power_cap` 13/12→7/20 s): plan-time KV pool unchanged at 2,010,560 tokens on 2+6 (head-owner GPU 1 +102.5 MiB, tight GPU 0 unchanged; 1+4 603,264→601,600, −0.28%; Spark +25.5 MiB/rank on 2+6); strict lossless at C16 10/10 on both arms; `concurrency` C1/C2/C4/C8/C16 40.21/48.63/74.48/101.46/132.17→40.11/48.76/73.62/101.56/130.82 tok/s, 32K `context_load` C1/C8/C16 30.35/104.15/124.20→30.15/104.44/126.13 tok/s, `sweep` C1/C4/C8/C16 73.82/187.84/287.00/264.96→72.98/180.77/290.68/203.41 tok/s (exact-row graph bank recorded 5,222 live captures in timed C16 vs 0 at 64). Default stays `auto` (`64`). | No precision change (controlled 128-row decode-shaped fidelity passes on V4.1, GLM Flash, MiniMax, Inkling, Kimi K3, Qwen4 and Muse); larger per-row workspaces, verify logits, replay records and Spark decode owners reduce the admitted KV pool accordingly. |
| `DRAFT_POLICY` | V4.1, V4, GLM 5.3 / ML4, GLM Flash, MiMo, Qwen4, Qwen dense, Kimi K3, Step-5, Muse, Inkling, Inkling Mini, Hy4-flash, MiniMax | `DRAFT_POLICY=clamped` (default `history`) | `history`'s calibrated product keep with a per-round cut clamp(K / (x + M), 0.3, 0.45), K = 2, M = 1 (`cuteafd_core::clamped`, second refit 2026-11-23 on the live 7-card traces, #2460: worst card 0.996, mean 1.0095 of `history` in counterfactual replay; the first constants K = 3.4, HI = 0.7 measured Qwen4 1+0 C4/C16 0.93/0.95 because round time is concave in rows). x = (N / requests) / e from an online least-squares line t = N + e*rows through each context band's recent per-row-count round times; no time tables or card constants; 0.3 until a band's line exists. Higher concurrency means smaller x and a stricter cut. Live 7-card A/B of the first constants vs `history` (2026-11-23, strict lossless 10/10 on all 14 arms): V4.1, Kimi and MiniMax gained up to 1.10 at C4-C8, Qwen4, Inkling and Muse lost up to 0.93 at C4-C16; the replay that priced rows by the band line (fixed-7, +14.8%) overstated the gain. Expected effect of these constants: about +1% mean, unmeasured live. Both policies share the exploration rule in the `DRAFT_POLICY` reference below. | Drafts fewer positions per request at high concurrency (higher acceptance, more target-only rounds); no precision cost. |
| `DRAFT_CONTEXT_WINDOW` | Kimi K3 | `DRAFT_CONTEXT_WINDOW=2048` (default `0`, native) | Bound full-context draft KV; unmeasured. Lithos K3.1 uses 40 MiB BF16 or 21.25 MiB FP8 per sequence at window 2048, instead of 20,480 or 10,880 bytes per token in the full-context pool. | Changes draft attention/acceptance, not verified target output. Native full context is retained at 0. |
| `GRAPH_PREWARM` | V4.1, V4, GLM 5.3 / ML4, GLM Flash, MiMo, Qwen4, Qwen dense, Kimi K3, Step-5, Muse, Inkling, Inkling Mini, Hy4-flash, MiniMax | `GRAPH_PREWARM=off` (default `on`) | Skip startup prewarm of rank-local graph sites; they capture lazily on first use. Spanning (cross-device) sites always prewarm, because they cannot be captured after readiness (#1878). With donor-always, V4.1 min measured graph high-water 3.114 GB to 0.902 GB; GLM Flash min 52.4 MB to 35.7 MB (2026-11-16, RTX PRO 6000, one warmed launch/arm). V4.1 max (2 RTX + 6 Sparks, dSpark): one prompt-matched launch per arm on dev `21376c2ded` (2026-11-19, no throttling), C1 173.8 → 180.6 tok/s (1.039×), C16 1010.6 → 1046.9 tok/s (1.036×), readiness 30.68 → 28.15 s, strict lossless 10/10 on both. `off` defers rank-local captures to first use in early serving, building a subset of the `on` warm set; startup graph arena fell from ~1.24 GB to 4–6 MiB per GPU. Neither arm captured during C1; C16 warm-up recorded 910 → 8,022 capture attempts and the timed wave 182 → 1,638 (#2506). Other two-GPU max cards (off/on ratios, one launch per arm on one seal, 2026-11-19, no throttling): GLM Flash C1 1.11 / C16 0.97; Kimi K3.1 MC C1 1.00 / C16 1.02; Inkling MC (split head) C1 1.12 / C16 1.05; readiness at most 1.5 s faster. Those pairs passed strict lossless 10/10 with identical fidelity. Other families unmeasured. | First-use capture latency; no intended precision change. Admission reserve is unchanged: lower physical high-water alone does not expand KV; a separately qualified lower explicit reserve is needed. |
| `GRAPH_DONOR_ALWAYS` | V4.1, V4, GLM 5.3 / ML4, GLM Flash, MiMo, Qwen4, Qwen dense, Kimi K3, Step-5, Muse, Inkling, Inkling Mini, Hy4-flash, MiniMax | `GRAPH_DONOR_ALWAYS=on` (default `off`) | Prefer a same-topology cold donor below the soft cap as well as above it. With prewarm off, V4.1 min C1/C16/post-C1 126.02/541.62/123.42 to 125.49/501.17/121.26 tok/s; 165,824 request-time updates/re-keys. GLM Flash min 109.41/393.98/87.17 to 109.02/397.93/87.10 tok/s; 1,270 updates/re-keys. One warmed matched launch/arm, RTX PRO 6000, no thermal slowdown, unchanged KL/top-1. | May replace a reusable cached shape and require a later update; no intended precision change. |
| `GRAPH_DONOR_COLD_LAG` | V4.1, V4, GLM 5.3 / ML4, GLM Flash, MiMo, Qwen4, Qwen dense, Kimi K3, Step-5, Muse, Inkling, Inkling Mini, Hy4-flash, MiniMax | `GRAPH_DONOR_ALWAYS=on GRAPH_DONOR_COLD_LAG=16` (default `0`) | Require age > K + N below the soft cap; pressure keeps age > K. V4.1 min, 1 RTX PRO 6000 + 3 Sparks, DSpark, dev `0edd9b6c4` (2026-11-17), one warmed matched launch/arm: defaults / donor-always lag 0 / lag 16 C1 132.73/125.88/132.99, C16 556.17/499.76/527.38, post-C1 124.24/117.20/123.11 tok/s. Donor arms disable prewarm. Lag 16 cuts request-time updates/re-keys 164,366 to 47,057; physical graph charge 3.043/0.935/1.669 GB (admission reserve unchanged). Its C16/default ratio 0.948 meets the experiment's stop bar, not the 0.98 recovery goal; no further tuning. No thermal slowdown; C16 power-cap seconds 1.004/0/0, max temperatures 60/57/58 C, min T.Limit headroom 38/38/37 C; donor busy clocks unavailable (default 2805 MHz). Other families unmeasured. | Lag 16 retains more graph memory than lag 0; quality unchanged (baseline KL 0.01328, top-1 96.956%). Remains opt-in; no intended precision change. |
| `L2_PREFETCH` | V4.1, MiMo, Qwen4 | `L2_PREFETCH=auto` or a MiB count (unset = off; GLM 5.3 and GLM Flash already default it to `auto`) | Pulls the next layer's weights into L2 while a one-lane decode step waits on its Spark exchange. V4.1 1× RTX 5090 + 4 Sparks: every C1 cell +1.7–3.9%, C4/C8 and 8K prefill unchanged (#623); GLM 5.3 weighted +4.4%, GLM Flash C1 +6–15% (2026-09-30). | None in values (outputs byte-identical); only extra L2 traffic. |
| `QWEN_FP8_DECODE` | Qwen4 | `QWEN_FP8_DECODE=on` (unset = checkpoint BF16) | Holds the GDN/attention projections as one resident E4M3 copy; with `QWEN_FP8_HEAD=on`, 1× RTX code C1 222→247 tok/s, C4 441→489, 8K prefill 6100→6177 (2026-10-03). | Converts target projections to FP8 at load: golden KL 0.036→0.046, top-1 88.5%→86.7%; fails the KL gate, so opt-in only. |
| `ATTENTION_PRECISION` | V4, GLM 5.3 / ML4, GLM Flash | `ATTENTION_PRECISION=fp8` (default `bf16`) | Faster, lower-quality attention tier, only where the family has a qualified path. Model-level speed is unmeasured for this shared key. Prior SM120 MG microbench (#2146, 2752 rows, synthetic medium-entropy keys): BF16 latency / FP8 is GLM 1.07x, GLM Flash 1.03x, Kimi h96 1.08x / h48 1.03x; these are MG-only, not the shipping GLM native prefill baseline. | The optional path can add QK/PV rounding independently of KV storage loss. Same microbench output RMS vs F32: FP8 3.53-3.98% vs BF16 0.21-0.23%; not a Standard model fidelity result. |
| `CORE_VIEW` | GLM 5.3 / ML4, GLM Flash, Kimi K3, Step-5, Muse, Inkling, Hy4-flash, MiniMax | `CORE_VIEW=k8-compact`, `k6-compact` or `fp8-compact` (unset = original) | Stored MC core or generic load-time block FP8 of exactly the original variant's `core_tensors`; BF16 index only. GLM Flash K4 MC, 2 RTX PRO 6000 / 0 Sparks, DFlash2, fixed 425/425 W (2026-11-12): FP8 vs original C1/C16/8K prefill 171.12/741.93/7837.57 to 196.74/772.56/7741.92 tok/s (+15.0%/+4.1%/-1.2%); pool 1628160 to 2097152. K6 is non-inferior on both measured Standard-v2 paths, with +22.8% C1 and -12.5% 8K prefill vs original. Muse K4 MC `fp8-compact` is measured below (+15.2% C1, C16 flat, -5.1% 8K prefill; larger pool). GLM/Step whole-core FP8, 2 RTX PRO 6000 + 6 Sparks, 425 W/card (2026-11-16): GLM C1/C16/8K 49.04/180.68/3392 to 53.52/191.81/3122 tok/s (+9.1%/+6.2%/-8.0%), onboarded layers 5 to 8; Step 59.87/216.73/2740 to 68.98/266.95/2562 (+15.2%/+23.2%/-6.5%), onboarded layers 13 to 19. Both keep the 2,097,152-token pool; one warmed launch per arm, strict lossless 10/10. Step FP8 replay matches all 106 saved original prompt hashes. No thermal slowdown; prefill power-cap windows differ. Requires v52e exports. Other checkpoints unmeasured. | K8/K6 and FP8 quantize target weights. FP8 uses E4M3 + FP32 128x128 block scales. GLM generated Standard decode/prefill KL 0.03686/0.03100 to 0.04530/0.03880, top-1 94.544%/95.177% to 94.091%/94.775%; Step KL 0.01337/0.01164 to 0.01597/0.01429, top-1 96.069%/96.008% to 95.166%/95.811%. Absolute family bars pass; no paired non-inferiority claim. Original stays default. Matched Standard-v2 generated positions: FP8-minus-original KL to official BF16 ancestry +0.00289 decode/+0.00325 prefill, top-1 -0.013/-0.526 percentage points; absolute ancestry gates pass, paired decode non-inferiority passes but prefill fails (0.005 margin). Those figures ran W8A8 prefill over every group; W8A16 prefill (`FP8_PREFILL=off`) also fails prefill non-inferiority (top-1 loss 0.00263, upper95 0.00614). FP8 stays opt-in. Unsupported executor roles refuse by tensor name, never skip. Plain checkpoints refuse every compact selector, including the former plain Flash FP8 path. |
| `KV_CACHE` | Muse | `KV_CACHE=fp8` (default `bf16`) | Muse K4 MC original, dev `5e28a1d30`, RTX PRO 6000 (2026-11-16): 1 RTX + 4 Sparks, speculation off, pool 510,272 to 1,056,768 tokens (2.071x), served context 510,272 to 1,048,576. Separate quiet 2 RTX + 6 Sparks pair with default NVIDIA DFlash2: C1 70.81 to 65.08 emitted tok/s (0.919x), C16 253.03 to 257.81 (1.019x), pool 2,009,216 to 2,097,152 and onboarded layers 0 to 9. One warmed launch per arm; no power-cap or thermal slowdown in timed windows. | Lossy native E4M3 target KV with per-token/head FP32 scales. Standard max generated decode: KL 0.00968 to 0.01689 (+0.00721), top-1 97.149% to 96.594% (-0.555 pp); prefill KL +0.00625, top-1 -0.716 pp. Absolute family bars pass, but paired quality and C1 default bars fail: remains opt-in. BF16 stays default; drafter storage is independent. |
| `QWEN_FP8_HEAD` | Qwen4 | `QWEN_FP8_HEAD=on` (unset = BF16) | Converts the shared target/MTP head to one E4M3 backing. Historical combined measurement is in the decode row; standalone effect unmeasured. | Changes target logits; the draft reads the same head, not a separate draft-only FP8 copy. |
| `MIMO_FP8_HEAD`, `MIMO_FP8_O_PROJ` | MiMo | Set either to `on` (unset/auto = checkpoint) | Explicitly converts a BF16 target head or O projection; native FP8 stays FP8. Unmeasured. | Target-weight quantization, independent of drafter precision. |
| `GLM5_FLASH_FP8_HEAD` | GLM Flash | `GLM5_FLASH_FP8_HEAD=on` (unset/auto = checkpoint BF16 on TP1 and TP2) | Converts the shared target/draft head to E4M3 on either topology; unmeasured. `packed` is a separate exact BF16 option on one GPU. | Changes target logits; metadata-selected block-FP8 and native trellis/EXLR8 heads ignore this setting; no second copy is created. |
| `GLM5_FLASH_KDA_FP8` | GLM Flash | `GLM5_FLASH_KDA_FP8=row128` or `channel` (unset/auto = off) | Converts BF16 KDA projections outside a metadata-selected native stage to one resident E4M3 copy; MC-selected q/k/v/o already use block FP8 and need no extra key. Prior 1 RTX + 2 Sparks, with FP8 head and drafter: C4 code 113.8 to 131.8 tok/s; MC original unmeasured. | Prior top-1 89.1% to 85.7%; six-row verify/decode KL 0.0056 to 0.0121 nat. Does not change KV dtype. |
| `FP8_PREFILL` | GLM 5.3 / ML4 | Explicit projection groups or `off` (default `auto` preserves the previous native routes) | Unmeasured. GLM 5.3 can select `mla`, `index`, `ffn`, and `experts`; ML4 can select `index` and `experts`. | Selectable W8A8 prefill routes quantize activations to FP8 instead of BF16; no weight conversion. Calibrated static W8A8 and fixed native routes are unchanged. |
| `FP8_PREFILL` (KDA groups) | GLM Flash | Add `kda-in` / `kda-o` to the default `mla,ffn`; needs FP8 KDA weights (`CORE_VIEW=fp8-compact` or `GLM5_FLASH_KDA_FP8`) | Adds W8A8 KDA prefill passes over already-FP8 KDA weights. Unmeasured. The opposite direction, `FP8_PREFILL=off` (W8A16 everywhere), costs prefill: 2× RTX PRO 6000 / 0 Sparks, DFlash2, one launch per arm, power-cap 4–7 s, no thermal slowdown (2026-11-13, #1914): K3.25-Core-FP8 8K prefill 8238→7847 tok/s (−4.8%; −2.9% in a second session), quick KL 0.0360→0.0348, top-1 95.57%→95.41%; K4-MC `fp8-compact` (all groups explicitly selected) 7957→6857 (−13.8%, host page reclaim during the run), KL 0.0226→0.0215, top-1 96.62%→96.20%. | FP8 activation quantization at prefill; never converts weights. |
| `SPARK_PREFILL_REPLY_FORMAT` | V4.1, V4, GLM 5.3 / ML4, GLM Flash, MiMo, Qwen4, Kimi K3, Step-5, Muse | `SPARK_PREFILL_REPLY_FORMAT=fp8` or `nvfp4` (unset = BF16; release builds only) | Spark expert partials return compressed instead of BF16: fewer exchange bytes per prefill wave. Unmeasured. | Reply values at FP8/NVFP4 precision instead of BF16. |
| `SPARK_DECODE_REPLY_FORMAT` | V4.1, V4, GLM 5.3 / ML4, GLM Flash, MiMo, Qwen4, Kimi K3, Step-5, Muse | `SPARK_DECODE_REPLY_FORMAT=fp8` or `nvfp4` (unset = BF16; release builds only) | Spark expert partials return compressed instead of BF16: fewer exchange bytes per decode wave. Unmeasured. | Reply values at FP8/NVFP4 precision instead of BF16. |
| `ENGRAM_WARM` | V4.1 | `ENGRAM_WARM=on` (unset = off) | After readiness, reads both Engram tables into page cache on a background thread so per-token table reads hit RAM on stores with slow random reads. Unmeasured. | Host page cache equal to the table bytes (about 203 GB for the official checkpoint), reclaimable; skipped when free memory is below the tables plus 16 GiB. |

Stored `k8-compact` and `k6-compact` views are offered by
`wrldsuksgo2mars/Kimi-K3.1-Pro-EXL3-K2-MC`,
`wrldsuksgo2mars/Muse-Spark-Open-EXL3-K4-MC`,
`wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K4-MC`,
`wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3.25-MC`,
`wrldsuksgo2mars/GLM-5.3-EXL3-K4-MC`,
`wrldsuksgo2mars/Inkling-2-Small-EXL3-K4-MC`,
`wrldsuksgo2mars/Hy4-flash-EXL3-K4-MC` and
`wrldsuksgo2mars/Step-5-Preview-EXL3-K4-MC`; `cuteafd-mc.json` declares the
views actually present at the selected revision. Step-5 and GLM 5.3 also execute
header-native K6/K8 target cores (including vocabulary heads). Repaired MC
revisions restore native BF16 MTP core matrices in `original`; stored compact
views retain their existing K4 MTP core. MTP routed experts are unchanged.
GLM 5.3 and GLM Flash use external drafters: their unused native MTP core and
experts are neither loaded nor charged, and are not converted by `fp8-compact`.
Step-5 executes its selected MTP core in one separately charged packed arena.
ML4 stays K6-only for quantized core storage;
its FP8 original has no quantized core. Plain checkpoints support only
`original`, including checkpoints that previously accepted Flash `fp8-compact`.
GLM 5.3 MC (`2e42b53e61f0f59f664c915fe2424ce9c9fa5f8e`) and Step-5 MC
(`aedda90d25bd02bb6520bcaf88e06e6c9bc932c4`) support `original`,
`k8-compact` and `k6-compact`. Their release cards select `original`;
compact-view performance and Standard fidelity are pending #1906.
With the v52e coordinator exports, GLM and Step-5 execute whole-core
`fp8-compact` as an explicit quality-affecting opt-in. Max-card measurements
and generated-row quality costs are in the optional-levers table; absolute
quality and strict lossless speculation pass, but original stays the default.
GLM leaves unselected KvB/indexer and unused native MTP unchanged. Step-5
includes selected target/MTP QKV, FFN, EH and private heads; routed experts
are never converted. Both use the shared block-FP8 head executor.
Muse MC (`cc5dc2537083177e3f02ea0b2d9dfc179d40425c`) likewise offers
`original`, `k8-compact` and `k6-compact`; separate
`muse-spark-exl3-k4-mc-{min,max}.config` examples select `CORE_VIEW=original`.
The standalone `muse-spark-exl3-{min,max}.config` examples keep serving Core-K6;
only their regular benchmark cards retire. Muse also executes the full
`fp8-compact` view: selected q/k/v/o, optional gate, pre/post-expert projections
and the vocabulary head. Enable it explicitly with `--set CORE_VIEW=fp8-compact`;
the examples and default stay `original`. The head stores 202112 output rows,
trims to the logical 202048 vocabulary, and shares that same head with the drafter.

Muse K4 MC original versus `fp8-compact`, 2026-11-15: same frozen WIP
`4e5e0dff0`, 2 RTX PRO 6000 + 6 Sparks, DFlash2, automatic pool/onboarding,
one warmed launch per arm, no host-reclaim flags. This is the W8A16 prefill
route over FP8 weights; the shared FP8-prefill resolver in #2119 is not in this
measurement. Its A8-over-FP8 route is not measured here.

| Metric | Original | FP8 compact | Change |
| --- | ---: | ---: | ---: |
| C1 code decode, tok/s | 61.77 | 71.18 | +15.2% |
| C16 aggregate code decode, tok/s | 245.64 | 244.75 | -0.4% |
| Post-load C1, tok/s | 76.44 | 75.57 | -1.1% |
| Exact 8192-token prefill, tok/s | 3371.17 | 3199.81 | -5.1% |
| Baseline 8K prefill, tok/s | 3327.74 | 3075.49 | -7.6% |
| KV pool tokens | 2009152 | 2097152 | +88000 |
| Resident target weights, GiB (both GPUs) | 40.345 | 26.214 | -14.131 |
| Generated-row KL to official BF16 root | 0.007085 | 0.008094 | +0.001009 |
| Generated-row top-1 | 97.60% | 97.33% | -0.27 pp |

Both 1832-generated-row quick quality checks pass Mid (KL <= 0.05, top-1 >=
93%). Prompt-end and turn-end restores plus cold recompute are byte-identical;
short fresh continuation coverage is not established. Speculation emits the
same 128 greedy tokens with drafts on/off. Neither arm thermally throttled.
Baseline power-cap slowdown was 0/0 s on GPU0/GPU1 for original and 7.0/6.0 s
for FP8; busy clocks were 2723/2832 versus 2720/2830 MHz, maximum temperatures
65/70 versus 66/73 C, and minimum T.Limit headroom 30/24 versus 30/22 C.
The exact-8192 prefill point had 0/0 versus 1/1 s power-cap slowdown, with
busy clocks 2647/2726 versus 2693/2752 MHz and identical 2752-row unit shapes.
The prefill cost is retained explicitly; no A8 speedup is claimed.

MC `fp8-compact` converts the original variant's `core_tensors`, except unused GLM/GLM Flash native MTP stages. Step-5's qualified native FP8 MTP executor includes its selected non-routed MTP core; published
Flash MCs include KDA and the target head. Explicit KDA/head conversion keys apply
only beyond that metadata-selected scope. No precision lever changes KV defaults.

## How options are set

A launch reads one plain `KEY=VALUE` config file (`cuteafd.config` by default):

```sh
./run.sh --config examples/configs/glm53-exl3-k4-mc-min.config --wip SLOT
./run.sh --config my.config --set CONCURRENCY=16 --set SPARK_COUNT=4
./run.sh --config my.config --dry-run --total-capacity   # validate and print the plan
```

- Keys are `[A-Z_0-9]+`. Values are unquoted; whitespace needs quotes.
- An empty value means "unset": the default applies.
- An unknown name fails at parse (`unknown configuration key`, exit 2).
- A known name the selected family's launch path never reads fails as
  `unused configuration key: KEY` (exit 2). That is the refusal mechanism every
  `✗` in the family table below means.
- `--set KEY=VALUE` appends an override to a copy of the file, so it gets the
  same validation. `SPARK_COUNT`, `SPARK_HOSTS`, `WIP_INSTANCE`, `WIP_ROOT` and
  `API_KEY_FILE` may also come from the environment.
- `CONCURRENCY` unset is a boot-time memory derivation; all C-dependent sizes
  follow it. Set it only to pin a value.

### Kimi Request Reasoning

Kimi thinking-off is simulated with an empty, preclosed native XTML `think`
channel followed by an open `response` channel. Assistant history likewise keeps
an empty `think` channel and drops its reasoning text when thinking is disabled.
The API advertises `supports_thinking_off: true`: generated answer bytes appear
as content, with no reasoning output. This preserves the trained channel shape;
it does not use the checkpoint encoder's direct-response-only branch. Explicit
`thinking: {"type":"disabled"}` and the supported `enable_thinking: false` aliases
select this form. Thinking-on rendering and effort selection are unchanged.

### Inkling Request Reasoning

The Inkling chat adapter reads the checkpoint's own `chat_template.jinja` and
`tokenizer_config.json`. Runtime serving qualification is tracked in #1930;
these request values do not imply that a Planned checkpoint has a serving executor.

| Request Field | Default | Values | Notes |
| --- | --- | --- | --- |
| `reasoning_effort` (Inkling) | `high` (= 0.9) | `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`, or a JSON number in `[0, 0.99]` | Native template values are 0, 0.1, 0.2, 0.7, 0.9, 0.99 and 0.99, respectively. A top-level value takes precedence over `chat_template_kwargs.reasoning_effort`; explicit thinking-off resolves to `none`. Reasoning and tool calls use native message channels. Only `<|content_model_end_sampling|>` (token 200006) ends generation; `<|end_message|>` closes a channel and is not an EOS stop. |

### PE Vision Qualification

Inkling and Step-5 use the shared bundled PE tower runtime and image admission.
Step-5 defaults to `VISION=auto`: joint admission selects a Spark-first packed
owner with RTX fallback. `VISION=off` explicitly selects text-only operation.
Inkling remains text-only by default while normal image answers are qualified (#1930).
Explicit `VISION=spark[:rank]` or `VISION=rtx[:gpu]` selects the jointly admitted packed owner;
`VISION_REPLICAS=1..8` and `MEDIA_CACHE_BYTES` use the shared media options.
Each tower has one weight arena and one scratch backing per replica; tensor
views do not add another charge. Step-5 images retain ordered local crops and
the global view with native separators; the 896x812 processor case expands to
493 feature rows plus 11 native markers (504 inserted tokens; 506 tokens in
the text/image/text boundary fixture). Audio remains unsupported.

### Inkling Serving

Inkling exposes `cuteafd serve --family inkling2` and `./run.sh --wip SLOT`
for serving on one or two RTX GPUs and 2-8 Spark expert endpoints. Standard
fidelity and Release smoke qualify the official mixed NVFP4/MXFP8 and EXL3 K4
MC original checkpoints on 1 RTX + 4 Sparks, and native MXFP8 on 1 RTX + 6
Sparks. Mixed TP4 owns and stores exactly 576 intermediate columns per rank;
no checkpoint weight padding is required. Target core, RTX-onboarded experts
and every Spark rank load in one required SparkNest push transaction.

Planner reports mark admitted configurations Serving, independent of checkpoint
repository, revision, MC view or fidelity measurement. Missing formats or native
programs still refuse with their structural cause. `RTX_GPUS=2` splits target
attention and the bundled drafter across both GPUs for every core view; the
lead owns the vocabulary head and residual stream. A matching WIP build must include the
Inkling padded-transpose load export and native expert packages.

Inkling K4 MC uses `CORE_VIEW=original|k8-compact|k6-compact|fp8-compact`.
K8 and K6 execute the stored trellis attention projections, vocabulary head
and selected recurrent drafter matrices. FP8 converts those same selected
roles from original BF16 to E4M3 with FP32 block-128 scales. Embeddings,
unselected projections, routed experts and the bundled vision tower retain
checkpoint precision. Gate and performance measurements for the compact
views are pending. The cold golden accepts the same view with `--core-view`.

`ATTENTION_PLACEMENT=auto|heads` and GPU embedding/device selection are supported;
Explicit `layers` and `context` are not implemented.
Set `AUDIO=off`; bundled vision follows `VISION=auto` and normal encoder admission. `SPECULATOR=off|dspark` defaults to `dspark`
(unset/`auto` selects the bundled drafter); explicit `off` wins and passes
`--dspark=false`. Direct serving also defaults on;
`dspark` selects the bundled recurrent-hidden block drafter (seven proposals
per eight-token block), with `DRAFT_KV_CACHE=fp8|bf16` (default `fp8`).
`SPECULATOR_FP8=auto|off` leaves the drafter precision to the selected core
view; it is not a separate conversion switch. `on` refuses.
Drafter weights join the target's required push transaction in their own
admitted arena. MC original EXL3 K4 on 1 RTX + 4 Sparks passes strict cold and
warm lossless with FP8 draft KV (10/10 each, exact ids, row hashes and emitted
bytes, complete prompt-end restores). BF16 draft KV also passes strict cold and warm 10/10; emitted-speed
measurement remains pending (#2320); no speed figure is claimed here. Inkling accepts the shared
`CONCURRENCY`, `DECODE_ROWS=auto|N` (`auto` = 64, up to 128 on 1 or 2 GPUs), `POOL_TOKENS=auto|N`, `MAX_CONTEXT_TOKENS`,
`PREFIX_CACHE_ENTRIES`, `HOST_CACHE_BYTES`, KV reservation and loop-abort keys,
plus `RTX_EXPERT_LAYERS=auto|all|N`, `PREFILL_ROWS` (a multiple of 64 up to
4096), `PREFILL_WAVES` (1-3), `GPU_GRAPH_RESERVE`, its extra/cap keys and
`GRAPH_TRACE` and `KV_CACHE=bf16|fp8`. `PREFILL_PACKING=unrestricted` is the
packing Inkling always runs; `exact` refuses at startup (no attested
coordinator packed-route cap). Global KV defaults to FP8; BF16 remains an explicit option;
FP8 stores native E4M3 payloads with per-token/head FP32 K/V scales. Local
rings stay BF16, convolution/residual state stays F32, and prefix restores
preserve captured state exactly. Report the admitted KV
pool rather than assuming the default target fits these owners.

### Inkling 2 Mini Serving

Inkling 2 Mini (`inkling2_mini`, `inkling2_loop`) exposes `cuteafd serve --family inkling2_mini`
(`serve-inkling2-mini`), `cuteafd golden --family inkling2_mini` (`inkling2-mini-golden`), and
`./run.sh --config examples/configs/inkling2-mini-1x-rtx.config --wip SLOT` on **1× RTX (0 Sparks, TP1)**.
All 24 physical routed-expert layers stay resident on GPU 0 (native NVFP4 W4A4 layers `0..20` +
native MXFP8 W8A8 layers `20..24`), executed across 48 visits with 48 distinct KV + short-conv slots
(8 global paged slots defaulting to FP8 + 40 SWA-1024 rings). `SPECULATOR=off|dspark` defaults to
`dspark` (bundled 3-layer recurrent Markov drafter, `DRAFT_KV_CACHE=fp8|bf16`), `VISION=auto|rtx|off`
admits the bundled 23-layer PE-Lang tower in its own arena, `AUDIO=off`, `DECODE_GRAPHS=on|off`
controls `#2697` soft-cap per-visit exact-row CUDA graphs, and `PREFILL_PACKING=unrestricted`.

### Hy4-flash Serving

Hy4-flash (`hy4_flash`, `hy_v4_flash`) exposes `cuteafd serve --family hy4_flash`
(`serve-hy4f`) and `./run.sh --wip SLOT` on one or two RTX GPUs with `0..=6`
Spark expert peers (#2478). `DECODE_ROWS=auto` resolves to 64; plan and serve
refuse larger budgets because the Hy4-flash split-K decode exports cover at
most 64 rows on either topology. Scratch is charged for the resolved budget.
Routed LatentMoE layers use MUL1 EXL3 (`hy4f:exl3-k45-mul1`).
`EXPERT_USAGE=on` (default) counts the canonical rank-0 routes, including Spark
routes but excluding graph startup and TP2 replicas; `off` omits its owners.
`CONSOLE_TEXT=off` disables console token text even for unlocked viewers.
Hy4-flash refuses `L2_PREFETCH` by name; leave it unset.
Engram tables are FP8-only (`engram-fp8-layer-*-part-*.safetensors`, 100-byte strided
E4M3 + F32 scale rows declared in `cuteafd-mc.json`, with no BF16 fallback): checkpoints
lacking FP8 Engram shards are refused by name in plan, serve and golden. All four Engram
layers (`[1, 11, 21, 31]`) read ahead from request admission (prefill) and immediately
when candidate tokens exist (decode/verify) onto the shared `mapped_table::lookahead`
planner and per-layer bounded `RowCache`s, with background gather and FP8→BF16 decode
overlapped with earlier GPU layers.

`CORE_VIEW=original|k8-compact|k6-compact|fp8-compact` selects the dense-core
view (`original` default; `k8-compact` and `k6-compact` use stored trellis cores;
`fp8-compact` converts the original BF16 core tensors at load with block-128
FP8). `KV_CACHE=fp8|bf16` defaults to `fp8` for absorbed MLA KV and keeps the
FP8 indexer key cache default. `ATTENTION_PLACEMENT=auto` selects whole layers
on one RTX GPU and `context` (ContextSplit, parity-owned DSA KV/index pages with
no DSA KV replication and head-split SWA-512) on two RTX GPUs; explicit
`ATTENTION_PLACEMENT=heads` opts into replicated-KV head-split (`layers`
refuses in `serve-hy4f`). `EMBEDDING=gpu` is required
(`host` refuses). `VISION=auto` (default) serves the checkpoint's bundled
27-layer ViT (`vit.*`, 0.87 GiB packed BF16 arena plus admitted scratch) from
the jointly admitted owner: an RTX GPU, or `spark[:rank]` replicas when every
expert layer is onboarded; `VISION=rtx[:gpu]|spark[:rank]|off`,
`VISION_REPLICAS` and `MEDIA_CACHE_BYTES` use the shared media options.
Images follow Tencent's serving processor (SGLang `smart_resize`, 32-pixel
cells, 64x64 to 2048x2048 pixels, bicubic, mean/std 0.5): a `h x w` patch grid
becomes `h/2 * (w/2 + 1)` image rows, the checkpoint's `image_newline` closing
each merged row, with interleaved 3D M-RoPE positions. Up to eight images
encode in one batched tower pass. `AUDIO` accepts `auto` or `off` and starts
none. `SPECULATOR=auto` (default) uses the checkpoint's bundled five-stage
DFly2 on Spark-free layouts, including two-RTX `context` (default) and explicit
`heads`; Spark layouts retain their drafter-off default. `SPECULATOR=dfly2`
explicitly enables it. It reuses committed target MLA KV
and compressed-indexer recall; a fully restored prefix takes one target-only
round before proposing. Proposals batch across requests; rank-local draft and
verify sections graph exact live rows. Recall gathering, metadata uploads and
TP2 head handoff remain eager. `SPECULATION_TRACE` uses the shared speculation trace.
Stored K8/K6 views load the bundled K8 MUL1 linears, except KV-B, which follows
the target's absorbed-BF16 layout. `SPECULATOR=off` admits no drafter storage. Hy4-flash uses the shared `DECODE_ROWS` budget
(up to 64 target rows, including anchors; `auto` is 64).
The minimum layout is one RTX PRO 6000 with no Sparks. In the original-core test, with the
bundled drafter enabled, its measured pool is 538,368 tokens (754,944 with the
drafter off, same 512 MiB graph reserve and KV cap). Both clear the 262,144-token
context floor. This layout intentionally falls short of the 2,097,152-token soft
placement target even without the drafter; it does not add a Spark to reach it.
`RTX_EXPERT_LAYERS=auto|all|N`,
`PREFILL_ROWS`, `PREFILL_WAVES` (1–3), `PREFILL_PACKING=unrestricted`,
`POOL_TOKENS`, `MAX_CONTEXT_TOKENS`, `CONCURRENCY`, `PREFIX_CACHE_ENTRIES`,
`HOST_CACHE_BYTES`, `GPU_GRAPH_RESERVE` and `GRAPH_TRACE` follow the shared
contracts. All Hy4-flash graph sites are rank-local and capture lazily. The
short/32K C1 census sets initial donor pressure and the charged hard cap to
184/192 MiB on 1 RTX, 196/204 and 194/202 MiB by rank on 2 RTX, and
108/112 MiB on 1 RTX with Sparks (measured with one Spark). `GPU_GRAPH_RESERVE`
overrides the hard cap; extra/cap overrides are not supported for these
working-set layouts. Unmeasured 2 RTX + Spark layouts default to zero reserve
and can opt into an explicit reserve. Donors are used only under pressure by default. `PREFILL_PACKING=exact` refuses at startup: Hy4
attests no coordinator packed-route cap yet. Exact prefix restores preserve SWA ring banks, DSA
KV/indexer pages and the Engram rolling token tail byte-identically.

### MiniMax Serving

MiniMax M3.1 Flash supports public `cuteafd serve --family minimax_m3` and
`./run.sh` on one RTX or a two-RTX FP32 split, qualified by the 2026-11-14
trace-free serving cards (#1856).
Use a build containing MiniMax native exports: MXFP8 dense math, NVFP4 W4A4
experts, FP8 MSA KV/index, GPU embedding and device token selection.
MSA K/V use per-token, per-KV-head FP32 `amax/448` scales. Each 64-token
page keeps aligned E4M3 payload rows followed by a 16-byte-aligned scale plane
in the same backing; admission and exact prefix snapshots include both. K or V
pages occupy 33,792 bytes for whole-head ownership and 17,408 bytes per Heads
rank. There is no unscaled FP8 KV mode.
On one RTX, `ATTENTION_PLACEMENT=auto|heads` selects whole ownership. On two RTX,
`auto|heads` selects native head splitting with half KV and replicated index.
Explicit `layers` keeps contiguous attention and FP32 split FFNs. Boot-C16 split
router, FFN and fold/publication sections replay through the shared graph cache;
attention and peer receive waits stay eager. Context placement remains unsupported.
`SPECULATOR=dspark` is the default (unset/`auto` resolves to DSpark) on one RTX,
two-RTX Layers and two-RTX Heads. The checkpoint's bundled native mixed BF16/MXFP8
drafter uses shared speculation and exact window snapshots; `SPECULATOR=off`
disables it. Direct `cuteafd serve --family minimax_m3` likewise defaults on;
`--dspark=false` disables it. Plan and serve admit the same drafter owners.
Windowed block dSpark drafters (MiniMax and GLM Flash) ingest only original tap
chunks overlapping the final context window, retaining a straddling chunk whole
to preserve projection shapes and bytes. Each planned intermediate prefix mark
gets its own window; MiniMax withholds an unexpected partial-prefill snapshot
until its window is complete. Full-context paged drafters are unchanged.
`DRAFT_LENGTH` fixes the verified width within the seven-token native block;
`SPECULATION_TRACE` records each draft round. MiniMax drafts on the shared
`history` policy like every other family (see [Speculation](#speculation)); its
calibrated confidence head scores each drafted token. The example config's
explicit off remains an override until its separate line change joins the next
placement-input publication batch.
The bundled still tower serves with `VISION=auto` (`VISION_REPLICAS` and
`MEDIA_CACHE_BYTES` apply); `AUDIO` accepts `auto` or `off` and starts none.
`CORE_VIEW` selects a declared core view as for the other MC families.

Shared launch keys keep their usual meanings. MiniMax accepts
`RTX_EXPERT_LAYERS=auto|all|N`, `PREFILL_ROWS`, `PREFILL_WAVES`,
`PREFILL_PACKING=unrestricted|exact`, `PREFIX_CACHE_ENTRIES`, `HOST_CACHE_BYTES`
(default launcher `auto`), `POOL_TOKENS`, `MAX_CONTEXT_TOKENS`, `CONCURRENCY`,
`EXPERT_USAGE=on|off` (default on), `GPU_GRAPH_RESERVE` and its extra/cap keys,
and `GRAPH_TRACE`. Only BF16 Spark replies are supported; compressed reply
formats refuse before loading. Native checkpoint API encoding is mandatory.
The traced one-RTX boot-C16 graph base is 41,943,040 bytes plus the standard
2,097,152-byte tail; shared bounded runtime extra gives 88,080,384 bytes total
by default. C1/C16 decode on that bank records no steady-state captures or
evictions. The two-RTX boot-C16 split has per-rank measured bases 65,011,712 /
44,040,192 bytes (62 / 42 MiB), each plus one 2,097,152-byte tail. The unchanged
shared extra rule charges 134,217,728 / 92,274,688 bytes by default (128 / 88 MiB).
The bounded rows 1/4/16 bootstrap records no steady C1/C16 captures or evictions;
only private arithmetic/peer scratch is padded, never KV or request rows.
Untraced rank/concurrency shapes keep an unpinned bootstrap with no borrowed
measured base.
Heads boot-C16 rows1/4/16 measured82/64MiB graph owners per rank (711/540
sections), plus one2MiB allocation tail each. The shared extra rule charges
168/132MiB by default; Layers stays128/88MiB. Attention and exchange stay eager.
Native512-row graph/eager logits and repeats are byte-identical. Quick generated
fidelity uses the same Mid bars; layout-dependent reduction order can change
quantization thresholds. Two-RTX auto selects Heads (TJ, 2026-11-15); one RTX
retains whole-layer execution, represented by the single-rank `heads` API value.
The first short run measured Layers versus Heads-graphs C1/C16/8K at
33.98/157.26/2215 versus 37.57/155.70/2336 tok/s, but host page reclaim
contaminated the timing and the external cold-prefill probe aborted (#2132).
TJ selected Heads as the two-RTX default after reviewing both orders; the
reclaim flags remain recorded, not erased. Quick generated KL was
0.02346 versus 0.02469 and top-1 95.23% versus 94.94%; both pass the Mid bars.

| Metric | Layers | Heads | Heads / Layers |
| --- | ---: | ---: | ---: |
| C1 code decode (tok/s) | 34.015 | 37.820 | 1.112 |
| 8K prefill (tok/s) | 2218.95 | 2336.01 | 1.053 |
| C16 confirm, Heads first (tok/s) | 235.400 | 284.182 | 1.207 |
| C16 earlier, reclaim-contaminated (tok/s) | 226.723 | 218.695 | 0.965 |

Both orders started at CPU/memory PSI 0/0; the confirm also started with zero
page scans. Heads C16 still recorded host reclaim (13,693 pages/s confirm,
31,090 earlier); these are observations with a user-approved default, not a
reclaim-clean speed qualification. Power-cap and thermal slowdown were zero;
max temperature 63 C, minimum T.Limit headroom 32 C. Named host allocations
differ only by 120 KiB of expert-usage telemetry; both keep the same 32 GiB
prefix host cache and identical pinned staging/RDMA rings. Replicated KV/index
buffers live on-device. Startup non-file RSS differs by about 2.23 MiB;
whole-host anonymous growth cannot be attributed to this process from these logs.

`./run.sh` options:

| Option | Meaning |
| --- | --- |
| `--config FILE` | Config file to load (default `cuteafd.config`). |
| `--set KEY=VALUE` | One override, repeatable. |
| `--family ID` | Force a family instead of reading the snapshot's `config.json`. |
| `--embedding-placement host\|gpu` | Same as `EMBEDDING=host\|gpu`. |
| `--wip SLOT` | Serve a `./wip.sh` slot's artifacts. |
| `--restart [--all]` | Replace running containers; `--all` also stops other instances. |
| `--dry-run [--total-capacity]` | Validate, plan and print without touching services. |
| `--rotate-console-secret` | Re-provision the console unlock secret. |

`./stop.sh [--config FILE] [--all]` stops the containers for that config
(`--all` also removes release expert workers on the configured hosts).

The `cuteafd` binary: `plan MODEL`, `serve` (family from `--snapshot` or
`--family`), `golden`, `expertd`, `gateway`, `usage`, `doctor`, `fabric`,
`bench [run|fidelity|publish|smoke|export|cancel|panels]`, `bench-rdma`,
`bench-rdma-ring`, `transport-capabilities`. `cuteafd plan --layout` reports fit
before any kernel work; `cuteafd bench --url URL --profile NAME` runs a profile
on a launched server. Family serve subcommands (`serve-native`, `serve-dsv4`,
`serve-glm`, `serve-glmf`, `serve-mimo`, `serve-qwen4`, `serve-qwen35`,
`serve-kimi`, `serve-step5`, `serve-muse`, `serve-inkling2`, `serve-inkling2-mini`, `serve-hy4f`) take the flags the launcher maps the
keys below onto, and `cuteafd serve --family ID --help` lists a family's serve
flags. Users set keys, not these flags or the `CUTEAFD_*` environment variables
the launcher derives from them.

`cuteafd golden --family minimax_m3` (hidden alias `minimax-golden`) is a
native full-model qualification path. Public one-RTX serving dispatches to the
same `serve-minimax` command used by the qualified card.
Explicit two-RTX heads serving is available with half KV and replicated index
storage; `auto` stays contiguous whole attention with neuron-split FFNs on two RTX.
The two-RTX layers recipe defaults to qualified FP32 split partials. The launcher
fixes `--split-wire fp32` and the serve flag defaults to `fp32`; there is no
user-facing wire configuration key. BF16 remains an explicit golden diagnostic
(`--wire bf16`), not a serving performance lever. One-RTX arithmetic is unchanged. The fixed qualification rule prefers BF16 only when its NLL differs from
FP32 by at most 0.002 nats and their top-1 agreement is at least 99.5%. Measured
BF16/FP32 generated-row NLL delta was 0.00294317 and agreement 95.0867%, so FP32
is selected. An explicitly unselected internal wire still refuses with
"MiniMax split FFN wire precision not qualified/selected". Each rank
normalizes the broadcast FP32 residual locally; the FP32 meeting preserves the
BF16 whole-FFN boundary before the replicated residual add. GPU0 owns the sole
Spark endpoint and router; onboarded TP2 routes reach GPU1 as exact IDs/weights
on the split exchange control sequence, without GPU1 router weights or logits.
Explicit two-RTX `heads` is supported with native FP32 O/FFN partial reductions
and eager peer exchange. Decode Q/K/V/index projections, O projections, router
and half-FFNs use the shared graph cache; context-sensitive attention and peer
publish/wait/merge stay eager. `GPU_GRAPH_RESERVE=0` selects fully eager decode.
`auto` selects `heads` on two RTX; explicit `layers` remains available.
Golden requires `--nll --golden DATASET --save-logits NEW_DIR`, a checkpoint snapshot,
native library and program manifest, and four real Spark `--peers` endpoints.
`--window ID[,ID...]` selects unique sealed quick-subset windows. `--prefill-rows` defaults
to 1024 (page-aligned 256..4096); `--prefill-waves` defaults to 3 (1..4).
`--check-headers` validates the immutable reference and checkpoint/export geometry
without CUDA or workers. Scores use the shared top1024-plus-tail estimator.
`--split-device ID` selects a distinct physical peer for the diagnostic serving
split path, with `--attention-placement auto|layers|heads`.
The peer diagnostic requires explicit `--wire bf16|fp32`; omitting it refuses,
and it does not select a serving default or qualify that precision. `--decode`
uses real one-row serving decode after the prompt instead of scoring all rows
through serving prefill. Without `--split-device`, the existing one-GPU Whole
cold golden is unchanged. Existing `--save-logits` probes contain normalized
log-probabilities; optional MiniMax-only `--raw-logits` additionally writes each
window's scored raw FP32 rows to `WINDOW.raw-logits.bin` for absolute-logit
comparison (positions follow the immutable reference's scored-row order).
Split prefill/decode currently serialize on each
rank's one compute/HopLink stream; slot buffers remain independent.
MiniMax's explicit `minimax-generated-aggregate` family scoring policy (#1779,
TJ 2026-11-13) qualifies the aggregate generated rows at the expert-bpw tier:
the official NVFP4 checkpoint uses mid-tier mean KL <=0.05 and top-1 >=0.93,
without the reference-root strict-tier override. Context rows, every window
(including measured sensitivity-floor windows such as d00), confident top-1
and top-3 are reported as informational, not independent vetoes. Missing or
nonfinite rows and invalid precision/provenance still fail. Report publication
keeps this aggregate gated even for native W4A4, rather than applying the general
NVFP4 informational override. All other families retain their existing
reference-root tier, calibrated tripwires, window floor and W4A4 report policy.
Golden, CLI and panel receipts expose the policy, generated/context summaries
and per-window numbers. CPU-only `cuteafd bench fidelity rescore-golden INPUT
--out NEW_JSON` recomputes the verdict from retained native golden row records
and records the input SHA256; it never reruns the model or overwrites evidence.
Diagnostic `--tap-layers 0,1,2,3` requires `--window` and unique layer IDs
0..59. It writes F32 input/post-attention/final residuals and BF16 projected
attention/local-MLP branches in `NEW_DIR/layer-taps/ID`, with zero-based input-row
suffixes and explicit diagnostic metadata. `--tap-residual-only` omits sublayer
branches. Both modes also export the final BF16 normalized hidden before the
head. Copies drain the issuing stream and reuse one admitted host buffer; taps
are off by default and do not waive scoring. The family reference CLI's
`--layers 0 1 ...` exports residuals, branches and scored final norm;
`--residual-only` omits branches without bypassing ordinary prefix qualification.

`cuteafd kimi-golden --rtx-expert-layers auto|all|N` defaults to `auto`, using
startup's real-device admission rather than forcing zero coordinator layers.
Golden's expert-layer hint must cover the planner-selected Spark layers under
`auto`; explicit `all` or `N` requires an exact list. Only the admitted list is
sent to workers and loaded. Golden receipts record the chosen coordinator count,
Spark layer IDs and inactive hinted/loaded counts.
`cuteafd kimi-golden --skip-routed-experts --nll` instead runs the router and
shared experts with zero routed contribution, no Spark peers, no expert catalog
or RTX onboarding. It accepts only a matching `cuteafd.kimi.skipped-routed/1`
reference generated by `python/reference/families/kimi_k3/golden.py
--skip-routed-experts`. The reference uses only the sealed eight-window **quick**
subset (legacy, a00, a04, a08, a20, c00, d00, e00); a20 scores beyond 8K history
and exercises sparse DSA selection. Full-model reference defaults are unchanged.
This all-layer dense/attention/DSA diagnostic is **not** a full-model golden.
The skipped-routed diagnostic is available only through the explicit golden
role; the flag is unavailable to serve and does not restrict ordinary serving.

`cuteafd glm-golden --nll --save-logits DIR` writes `tokens.bin` and all
prefill rows as F32 `logits.bin` for same-engine build or core-view comparisons.
The option is listed in CLI help so the sealed golden runner can validate it.

`cuteafd golden --family inkling2 --width-invariance 1,8 --tokens TOKENS.json
--prefill N --peers HOST:PORT,... --save-logits DIR` compares teacher-forced
serial decode with actual block verification after an identical admitted prefix.
This path loads the selected MC core view through serving startup and its
required push transaction with real Spark experts. `--rtx-expert-layers`,
`--pool-tokens`, `--max-sequences`, `--prefill-rows`, `--prefill-waves`, target
`--kv-cache`, `--draft-kv-cache` and shared prefix args select the served shape.
Widths are 1..8 total verify rows (anchor included), independently of the
64-row packed physical capacity. Exact prompt-end SWA/FIR marks are restored
before each width; width 1 repeats after the wide pass. The JSON receipt names
the first differing layer, stage and operand per row, including route ids,
weights and expert outputs; no tolerance or automatic numerical fix applies.
Add `--commit-invariance` to compare every accepted prefix (0..width) against
serial decode: live global KV rows, sliding-window KV, contiguous FIR bytes and
logical bookkeeping, followed by one more cycle's logits. Dead rejected KV
outside the committed horizon is excluded. The receipt names the first owner
and differing byte; this remains a diagnostic, not a tolerance.
Run through `scripts/bench/golden-run.py --family inkling2 --spark-hosts ...`.
Without `--width-invariance`, Inkling cold golden streams one native MXFP8 or MC EXL3 expert layer at a time.

`cuteafd step5-golden --golden TOKENS_DIR --nll --last-layer 91` accepts a
`tokens.bin`-only directory for same-engine core-view comparisons (2..4096 U32
tokens). No `--window` is needed; `--save-logits` retains the actual selected
view identity. This mode reports finite logits and engine NLL only, not reference
qualification, and refuses reference-dependent layer, final-norm, FP8 and prefix
checks. A present `meta.json` always selects the existing pinned-reference path
and its identity checks; missing or invalid reference fields never fall back.

`cuteafd qwen35-golden --golden DATASET --window ID` also accepts the pinned
published compact fidelity dataset for graph qualification when decode graphs
are enabled. It compares full decode logits byte-for-byte across graph-off,
graph-on warm-up and graph-on measured passes, requires measured replays and
identical NLL, and optionally runs `--verify-check` and `--prefix-check`. This
same-engine oracle does not replace the standard card's external top-1024
quality, strict lossless speculation or prefix-cache gates. Full-logits golden
inputs retain their existing reference checks; compact graph mode does not
support draft/MTP comparisons or `--save-logits`.

## Identity and instance

| Key | Default | Values | Notes |
| --- | --- | --- | --- |
| `MODEL_ID` | `deepseek-ai/DeepSeek-V4.1-Flash-1017` | HF repo id | Checkpoint to serve; starting configs always set it. |
| `MODEL_REVISION` | `1bae0e1088993b9b82de08f59de5be5a34e42170` | empty or 40–64 lowercase hex | Empty takes the hub checkout. |
| `SERVED_MODEL_ID` | `MODEL_ID` | string | Public id in the API. |
| `INSTANCE` | unset | `[A-Za-z0-9_.-]` | Namespaces containers and state. |
| `WIP_INSTANCE`, `WIP_ROOT` | unset | instance / absolute path | `--wip` slot staging. |
| `WIP_BUILD_SEED` | `kakapo` | seed | Site build seed; every launch path accepts it. |

A local `scripts/build/build-dev-images.sh` rebuild immediately moves the shared
coordinator and Spark default image tags, even before its branch merges. Existing
containers retain their old images; rebuild them with `./wip.sh --refresh-containers`
when adopting the new toolchain. Each WIP build pins immutable per-host image IDs
for its entire invocation, including container creation, native cache keys and
sealed metadata, so a concurrent default-tag move cannot change its build inputs.
Versioned `tc-*` image tags remain available for older checkouts.

WIP coordinator and Spark AOT kernel exports run CPU-only (`CUDA_VISIBLE_DEVICES=""`)
with no GPU device or hardware lock required, bounded by host-wide CPU worker
admission and `build-slot.sh`.

`./wip.sh --from-slot NAME` reuses that slot's seed within the selected
`WIP_INSTANCE`/`WIP_ROOT`, rather than choosing the least-loaded seed. New
builds record `slots/NAME/seed-host`; older slots are located with bounded
instance-container probes. An explicit `--seed HOST` that conflicts with the
source seed fails with the host and source path before cloning.

The host-only environment overrides `CUTEAFD_SPARKNEST_RUN_DIR` and
`CUTEAFD_FUSE_SYSFS_ROOT` select the source directories for the optional
read-only SparkNest control and FUSE connection mounts (defaults
`/srv/sparknest/run` and `/sys/fs/fuse/connections`). Container destinations
remain unchanged. Tests seed private directories through the shared fixture;
these are not `KEY=VALUE` launch-config keys.

## Hardware placement

| Key | Default | Values | Notes |
| --- | --- | --- | --- |
| `RTX_GPUS` | `auto` | `auto`, `1`, `2` | Coordinator GPUs the layout may use. |
| `COORDINATOR_GPUS` | lowest index | ordered ids, primary first | Explicit GPU pair. |
| `COORDINATOR_SPLIT` | `auto` | `auto`, `heads`, `off` | Split a two-GPU layout by heads; `off` serves from the primary. |
| `SPARK_COUNT` | every available | `0`–`8` | Upper bound on physical Spark ranks available to the deployment (experts, media or drafter); the planner may use fewer when ranks have no role. `0` is accepted for every family when the planner can fit all owners on the selected RTX GPUs. Only ranks assigned a role start containers, using their configured host indices (unused indices may be skipped); expert peers include expert ranks only. |
| `SPARK_HOSTS` | pool order | comma list | Explicit rank order, same length as `SPARK_COUNT`. |
| `SPARK_<rank>_HOST` | pool host | hostname | Ranks 0–7; required for a host outside the pool. |
| `SPARK_<rank>_LANE_A` | pool value | IPv4 | Rail A address for that rank. |✓ |
| `SPARK_<rank>_LANE_B` | unset | IPv4 | Rail B; all active ranks or none. |
| `EXPERT_BACKEND` | `auto` | `auto`, `local`, `spark` | `local` requires `SPARK_COUNT=0`. Dense Qwen is local-only; the planner admits other families by fit. |
| `RTX_EXPERT_LAYERS` | `auto` | `auto`, `all`, `N` | Pool-first onboarding of routed-expert layers on the RTX pair. |
| `RTX_EXPERT_PEER` | `off` | `off` | `on` is refused; TP2 halves replaced it. |
| `EMBEDDING` | `gpu` | `host`, `gpu` | Untied token-embedding placement. Kimi, Step-5 and Muse accept `gpu` only; plan and serve refuse `host`. |
| `EXPERT_PORT` | `19441` | 1–65535 | Expert-worker port. |
| `HOST_ISA` | `auto` | `auto`, `x86-64-v2`, `x86-64-v4` | Native library ISA. |
| `RDMA_BOND_BALANCE` | `off` | `off`, `labels`, `probe` | RoCE v2 flow-label placement on a coordinator bond. |
| `MAPPED_SCALE_PRELOAD_GIB` | `0` | GiB | Host-RAM cap for preloading mapped tables (V4.1 Engram); 0 keeps all mapped. |
| `ENGRAM_WARM` | `off` | `on`, `off` | V4.1 only: read both Engram tables into page cache after readiness. |
| `CUDA_MODULE_LOADING` | CUDA default (`LAZY`) | `LAZY`, `EAGER` | V4.1 and GLM Flash; `EAGER` costs startup time and memory, avoids first-shape hangs. |
| `SPARK_INTAKE` | `auto` | `auto`, `gpu`, `pinned`, `host` | How routed partials reach the coordinator GPU. |
| `SPARK_PREFILL_REPLY_FORMAT` | unset (BF16) | `bf16`, `fp8`, `nvfp4` | Spark prefill replies; FP8/NVFP4 exported only in release builds. |
| `SPARK_DECODE_REPLY_FORMAT` | unset (BF16) | `bf16`, `fp8`, `nvfp4` | Same for decode replies. |
| `MIMO_WEIGHT_POLICY` | `auto` | `auto`, `checkpoint` | MiMo weight precision policy. |
| `MIMO_PREFIX_DRAFT` | `off` | `on`, `off` | MiMo draft on the prefix path. |
| `MIMO_SNAPSHOT_WAIT` | `off` | `on`, `off` | MiMo snapshot wait. |
| `MIMO_PREFILL_CHUNK_S` | unset | seconds | MiMo prefill chunk time. |

## Memory and pools

One memory policy covers every family
([`cuteafd-core/src/memory_policy.rs`](../rust/crates/cuteafd-core/src/memory_policy.rs)).
Each device has one cap, and weights, KV, workspaces and the reserve all live inside it.
The planner fixes the layout (layer placement, onboarding, buffers) for `KV_TARGET_TOKENS`;
the pool then grows into what the fixed layout leaves, up to `KV_MAX_TOKENS`, and the launch
is refused below `KV_MIN_TOKENS`. The reserve is the sum of named, measured components
(CUDA context and cuBLAS, loaded code, graphs, runtime growth, Spark worker process residency)
plus `FAMILY_RESERVE` and the explicit per-coordinator-GPU `release padding` allowance; `GPU_RESERVE` / `SPARK_RESERVE` replace the reserve components,
not physical driver page tables or startup graph owners (including node padding).
Startup and `cuteafd plan` print one line per device with the cap,
weights, workspaces, every reserve component and the KV target, min, max and pool.
Sparks use each host's idle `/proc/meminfo` `MemAvailable` (GB10 unified
memory reclaims clean file cache; neither `MemFree` nor CUDA free is capacity).
The launcher refreshes a host-local idle sample when no GPU compute process is
resident. A busy host, including a server about to restart, uses only the last
idle sample from its current boot ID; missing or reboot-stale samples fail
closed with a stop-first hint. On `--restart`, this coordinator and selected
host/port workers stop before the idle probe, even if subsequent admission
refuses. Dry runs skip teardown and live probing. Plans retain sample
timestamps, uptime and age.
Offline and simulated plans use the provenance-stamped per-host
`serving_capacity/spark_memory.json` inventory in configured rank order; unknown
hosts require an explicit cap or live measurement. Direct `plan --config` resolves
named hosts in config rank order; `REPLACE-ME` placeholders or no host list use
labelled simulated canonical inventory. Rank-to-host order is not
rearranged. Workers also check live allocatable memory before loading weights.
`--spark-memory-samples` is the launcher's JSON measurement handoff to plan and
serve, not a user policy knob. Explicit `SPARK_MEMORY_CAP` /
`--spark-memory-cap` wins over measurements.

The coordinator charges the SM121 worker process census (1,596,606,464 B,
moa/driver 580.178.04, 2026-11-07, #1454) inside `SPARK_MEMORY_CAP`:
steady anonymous PSS + file PSS + context-phase consumption, subtracting
context-phase file PSS growth to avoid overlap. The worker's live admission
checks future allocations against `MemAvailable`, so it does not charge that
already-resident process/context a second time. The historical 109,760,512-B
unattributed startup remainder is not an owner and is not charged.
`driver page tables` is one opaque runtime owner with distinct backing models,
not a tracked workspace or replaceable reserve. RTX portable pinned backings pay
ceil(bytes/512) per mapping coordinator context; aliases do not pay again.
Spark expert arenas pay 48 B per 4096-B page
(40 B vmalloc metadata + 8 B secondary page tables), rounded once per backing.
This is calibrated from real-worker kernel fields at 40.508, 75.059 and
103.652 GiB (zero-intercept fits 10.052338 + 1.949590 MiB/GiB).
Source attributes 16 B/page to `nvidia_pte_t` and 8 B/page to the retained DMA
page array; 16 B/page of vmalloc remains unattributed. Neither the fit's
intercept nor whole-host reclaim residuals are charged as fixed allowances.
Driver bytes survive `GPU_RESERVE` / `SPARK_RESERVE`, including zero. Ring, workspace,
validation and upload owners also stay separate and are charged once.
Memory sizes take `B`, `MB`, `GB`, `MiB` or `GiB` (a bare number is bytes), not percentages.
`GRAPH_NODE_PADDING` adjusts graph-node and executable rates and accepts percentages.
`MEMORY_PADDING` defaults to `100MiB` (104,857,600 B) per coordinator GPU, inside
`GPU_MEMORY_CAP`. This visible release allowance covers still-unattributed memory;
each release lowers it as owners are attributed. It is charged once, separately
from measured owners and `FAMILY_RESERVE`. `MEMORY_PADDING=0` removes only padding;
`GPU_RESERVE` replaces the whole reserve including padding, even when set to zero.
Sparks are unchanged and use `SPARK_RESERVE`, not `MEMORY_PADDING`.

`GRAPH_NODE_PADDING` is separate: by default it adds **2.5% to both measured
coefficients**, bytes per kernel node and bytes per graph executable, rounding each
adjustment to the nearest whole byte. The model sums `nodes * (a + pad_a)` and
`executables * (c + pad_c)`, then packs once per physical device into 2 MiB chunks.
Both counts come from the declared set; rates are per family and architecture.
It applies even with `GPU_RESERVE` or `SPARK_RESERVE`. An integer adds that many
bytes to each coefficient (per node and per executable); per-architecture values
may mix forms, e.g. `rtx=2.5%,spark=100`. Missing Spark rates require a Spark census,
never an RTX substitution. Whole-byte GLM SM120 coefficients reproduce the 1M
calibration exactly; the independent 128K prediction is one 2 MiB chunk above
measurement (within the independent-validation bar of plus or minus one chunk).
Flash's archived calibration also retains a one-chunk overprediction; it has no
independent context validation yet.

For the archived GLM 5.3 count this adds 74,660,034 B (about 71 MiB) before chunk
rounding; Flash adds 57,692,320 B (about 55 MiB). Together with `MEMORY_PADDING`'s
100 MiB default this is explicit driver-drift headroom, not a second graph charge.
Successful captures above the prediction warn and continue; actual allocation
failures stop. This shared model is staged: existing whole-capture families do
not charge the node padding until their admission is migrated.

Each key is also a global `cuteafd` flag (`--gpu-memory-cap`, `--kv-max-tokens`, ...).

| Key | Default | Values | Notes |
| --- | --- | --- | --- |
| `KV_TARGET_TOKENS` | 2M on every card | positive count | KV the layout is sized for: layer placement, onboarding, buffers. |
| `KV_MIN_TOKENS` | min(256K, model max context, KV target), on every layout | positive count | Refuse automatic pools below this; missing the target is fine down to it. Automatic context serves min(model max context, admitted pool); an explicit context still requires the pool to cover it. |
| `KV_MAX_TOKENS` | unbounded (all remaining memory) | positive count, at least the target | After the target-sized layout and expert onboarding are fixed, the pool takes the remaining admitted memory, bounded by native encoding limits. An explicit value caps this growth. A pinned `RTX_EXPERT_LAYERS` already makes the pool the output (it fills what is left). |
| `GPU_MEMORY_CAP` | full device | size | Total per-GPU ceiling for plan, serve and golden; leaves SM count and L2 unchanged. |
| `SPARK_MEMORY_CAP` | each host's measured idle `MemAvailable` | size | Explicit total per-Spark engine ceiling overrides measured capacities, including worker process residency and routed weights. No extra margin. |
| `GPU_RESERVE` | auto | size | Sets the total GPU reserve; built-in reserve components (including release padding) become 0, but physical driver page tables, named sampled pre-weight occupancy and the bounded graph-cache reserve stay charged. Still inside the cap. |
| `SPARK_RESERVE` | auto | size | Replaces Spark reserve components, not arena driver metadata or actual ring/workspace/staging owners; these survive even a zero override. |
| `SPARK_CARVEOUT` | `auto` | `auto`, `off` | Spark routed-expert weights in the GB10 display carveout lent by a validated [spark-attic](https://github.com/bertholomus/spark-attic) `atticd` (#2546), about 2,044 MiB per Spark on driver 580.178.04. `auto` uses it where present and runs exactly as without it otherwise, logging one line why (not installed, root-only install, refusing, unknown version). Stock interface only: socket `/run/spark-attic/attic.sock`, key `/etc/spark-attic/key`, or the project's own `SPARK_ATTIC_SOCK` / `SPARK_ATTIC_KEY_FILE`. Non-root serving containers need the project's documented step (spark-attic INSTALL.md §6): atticd `--socket-group <group>` plus the key `chgrp <group>`, mode 0640; the launcher mounts the socket directory and key read-only and adds the socket's gid. The idle probe credits each rank's `largest_mib` (outside `MemAvailable`) only when the daemon is ready, validated, fresh, unlent and reachable; the worker maps it carveout-first into its routed-expert arena (one range, carveout then ordinary tail, owner `experts/spark-carveout`, charged once against the lendable bytes). Offline planning uses the measured idle census and its carveout allowance only on expert ranks; live startup samples supersede the census. `off` never connects and plans without it. Global flag: `--spark-carveout`. |
| `MEMORY_PADDING` | `100MiB` | size | Explicit per-coordinator-GPU release padding, inside the cap; `0` disables only padding. `GPU_RESERVE` replaces it. Global flag: `--memory-padding`. |
| `GRAPH_DISABLE_LOW_MEM` | half of `MEMORY_PADDING` (50 MiB by default) | size or `gpuN=size,...` | Coordinator-only physical free-memory floor after the estimated graph and existing instantiate headroom. `0` disables the check. Uses the latest boundary or periodic sample minus ledger and pending graph growth, without a new miss/replay CUDA query; a skip runs eagerly without a ticket or OOM cooldown. Each GPU decides independently; all-rank admission is follow-up work. Independent of `GPU_RESERVE`. Global flag: `--graph-disable-low-mem`. |
| `GPU_GRAPH_RESERVE` | trace-derived family base + bounded extra | size or `gpuN=size,...` | Optional **whole** coordinator graph reserve per device; a user value is charged exactly, without extra, and conflicts with either extra key for that device. Unspecified devices usually use the checked-in prewarm arena plus its 2 MiB tail as the base, then add `min(GPU_GRAPH_RESERVE_EXTRA, GPU_GRAPH_RESERVE_EXTRA_CAP)`. GLM Flash, Qwen4, measured Spark-free Hy4-flash DFly2, and measured V4.1 layouts instead use C1 working-set sizing: the initial soft cap is `max(1.2 × W1, W1 + 100 MB, W1 + 2 × step)` rounded up to 2 MiB, where W1 is the measured C1 working set and step is the largest new-capture growth in a decode step. Admission charges the hard cap: soft plus twice the largest measured single-capture growth rounded up to 2 MiB (one-chunk minimum, headroom capped at half of hard). An explicit total is the exact hard cap; the initial soft cap is total minus that same growth headroom, capped at half the total. Neither extra key applies in these working-set layouts. The runtime can lower the initial soft cap if live capture growth exceeds the census before ARC calibration. V4.1 scales the lane-local working-set and step allowance with resolved `DECODE_ROWS`; the one-lane 64-row C1 working sets are 1,064 MiB (1 RTX) and 604/608 MiB (2 RTX). At 128 rows the no-churn StridedPLE census (#2460, 2026-11-25, C1–C16) retained 3,286 MiB and 1,348/1,356 MiB respectively; these high-water working sets (not C1-only measurements) seed the 128-row reserve. Shared bytes remain fixed, per-lane bytes interpolate/extrapolate with row count, and the 4 MiB maximum single capture is unchanged. Resolved checkpoint/layout defaults: GLM 5.3 one-rank `glm_prewarm_1.json` base 337,641,472 bytes, charged 675,282,944 bytes; two-rank `glm_prewarm_2.json` base 1,379,926,016 bytes (1,377,828,864 measured plus the 2 MiB tail), charged 1,879,926,016 bytes per rank, traced on K4+Core-K6 with 2 RTX + 6 Sparks, 73 replayed geometries and no pins; untraced coordinator rank counts retain the one-rank bootstrap geometry/base without borrowing its pins and warn at startup; GLM Flash uses exact live-row keys at rank-local decode and verify sites, with donors used only under pressure by default. No measured shape accounts for at least half the cycles in a band, so startup pins and prewarm are empty. Its C1 working sets are 74 MiB on TP1, 94/94 MiB on TP2 with split DFlash, and 94/92 MiB on TP2 with a whole or disabled drafter. With 4 MiB maximum single-capture growth per rank, soft/hard caps are 170/178 MiB on TP1, 190/198 MiB on both split-DFlash ranks, and 190/198 and 188/196 MiB on whole/off TP2. Explicit `layers` attention placement keeps decode eager because its layer segments contain peer waits; Mistral Large 4 uses `ml4_prewarm_1.json` (base 715,128,832 bytes) or `ml4_prewarm_2.json` (base 914,358,272 bytes), each plus 500 MB extra; V4 selects `v4_prewarm.json` by resolved shape (coordinator GPU count x Sparks attached: `rtx1-spark`, `rtx1-local`, `rtx2-spark`, `rtx2-local`); a traced shape's base is its startup warm's measured arena plus its 2 MiB tail — 671,088,640 bytes on `rtx1-spark`; 954,204,160 on each `rtx2-spark` rank; 914,358,272 and 924,844,032 on `rtx2-local`, charged 1,171,088,640 / 1,454,204,160 / 1,414,358,272 and 1,424,844,032 bytes at the default extra — covering the replayed width-0 startup rows and 53 / 42 / 37 recurring verify geometries out of the traces' 5,476 / 5,169 / 5,264 verify keys (a width-0 row the trace never replayed, or a verify geometry used only inside one sampling interval such as a single panel or the 32K context pass, stays lazy under the extra), and an untraced shape keeps its provisional numbers (charged 751,619,276 bytes on `rtx1-local` at the default extra) and logs that its shape has no trace card. Qwen4 uses exact live-row keys on rank-local `qwen4.segment` (`Scope::Local`, 0 spanning bytes) with donors used only under pressure by default and empty startup prewarm and pins; its C1 working sets from the donor-off census (`qwen38-exl3-min` / `qwen38-exl3-2rtx`) are 30.08 MiB on 1 RTX (step growth 6.96 MiB, largest capture 4 MiB; soft/hard 126/134 MiB) and 15.25 / 14.69 MiB on 2 RTX (step growth 3.53 / 3.39 MiB, largest capture 2 / 4 MiB; soft/hard 112/116 MiB and 112/120 MiB). V4.1 sizes its measured layouts from the shipping-settings C1 working set scaled by `DECODE_LANES` (`W1 = lanes × W1_lane + shared`, `step = lanes × step_lane`, `capture_growth = 4 MiB` across all ranks): one-rank GPU 0 `W1 = lanes × 984 + 80 MiB` (`step = lanes × 27 MiB`), soft/hard 1,278/1,286 MiB (1 lane) and 2,458/2,466 MiB (2 lanes); two-rank GPU 0 `W1 = lanes × 592 + 12 MiB` and GPU 1 `W1 = lanes × 594 + 14 MiB` (`step = lanes × 11 MiB`), soft/hard 726/734 and 730/738 MiB (1 lane) and 1,436/1,444 and 1,444/1,452 MiB (2 lanes). Startup prewarm inventories remain in `v41_prewarm.json`. An unmeasured V4.1 layout borrows neither another layout's base nor its pins. MiMo selects `mimo_prewarm.json` by checkpoint variant, coordinator rank count and Spark attachment: Flash `mimof-48` 1 RTX + Sparks has a 423,624,704-byte base including its 2 MiB tail, charged 847,249,408 bytes. Pro `mimop-70` 2 RTX + Sparks uses its own per-rank measured row shares and tails; an untraced checkpoint/layout uses unpinned bootstrap, never another key's evidence. Charged once before KV, independent of `GPU_RESERVE`; cache bound equals charged total. GLM/Mistral prewarm fits the base (or user total): pinned entries first, then list order until the next entry does not fit; if the pinned set cannot fit, prewarm none. V4 warms every geometry its list holds that its pool can express. Skips warn and become runtime misses. Zero runs eagerly. A scalar applies independently to each GPU; unspecified `gpuN` selectors retain defaults. Startup, Ready and `/v1/stats` report charge and policy sources. Kimi K3 selects the per-physical-rank base in `kimi_prewarm.json` by checkpoint-native core format (`k6` or `bf16`) and coordinator rank count, plus the same fixed tail and bounded extra; untraced core views use bootstrap keys. BF16 2 RTX + 6 Sparks uses measured bases 1,281,359,872 / 1,279,262,720 bytes, charged 1,783,457,024 / 1,781,359,872 bytes including one 2 MiB tail and 500 MB extra; its coverage-1 adaptive trace selects 168 geometries with 21 pins. Untraced core/layout combinations use unpinned bootstrap geometry and a provisional census base, never another key's measured base, list or pins. Kimi startup fits indivisible pinned geometries first, then replay-hot unpinned geometries using measured per-rank capture shares within the base, never runtime extra. Kimi prefill remains eager and startup expands only replayable rows 1..64. Dense Qwen uses checkpoint/quant/KV-specific single-RTX inventories in `qwen35_prewarm.json`; untraced variants bootstrap unpinned with a provisional 64 MiB base plus one 2 MiB tail. Muse uses `muse_prewarm.json` per rank count and resolved onboarded-layer prefix: (1,2) measured base 1,056,964,608 bytes, charged 1,559,061,760 bytes; (2,0) measured bases 1,287,651,328 / 1,302,331,392 bytes, charged 1,789,748,480 / 1,804,428,544 bytes (each includes one 2 MiB tail and 500 MB extra); an untraced layout has an empty, unpinned bootstrap (zero base), so set an explicit reserve for initial trace collection. Step-5 EXL3 K4+Core-K6 inventories in `step5_prewarm.json` have measured bases 3,407,872,000 bytes (1 RTX + 3 Sparks) and 2,835,349,504 / 2,629,828,608 bytes (2 RTX + 6 Sparks); each adds one 2 MiB tail and the shared 500 MB extra when `DECODE_GRAPHS=on`. Untraced Step-5 layouts have no borrowed base or pins; explicit totals or extra enable bounded bootstrap captures. Inkling EXL3 K4 MC original lag-8 working sets (1 RTX + 4 Sparks; 2 RTX + 6 Sparks) were 302 MiB and 358/144 MiB; admission charges 398 MiB and 454/240 MiB by RTX rank respectively, including one 96 MiB margin per rank and no startup pins. Explicit reserve and headroom keys still override. Other families await cache integration. |
| `GRAPH_PREWARM` | on | `on`, `off` | `off` skips startup prewarm only for rank-local graph sites, which then capture lazily on first use. Accepted by every family on one and two GPUs; Hy4-flash always captures rank-local sites lazily. Spanning sites (a peer wait, peer copy or second-device kernel inside the graph) always prewarm whatever the value: after readiness they run eagerly on a miss, because runtime cross-device instantiation can deadlock behind a peer wait (#1878). Each family classifies its sites in a checked-in table with source evidence; an unclassified site counts as spanning. Does not reduce the admitted graph reserve. CLI: `--graph-no-prewarm`. |
| `GRAPH_DONOR_ALWAYS` | `off` | `on`, `off` | Off by default: keep graphs on a miss below the soft cap; reuse the oldest safe same-topology donor only under pressure. `on` also reuses below the soft cap: owner age must exceed K (V4.1 K=2, other families K=1), with matching native topology. Spanning sites never donate, including prewarm, because their missing shapes cannot be recaptured after readiness; their update safety under peer waits is unproven. CLI: `--graph-donor-always`. |
| `GRAPH_DONOR_COLD_LAG` | `0` | Non-negative integer | Extra idle cycles N for donor-always below the soft cap: donor age must exceed K + N. Non-zero requires `GRAPH_DONOR_ALWAYS=on`. At or above the soft cap (including hard-cap pressure), age > K is unchanged. CLI: `--graph-donor-cold-lag N`. |
| `GPU_GRAPH_RESERVE_EXTRA` | 500 MB; Inkling 96 MiB | size, percentage or `gpuN=value,...` | Runtime capture headroom beyond the family prewarm base, coordinator GPUs only. MB is decimal: 500 MB = 500,000,000 bytes (MiB is binary). Percentages are of the base. An explicit value has no implicit cap; otherwise the default cap is 100% of the base. Not supported in working-set layouts (GLM Flash, Qwen4, measured Spark-free Hy4-flash DFly2, and measured V4.1). Conflicts with an explicit `GPU_GRAPH_RESERVE` total on that device. |
| `GPU_GRAPH_RESERVE_EXTRA_CAP` | 100% of base for default extra; no cap for explicit extra | size, percentage or `gpuN=value,...` | Upper bound on extra runtime headroom; percentages are of the family base. Charged total = base + min(extra, cap). Not supported in working-set layouts. Conflicts with an explicit `GPU_GRAPH_RESERVE` total on that device. No Spark extra keys: workers have no graph owner. |
| `SPARK_GRAPH_RESERVE` | worker family default | size | One graph reserve per Spark worker, matching the scalar `SPARK_RESERVE` convention. Forwarded to workers for GLM 5.3, GLM Flash, Qwen4, V4.1 and MiMo; the live native worker currently runs eagerly with no capture/replay sites, so no worker graph owner is active. Enabling worker capture requires admission ownership and bounded-cache integration together. |
| `GRAPH_TRACE` | off | absolute path | GLM 5.3 / GLM Flash / Mistral Large 4 / V4 / V4.1 / MiMo / Kimi K3 / dense Qwen / Muse schema-2 graph identity, per-cycle/lifetime hit counts, eviction/recapture/ghost counts, first/last-hit time, rolling 60 x 10 s hit intervals, measured boundary capture-byte shares, permanent node/exec cost basis, width/device-span and device high-water snapshots, written as `<stem>.device-N.json`; parent directory must exist. A multi-GPU shape writes one file per coordinator GPU, and its V4 trace invocation passes every rank's file. `scripts/dev/graph-prewarm-from-trace.py --family glm5 --layout 1|2 --trace FILE --output FILE` (GLM 5.3 coordinator rank count; pass both device traces for layout 2, whose half-table keys are doubled to the built width), `--family glm5_flash --layout 1-sparks|2 --trace FILE --output FILE` (GLM Flash, per resolved card layout), or `--family mistral_large4 --layout 1 --trace FILE --output FILE` (one rank; `--layout 2` takes both devices' traces and doubles each traced half table back to the built width), `--family deepseek_v4 --layout rtx1-spark|rtx1-local|rtx2-spark|rtx2-local --list rust/crates/cuteafd-loader/src/serving_capacity/v4_prewarm.json` instead replaces that one shape's per-rank arrays and marks it traced, printing each heat class's keys and arena bytes (sustained pins; plain C1 and the layout's highest-lifetime-hit verify geometry pin whatever their heat; other regular and single-interval rows are prewarmed and stay evictable; rare stays lazy; a width-0 startup row needs a replayed hit like any other geometry; entries are ordered pinned first, then lifetime replay hits descending). `--family deepseek_v41 --layout 1|2 --prefill-rows N --trace FILE --output FILE` generates the V4.1 rank inventory; `--family qwen4` generates shape-keyed entries. `--family mimo_v2 --ranks N --variant mimof-48\|mimop-70 --sparks yes\|no --trace FILE --output FILE` generates MiMo row-only entries, priced from every selected row's indivisible segment/head siblings; capture-only rows remain lazy. Kimi pins plain C1, the dominant verify width and its speculative siblings, and each device/operation lifetime-hit top tier carrying 50% of replay hits (ties included); sustained rolling heat and repeated recapture can also pin. Startup `Other` counts never create request heat. The checked-in Kimi adaptive fixture selects every replayed geometry with coverage 1; zero-replay geometry stays lazy. Kimi uses `--family kimi_k3 --variant bf16\|k6 --ranks N --layout NAME` with every physical device trace for that layout; it sums completed-boundary capture shares for the selected operation/row/speculative geometries, including startup-only siblings, and rounds once per logical rank. Missing shares fail generation; zero shares are valid allocator reuse. `--family qwen3_5 --variant h5120-l64-bf16-kvfp8-rtx1 --trace FILE --output FILE` generates checkpoint/quant/KV-specific dense Qwen entries from replayed complete-layer geometries, including their zero-hit segment siblings in the price. `--family muse_spark --ranks N --onboard-layers N --trace FILE --output FILE` generates Muse row geometries and measured sibling costs, ordered pinned first then replay heat. Runtime headroom comes from the shared extra keys, not the trace. Pinned-hit times are boundary-attributed; ARC-hit times use the cycle/boundary clock. V4.1 combines same-layout snapshots by device/site/bank/geometry and key-order ordinal, summing replay evidence and taking the largest measured capture share per identity; identities never measured in any snapshot are refused. Other families use their latest complete device snapshots. A layout without a checked-in trace prewarms the family bootstrap set unpinned and warns; only trace-derived entries pin. `/v1/stats` exposes top-16 hit/churn identities and per-key telemetry payload bytes without tracing; review and qualify the proposal before checking it in. |
| `GRAPH_NODE_PADDING` | `2.5%` | percentage, integer bytes per coefficient or `rtx=VALUE,spark=VALUE` | Added to both measured node and executable rates, packed once per device into 2 MiB chunks. Independent of `GPU_RESERVE`, `SPARK_RESERVE` and `MEMORY_PADDING`; included once in the graph owner, not a reserve. Global flag: `--graph-node-padding`. |
| `FAMILY_RESERVE` | per-family constant (0 in every family) | size | A named per-GPU reserve added to the measured components; `GPU_RESERVE` overrides it. One key for coordinator GPUs only: no family keeps a per-family Spark margin (Sparks use `SPARK_RESERVE`). |
| `KV_POOL_SIZE` | unset | `B/MB/GB/MiB/GiB` | V4.1 only: fixed KV pool size. |
| `POOL_TOKENS` | `auto` where offered | `auto` or count | Automatic pool uses memory left after target-sized placement and onboarding; a number pins it. |
| `HOST_CACHE_BYTES` | `auto` | `auto`, `0`, byte size | Prefix-cache host tier; `0` disables it. The daemon's `--host-cache-min-tokens` (512) is the shortest snapshot it keeps, except in families with positional marks (recurrent or sliding-window state): with the host tier on they keep only two device staging marks, so a retained mark lives on the host or nowhere, and their host minimum is clamped to `--prefix-cache-min-tokens` (64). Pages-only families keep 512; their short snapshots stay on the device. Host eviction is byte-accounted least-recent (leaves before reusable ancestors), so short snapshots cost only their own pages and mark ([#2582](https://github.com/tpurtell/cuteafd/issues/2582)). |

## Attention and KV

The old `-HQA` checkpoints were withdrawn from Hugging Face. Historical results remain in the [benchmark archive](../benchmarks/README.md#v5-hqa-comparison).

| Key | Default | Values | Notes |
| --- | --- | --- | --- |
| `CORE_VIEW` | unset (= `original`) | `original`, `k8-compact`, `k6-compact`, `fp8-compact` | Immutable MC dense-core view. Plain checkpoints support original only. FP8 converts exactly original `core_tensors` from its BF16 index with the shared block-128 recipe; no alternate index reads. Flash supports its full MC core. Muse supports its full MC selection (q/k/v/o, optional gate, pre/post-expert projections and padded vocabulary head); it stays explicit, not default. Kimi K3.1 supports its full MC selection (KDA/MLA projections, shared FFN, routed latent projections and vocabulary head). GLM/Step support their selected target core; Step includes MTP core, never routed experts. Unselected operands stay native. GLM and Step require v52e coordinator exports. Measured speed/quality costs are in Optional performance levers; original stays default, FP8 stays opt-in. Already-FP8 sources refuse. |
| `ATTENTION_PLACEMENT` | `auto` | `auto`, `context`, `layers`, `heads` | Per-family defaults and refusals are in the table below. |
| `ATTENTION_PLAN_ROWS` | `auto` | `auto`, positive int | Decode/verify reduction plan rows. |
| `ATTENTION_PRECISION` | `bf16` | `bf16`, `fp8` | Shared MLA quality tier for prefill, decode and verification. The default `bf16` uses the fastest qualified kernel that meets BF16-tier accuracy, regardless of operand format. The opt-in `fp8` selects a faster, lower-quality route where available. Neither GLM 5.3 nor GLM Flash has a separate lower tier for native prefill: GLM 5.3 uses native BF16 and GLM Flash uses two-term E4M3 Q2P2 under both settings. The key still selects the qualified split-decode tier. Neither tier changes projection or weight precision or KV storage. Kimi K3 compact-record native prefill already widens to BF16 and uses BF16 MMAs: both settings keep it, and K3 prefill has no lower tier. K3.1 sparse MLA keeps its qualified attention under either setting. V4.1 accepts `fp8` but resolves it to `bf16`; it has no FP8 attention tier. Non-MLA families without an FP8-tier program keep their native attention under either setting. Plan, serve and golden use `--attention-precision`. |
| `MAX_CONTEXT_TOKENS` | `auto` | `auto`, 1–1048576 | `auto`: serve min(model maximum context, admitted pool). A number pins context and requires the pool to cover it. Plan and every serve command use `--max-context-tokens` with `auto` or a number; explicit CLI overrides the config. Compiled workspace/table extent is internal, not a user context setting. |
| `MAX_OUTPUT_TOKENS` | `131072` | integer | Generation cap. |
| `KV_CACHE` | family default | `bf16`, `fp8`; MiMo also `int8` | See the precision table below. |
| `SWA_KV_CACHE` | `int8` | `bf16`, `int8`, `fp8` | MiMo sliding/MTP rings. |
| `DRAFT_CONTEXT_WINDOW` | `0` | Nonnegative token count | Kimi K3.1 DSpark context override (other launcher families reject this key). Unset/0 preserves native attention (full context for Lithos K3.1 DSpark, trained window for Inferact); a positive value selects a window, e.g. 2048. It never changes target KV or target taps. |
| `DRAFT_KV_CACHE` | `fp8` | `bf16`, `fp8` | Drafter ring storage (GLM 5.3, GLM Flash, MiMo, Muse, Kimi GQA block dSpark and MLA DFlash2, Inkling). Every drafter selects BF16/FP8 independently of target KV and body weights; default FP8. |
| `PREFIX_CACHE_ENTRIES` | `auto` | `auto`, non-negative | Snapshot entries; auto derives from context × concurrency. |
| `PREFIX_CACHE_MARK_MIB` | unset | MiB | Recurrent/ring state retained per mark. |
| `PREFIX_PARTIAL` | `off` | `on`, `off` | MiMo approximate partial reuse; off keeps exact restores only. V4.1 always uses its official bounded replay for partial divergence (approximate encoder SWA; 128 tokens, extended before overlapping images). Both pick the snapshot that leaves the fewest rows to prefill or replay; an exact restore wins a tie. A retained entry whose positional mark is gone restores by bounded replay from its own pages. Approximation is a label only (`ApproximateReplay`, `approximate_lineage_hits`), scoped to the replay's shared frontier: pages before it keep content identities and host dedup. Exact restores copy the retained state byte-exactly. Other families restore exact snapshot frontiers only and refuse `on`. |
| `PREFIX_POINT_BOUNDARIES` | `1` | `0`, `1` | Dense Qwen: retain the exact frontier before the generation message. |
| `GLM5_FLASH_PREFIX_MARKS` | unset | `arena`, `pool` | GLM Flash KDA state marks. |
| `GLM5_FLASH_KDA_STATE` | `f32` | `f32`, `bf16` | GLM Flash KDA recurrent state. |
| `GLM5_FLASH_INDEX_CACHE` | `auto` | `keys`, `compact` | GLM Flash index cache; auto picks compact on one GPU. |
| `DECODE_ROWS` | `auto` (64) | `auto`, `1..=CONCURRENCY×(max_draft+1)` | Global decode/verify row budget per decode lane for every family. `auto` resolves to `64`. Values above `64` require exported coordinator decode programs for that family and topology (V4.1 up to `256` on 1 or 2 GPUs; GLM Flash up to `128` on 1 GPU; MiniMax, Inkling, Kimi K3, Qwen4 and Muse up to `128` on 1 or 2 GPUs; other families refuse `>64` naming the missing exports). Admission charges the resolved rows once across workspaces, verify logits and Spark decode owners, reducing KV headroom accordingly. |
| `DECODE_LANES` | unset (`1` on 1× RTX, `2` on 2× RTX for V4.1; `1` for others) | `1`, `2` | Independent decode lanes, each with its own workspaces, graphs, Spark link and decode execution. V4.1 resolves by coordinator RTX count (`1` lane on 1× RTX, `2` lanes on 2× RTX) at `DECODE_ROWS=64`, honors explicit `1` or `2` overrides, and always resolves `1` lane at `CONCURRENCY=1`; one lane holds up to `min(CONCURRENCY, 64)` requests in one batch within the `DECODE_ROWS` budget, returning the second lane's bytes to VRAM. Every other family runs one lane and refuses `2`. |
| `DECODE_SHARE` | `0.2` | `[0, 1)` | Share of time running requests keep while prompts prefill; 0 prefills whole prompts. |
| `DECODE_GRAPHS` | GLM 5.3 / ML4, MiMo, dense Qwen, Muse, Step-5 and Inkling `on` | `on`, `off` | Dense Qwen graphs complete rank-local decode/verify layers on one or two RTX GPUs; shared residual hops between whole-layer owners and prefill stay eager. Two-rank layouts use separate untraced bootstrap reserves, never single-rank trace pins. With graphs off, its graph reserve is zero and page-table geometry stays on the original eager path. MiMo bounded per-layer CUDA graph replay, keyed by checkpoint and layout. Muse Spark defaults to `on` for bounded rank-local decode/verify segments; its prefill stays eager. Step-5 bounds decode/verify sections around eager KV staging and host joins; prefill remains eager. Step-5 selects trace-owned row geometries per native format, dense core, RTX count and Spark count from `step5_prewarm.json`, fitting pinned-first then a heat-ordered prefix across all ranks; startup uses disposable full-step transactions without publishing KV state. The shipped EXL3 K4+Core-K6 layouts (1 RTX + 3 Sparks and 2 RTX + 6 Sparks) retain their historical measured geometry pins. The shipped min/max traces select 58/59 replayed row geometries, respectively; min pins sustained row46 and max has no sustained pins. Capture-only rows remain lazy. Other Step-5 layouts have no borrowed pins or measured base. Inkling uses exact live-row rank-local attention and FFN sections for decode and verify on one or two RTX GPUs; prefill, all peer exchanges and expert dispatch stay eager. `off` removes its graph allowance from admission and disables capture and replay. Inkling captures lazily with no startup pins and uses the shared pressure-only donor default. GLM 5.3 / ML4 `off` runs eagerly and overrides coordinator graph reserves; standalone `plan --glm-decode-graphs on\|off` matches serving. Standalone `plan --mimo-decode-graphs on\|off` and `--step5-decode-graphs on\|off` match serving defaults. |
| `KV_DECODE_RESERVE_CHUNK` | unset | tokens | Decode-chunk KV reserve top-up. |
| `KV_RESERVE_DRAIN_BUFFER` | unset | 0 = whole remaining output | Drain buffer. |
| `KV_EARLY_TERMINATION_MIN_OUTPUT_TOKENS` | unset | `-1` or non-negative | Early termination floor. |

## Speculation

| Key | Default | Values | Notes |
| --- | --- | --- | --- |
| `SPECULATOR` | family default | `auto` (Qwen4/Kimi), `off`, `dflash2`, `dspark`, `mtp`, `dfly2` | An eligible Kimi/GLM Flash automatic external drafter must be installed, including its config; a missing checkpoint fails with a `nest hf download` command rather than silently serving without speculation. Explicit `off` and unsupported automatic targets remain off. K3.1 pairing reads verifier model/architecture declarations and target hidden/vocabulary/tap/mask dimensions from checkpoint metadata, including MC source identity; no checkpoint revision or core-view allowlist is used. Pairing mismatches name the field and refuse launch. |
| `SPECULATOR_MODEL_ID` | family default | HF repo id | External drafter checkpoint. |
| `SPECULATOR_MODEL_REVISION` | empty | 40–64 lowercase hex | Drafter revision. |
| `SPECULATOR_DEPTH` | 3 for local Qwen/Step-5, else 1 | positive int | Native MTP draft depth. |
| `DRAFT_LENGTH` | unset (follows `DRAFT_POLICY`) | positive int | Measurement override: verify a fixed number of draft tokens per step, min(value, the drafter's native block), further capped by the row budget at high concurrency. Overrides `DRAFT_POLICY`; acceptance history is still recorded. Every family and speculator (DFlash2, dSpark, MTP, Markov). Without a speculator it is ignored with one startup log; reports and cards say `no speculation`. |
| `SPECULATOR_FP8` | `auto` | `auto`, `on`, `off` | GLM/Qwen/MiMo/Kimi drafter precision: unset, `auto` and `on` select single-copy FP8 body linear projection weights; `off` keeps BF16. Rust resolves the same MiMo precision for planning and serving, independent of checkpoint alias or core view. Kimi block DSpark supports TP1 and physical TP2 with native BF16 norms/Markov tensors; FP8 applies to content-paired K3.1 DSpark in every core view and layout. Kimi MLA includes QKV-A, Q-B, absorbed UK/UV and output; embeddings, norms, codebooks and prediction heads remain native. Independent of `DRAFT_KV_CACHE`. Inkling accepts `auto` or `off` only (checkpoint native). |
| `DRAFT_POLICY` | `history` | `history`, `clamped`, `off` | How many drafted positions each request verifies. `history` keeps the leading drafts while the product of their calibrated per-position probabilities stays above 0.3. Both `history` and `clamped` share one scored neural explore row per round or lane. Choose the least recently judged position/score bin, then nearest the joint keep cut, then ticket order. Skip it if it would grow the padded verify shape into a bucket larger than eight rows; exact-row paths allow it within their verify budget. Token-only and copy tracks keep per-request exploration. `clamped` is `history` with each round's cut set to clamp(2 / (x + 1), 0.3, 0.45), where x is the round's non-row time per request over its per-row time, learned online per context band from the engine's own round times (0.3 until the band's line exists); see Optional performance levers. `off` verifies no drafts. See the score sources and verify-row budget below. |
| `DRAFT_CONTEXT_SLOTS` | `max(20, drafts)` | positive int | GLM/MiMo drafter context slots. |
| `DRAFT_SEQUENCES` | engine default | positive int | Maximum drafts in one batch. |
| `SPECULATION_TRACE` | unset | absolute path | JSONL speculation trace. Shared `draft_round` records include each request's numeric engine ticket and HTTP `request_id` (the identity in `usage.speculation`), per-position history probabilities/counts, the per-token `score` the policy keys on, the drafter's raw per-token `head` and its engine-calibrated `calibrated` probability (one online per-position calibration trained on verifier-reached positions, for dSpark heads, DFlash2 candidate probabilities and Inkling's candidate probability), exploration and verifier outcomes. Scores are read back under `history` or whenever this trace is open, so fixed `DRAFT_LENGTH` traces carry them too. V4.1, MiMo and GLM Flash rounds also carry `lane`, the host-clock `round_us`/`draft_us`/`verify_us` their console round measures, and each request's committed `context`; the shared scheduler's `cycle` record and GLM 5.3's `cycle` lines carry the same times. |
| `GLM5_FLASH_DRAFT_LINEAR` | auto by drafter | `w8a16`, `wide`, `w8a8` | GLM Flash drafter GEMM; Rust resolves `w8a8` with an external FP8 drafter, `w8a16` otherwise. |

`history` uses one keep rule for every family and score source (#2460): keep
the leading drafts while the cumulative product of their calibrated
per-position acceptance probabilities is strictly greater than 0.3, the
estimated chance the whole prefix survives verification. The probabilities:

- Per-token score: a trained confidence head (dSpark), a raw candidate-token
  probability (DFlash2), or Inkling's recurrent-head candidate probability (its
  argmax log-softmax). Every score goes through one online, per-position Platt
  fit; only positions reached by the verifier train it. A cold fit is the
  identity.
- No per-token score (MTP): that track's per-position history estimate. The
  estimate uses the request's accepted/reached counts over its last 8 verify
  rounds, shrunk toward the server-wide rate over the last 256 rounds that
  reached the position. The prior starts at 8 and fades linearly to zero over
  the request's first 3 completed verify rounds. An unreached position stops
  the plan after the fade; the cold server rate is 1 for neural drafts and
  0.25 for copies.
- A history copy extending an agreeing draft keeps its per-position rule: each
  copy position is kept while its own history estimate exceeds 0.3.

Scored drafters can propose their full native block (7 drafts for V4.1).
The plan includes the first position past the kept prefix as an explore if
that position exists; there is no separate head skip. If the round's verify-row
budget cannot fit the plan, the next position of each request's prefix competes
for rows by the same score or history estimate, highest first.

With a drafter and an enabled draft policy, responses include request-local
`usage.speculation` counters (`request_id`, `proposed`, `accepted`, `cycles`),
including zero counters when the request proposed nothing. The field is absent
when the server has no drafter or its effective draft policy is `off`.

## Prefill

| Key | Default | Values | Notes |
| --- | --- | --- | --- |
| `PREFILL_PACKING` | engine default | `unrestricted`, `exact` | Pack short prompts directly; `exact` preserves expert arithmetic routes. GLM Flash uses this option alone (default `unrestricted`), currently on one RTX only; TP2 packing is tracked in #1865. |
| `PREFILL_BATCH_TOKENS` | `2048` | 80–4096 | Prefill chunk size. GLM 5.3 / ML4 use `PREFILL_ROWS` for per-unit rows. |
| `PREFILL_ROWS` | planner | multiple of 64, ≤4096 | Per-unit rows override. |
| `PREFILL_WAVES` | planner | 0–4 | Units in flight. |
| `PREFILL_SPAN_ROWS` | planner | positive int | GLM/MiMo shared span. |
| `TARGET_HEAD` | `tensor` | `tensor`, `exact` | BF16 target head: tensor cores with FP32 accumulation, or the exact FP32 GEMM diagnostic. |
| `FP8_PREFILL` | unset (= `auto`) | `auto`, `off`, comma-separated projection groups | Shared prefill activation policy over already-resident FP8 weights; never converts BF16/K6/K8. GLM Flash groups: `mla`, `ffn`, `kda-in`, `kda-o`, `experts`. Auto preserves W8A8 MLA+FFN and leaves KDA opt-in. Missing W8A8 geometry falls back to W8A16 with a plan hint. Quantization scratch aliases the admitted program workspace, charged once. Experts auto preserves native input-format defaults; explicit `experts` selects W8A8 for FP8 experts, global `off` selects W8A16. Experts groups not named keep their native default; `experts` must be named to force W8A8. MXFP4, NVFP4 and BF16 expert math is untouched. GLM 5.3 has selectable `mla`, `index`, `ffn`, and `experts` routes; ML4 has selectable `index` and `experts` routes. Both refuse unavailable explicit groups and preserve calibrated or fixed native precision. Other dense-family routes follow incrementally. |
| `EXPERT_INPUT` | family default | `fp8`, `bf16`, `bf16-decode` | MiMo Spark expert inputs. |
| `EXPERT_PARTIAL_BF16_MAX_ROWS` | engine | row count | BF16 expert partials up to this row count. |
| `MIMO_PREFILL_CHUNK_S` | unset | seconds | MiMo prefill chunk time. |
| `GLM5_FLASH_EXL3_SCHEDULE` | `default` | `default`, `gb10` | GLM Flash Spark EXL3 decode schedule. |
| `GLM5_FLASH_EXL3_WORKER_PATH` | `async` | `async`, `blocking` | GLM Flash Spark EXL3 upload/wait path. |
| `GLM5_FLASH_EXL3_ROUTE_DUMP` | unset | absolute dir | Per-Spark EXL3 route dump. |
| `GLM5_FLASH_EXL3_ROUTE_DUMP_CALLS` | `200000` | positive int | Dump call cap. |
| `GLM5_FLASH_REPLAY_RECORDS` | `auto` | `shared`, `own` | GLM Flash replay records (`auto` selects `shared` on one GPU with an automatic pool and admitted Spark experts, else `own`). |

## Media

| Key | Default | Values | Notes |
| --- | --- | --- | --- |
| `VISION` | `auto` | `auto`, `off`, `rtx[:gpu]`, `spark[:rank]` | Vision tower placement; `auto` enables the family's bundled tower unless it would drop the KV pool below `KV_MIN_TOKENS`. `spark[:rank]` may use a media-only Spark when every expert layer is on RTX (include it in `SPARK_COUNT`); that rank starts only the tower, not expert transport. GLM Flash passes physical tower ranks independently of expert peers, including sparse ranks and replicas. |
| `AUDIO` | `auto` (V4.1: `off`) | `auto`, `off` (V4.1: `off` only) | Bundled audio tower. V4.1 has no audio encoder. |
| `VISION_REPLICAS` | `1` | 1–8 | Vision replicas. |
| `MEDIA_CACHE_BYTES` | engine | byte size | Host embedding-cache quota for an enabled media tower. On the GLM engine this applies to ML4 vision only (default min(8 GiB, 5% of host RAM)); the same quota goes to plan and serve. |
| `MEDIA_FEATURES_DIR`, `AUDIO_FEATURES_DIR` | unset | absolute dir | Encoder feature caches. |
| `CHAT_TEMPLATE_FROM` | checkpoint | snapshot dir | GLM Flash template source when the checkpoint's template cannot mark images. |

`./run.sh --dry-run` and serving both verify the native AOT exports on the
planned media owners, without starting CUDA work. A slot or image that lacks
required exports is refused rather than silently changing placement. Rebuild
with `CUTEAFD_WIP_VISION_ATTENTION_AOT=ON` / `CUTEAFD_WIP_AUDIO_AOT=ON` (or the
corresponding `CUTEAFD_RELEASE_` flags), or turn the unavailable tower off.

The direct daemon's `--max-image-tokens` accepts an explicit 1-16384 per-image
LM span cap. Omission uses the processor's default span, limited by its
architectural ceiling: existing families default to 4096 (Qwen capped at 1024,
Mistral at its checkpoint span), MiniMax to 576, and Kimi to 16698. Spark encoder workers accept this full native ceiling; each
checkpoint tower contract still validates its own smaller shape limit. An explicit
cap can only lower a declared ceiling; it never changes vendor resize geometry
or silently truncates feature rows. Kimi's full default is intentionally above
the CLI's largest explicit value. These registration limits do not qualify an
otherwise inactive tower for serving. `detail=low` remains capped at 256.

Kimi K3.1 (`kimi_k31`) uses the shared Kimi engine. MC `original` (BF16),
MC `k6-compact` and standalone native Core-K6 are launchable with real Spark
experts after accepted full-model fidelity and partial paired scoring proof
(#1777/#2128: 77 exact windows, zero mismatches, explicitly accepted by TJ).
The [max example](../examples/configs/kimi-k31-exl3-k2-mc-max.config) selects
MC `CORE_VIEW=original`, 2 RTX + 6 Sparks and automatic pool/onboarding.
With measured idle Spark capacities (#2082), MC `k6-compact` plans on
1 RTX + 6 Sparks with an automatic 2,081,152-token pool and no RTX expert
onboarding; this is a CPU placement result, not a new serving benchmark.
K8 compact serves under the same structural and memory admission as original
and K6; fidelity measurements are reported separately, never a launch gate.
`fp8-compact` is opt-in and converts the original MC selection to E4M3 with FP32 block-128 scales. MLA KV-A pads 576 output rows to 640 for storage; attention consumes the same logical 576 channels. The published K3.1 MC leaves the layer-zero dense FFN, absorbed KV-B, indexer, router and recurrence weights native. Unsupported selected roles refuse by tensor name before loading. Requires matching coordinator FP8 exports; hardware performance and Standard fidelity are not yet measured.
All accepted views retain native BF16 indexer weights. K3 DFlash2 hidden-state
compatibility is independently unqualified. `SPECULATOR=dspark` selects the
checkpoint-native Inferact K3.1 block drafter (override with `SPECULATOR_MODEL_ID`
and `SPECULATOR_MODEL_REVISION`), with BF16 or opt-in FP8 projection weights (`SPECULATOR_FP8=on`) and independent
`DRAFT_KV_CACHE=fp8|bf16` window rings (default FP8). One coordinator runs TP1;
two coordinators split query/KV heads and FFN neurons across both devices,
with per-rank packed arenas, local rings and exact local prefix marks. FC and
norms are replicated; only the head GPU owns Markov/confidence and borrows the
target vocabulary head. The shared GQA TP2 reduction uses drafter-private
exchange flags, never target exchange slots. Unset/`auto` `SPECULATOR` selects DSpark
for `wrldsuksgo2mars/Kimi-K3.1-Pro-EXL3-K2-MC`, `CORE_VIEW=original|k6-compact`,
on one or two coordinators with target FP8 KV. The independent defaults are
`SPECULATOR_FP8=auto` (FP8 projection weights) and `DRAFT_KV_CACHE=fp8`;
explicit `SPECULATOR=off`, weight precision or draft KV overrides win.
The max example uses this automatic selection, not an off pin. Qualification
uses the measured config and MC-manifest SHA-256 digests plus selected view and physical TP,
never a repository/path string or Spark count; identical local/renamed copies qualify.
The Inferact drafter config digest is checked independently;
normal memory admission still decides fit. K8/whole-core FP8 target views,
other checkpoints and target BF16 KV remain default-off.
Cold and full-prompt warm strict sets passed 10/10 on the qualified TP2 views;
TP1 K6 is separately gated. FP8 weights save 1.227 GB net per coordinator versus
BF16 after scratch, independently of draft KV. Closed-batch C16 has a substantial
uneven-completion drain penalty; sustained C8/C16 measurements in #2261 select
the throughput arm. Sampled acceptance is not directly comparable to Inferact's
reported 4.16 because its draft sampling/rejection method differs (#2312). `TP2_DFLASH` does not select this GQA split.
The optional causal `RadixArk/Kimi-K3.1-Dflash2-Preview` MLA drafter verifies
8 tokens (current token plus 7 drafts). Its checkpoint `block_size=5` describes
the trained block; causal path sampling extends it to width 8. Admission and
startup graph inventories use the full verification width. Noncausal DFlash2
block-8 drafters are unchanged.

MiniMax and Kimi default to their bundled still-image towers (`VISION=auto`), placed
Spark-first with RTX fallback by joint admission. The tower can reduce the KV
pool below its target, but not below `KV_MIN_TOKENS`. `VISION=off` keeps text-only
serving and rejects image chat. Dense Qwen has automatic media off. Kimi and MiniMax `plan --layout` reserve their bundled
still-image tower as an independent packed arena plus phase-peak scratch,
Spark-first with RTX fallback in the joint admission solve. Native still-image serving uses the admitted owner through the shared vision
runtime; the numerical drift diagnostics remain on #1984 and #1779. GLM 5.3 / ML4,
GLM Flash, MiMo, Qwen4 and Muse carry their own towers. Native GLM Flash EXLR8
K4 min/max examples use automatic bundled vision; the planner admits its resident
weights and scratch alongside the KV pool and routed experts.

## Precision

Target weights keep checkpoint precision unless an explicit conversion option is
selected: unset = checkpoint/native format. This includes ordinary BF16
checkpoints and the MC `original` view. Native FP8 checkpoint weights remain
FP8. Drafters may retain their qualified FP8 defaults: target verification
controls output. Qwen has one shared target/MTP head: BF16, the draft reads it;
there is no extra draft-only FP8 copy. KV cache dtype is independent and unchanged.

Keys here that are optional and off by default are also listed under
[Optional performance levers](#optional-performance-levers).

| Key | Default | Values | Notes |
| --- | --- | --- | --- |
| `EXLR8_MANIFEST` | published `bit325` | manifest name (`up1`, `bit325`, …) | Native EXLR8 experts; needs 1–2 RTX and 1–8 Sparks. |
| `EXLR8_PREFILL_ACTIVATIONS` | `a16` | `a16`, `a8` | Native EXLR8 trellis prefill activations; decode stays A16. |
| `V41_FP8_HEAD` | `auto` (engine `packed`) | `auto`, `off`, `draft`, `all`, `packed` | V4.1: unset/`auto` selects the exact packed BF16 target head and coarse dSpark plane on single/dual RTX, including push loading. `all` uses E4M3 for both; `off` keeps BF16; `draft` keeps BF16 plus E4M3 and requires mounted loading. Packed adds 300.214 MiB total over `all` (150.107 MiB/GPU on dual RTX); ordinary admission decides fit. |
| `FP8_SCALES` | `amax` | `amax`, `pow2`, `best` | Scale rule for FP8 copies made at load. |
| `L2_PREFETCH` | `off` | `off`, `auto`, MiB | Pull the next layer's weights into L2 during one-lane decode. |
| `QWEN_FP8_DECODE` | unset | `on`, `off` | Qwen GDN/attention projections to resident E4M3. |
| `QWEN_FP8_HEAD` | unset (BF16) | `on`, `off` | Explicit conversion of the shared target/MTP head to resident E4M3. |
| `MIMO_FP8_HEAD` | unset (= `auto`, checkpoint) | `auto`, `on`, `off` | Target head FP8 is opt-in; native FP8 stays FP8. Drafter defaults are independent. |
| `MIMO_FP8_O_PROJ` | unset (= `auto`, checkpoint) | `auto`, `on`, `off` | Explicit BF16 target O-projection conversion; native FP8 stays FP8. |
| `GLM5_FLASH_FP8_HEAD` | unset (= `auto`, checkpoint BF16 on 1 or 2 GPUs) | `auto`, `on`, `off`, `packed` | GLM Flash BF16 checkpoint head: FP8 copy, BF16, or an exact 12-bit operand copy (one GPU). Metadata-selected block-FP8 and native K6/EXLR8 heads ignore this conversion setting. |
| `GLM5_FLASH_KDA_FP8` | unset (`auto` = `off`) | `auto`, `off`, `row128`, `channel` | Explicit BF16 KDA projection conversion to resident E4M3; metadata-selected native KDA stages supersede this key without duplicate residency; auxiliary gates keep BF16. Recurrent state and KV defaults are independent. |
| `GLM5_FLASH_KDA_SPLIT` | `auto` | `auto`, `partials` | GLM Flash KDA split output. |
| `GLM5_FLASH_TARGET_HEAD` | `auto` | `auto`, `tensor`, `exact` | GLM Flash's `TARGET_HEAD` spelling. |
| `TP2_ATTENTION`, `TP2_QUERY_PROJECTION`, `TP2_OUTPUT_PROJECTION` | `off` | `on`, `off` | V4.1 TP2 projections on the RTX pair. |
| `TP2_DSPARK_EXPERTS` | `on` with dual RTX + dSpark, otherwise `off` | `on`, `off` | V4/V4.1 TP2 draft experts; requires two RTX GPUs and dSpark. Explicit `off` retains GPU0-only draft experts. |
| `TP2_DFLASH` | GLM 5.3/GLM Flash/MiMo/Muse dual-RTX DFlash2, Mistral Large 4 dual-RTX dSpark and Hy4-flash dual-RTX DFly2: `on`; Kimi: checkpoint geometry; otherwise: `off` | `on`, `off` | Drafter tensor parallelism on an active two-RTX split with DFlash2, Mistral Large 4 `mtp_dspark` (`SPECULATOR=dspark`), or Hy4-flash DFly2. GLM 5.3 defaults to `on` with two RTX GPUs and DFlash2 (explicit `off` keeps the whole movable drafter; `--drafter-gib` forces whole placement): query/KV heads and MLP neurons split across both GPUs over the target peer exchange, residual/convolution/norm owners stay replicated, and selector weights and vocabulary head stay on GPU0. Mistral Large 4 defaults to `on` with two RTX GPUs and bundled `dspark` (explicit `off` keeps the whole movable drafter; `--drafter-gib` forces whole placement): 32 MLA heads and half dense FFN intermediate per rank over the target peer exchange, replicated context projection/norms/656-byte FP8 window cache, and RNN Markov head/masks on GPU0. Muse defaults to `on` with dual-RTX DFlash2 (`off` remains available): rank-local query/KV heads and MLP neurons use the existing target peer exchange; selector and vocabulary head stay on GPU1. Hy4-flash defaults to `on` with dual-RTX DFly2 (`off` retains the whole GPU0 drafter): attention heads (16/16) and MLP neurons (4608/4608) split across both GPUs using the existing target `PeerExchange`, each rank gathers target DSA KV following the active target placement (local+peer parity under ContextSplit, local under Heads), norms and small projections replicate, and `mtp_fusion`, `hidden_correction`, `final_norm`, and argmax stay on GPU0. GLM Flash defaults to the head split with two RTX GPUs and DFlash2, regardless of admitted KV pool size; explicit `off` keeps the whole drafter on GPU0. `--drafter-gib` forces whole placement because TP2 requires checkpoint-derived physical owners. GLM Flash selector weights and head-only workspace stay on GPU0. Plan accepts `--draft-fp8 true|false` (omission defaults to FP8 for GLM 5.3 and GLM Flash) and `--draft-linear w8a16|wide|w8a8` and charges the same representation and scratch as serve. GLM 5.3, MiMo and GLM Flash split query/KV heads and MLP neurons, retaining replicated residual/convolution owners. Kimi derives Context support from checkpoint geometry: noncausal h64 defaults to split cache/attention with producer weights and FFN on GPU1; causal h96 defaults to the admitted whole-drafter home because its Context partial kernel is absent. Explicit `on` for h96 names the missing kernel and refuses before allocation. CLI false overrides config on. |

Benchmark cards report the resolved drafter placement, process CUDA owner and lead ranks, and admitted bytes per owner, not just the requested launch settings. The JSON report also records the physical GPU UUIDs when the launcher supplies them.

## Concurrency and serving

| Key | Default | Values | Notes |
| --- | --- | --- | --- |
| `ADDR` | `0.0.0.0:8000` | `HOST:PORT` | Listen address. |
| `CONCURRENCY` | boot-time derivation | 1–64 | Decode batch concurrency; every C-dependent size follows it. |
| `HTTP_QUEUE_DEPTH` | `CONCURRENCY` | positive int | Accepted request queue; GLM 5.3 / ML4 accept 1–4096. |
| `HTTP_QUEUE_WAIT_MS` | `25000` | ms | Queue wait before refusal. |
| `API_KEY_FILE` | unset | readable file | API key; env `API_KEY_FILE` also works. |
| `ENABLE_BENCH` | `off` | `on`, `off` | Enable the in-server bench endpoints. |
| `GATEWAY` | binary default `on` | `on`, `off` | Messages/Responses/Realtime routes. |
| `USAGE` | `on` | `on`, `off` | Usage history tier. |
| `EXPERT_USAGE` | `on` | `on`, `off` | Device-side executed expert routes, separate prefill/decode tallies and 5 s / 60 s / total windows at `/v1/stats`. Active for V4.1, V4, MiMo, Kimi K3, GLM 5.3 / ML4, Hy4-flash and MiniMax by default; `off` disables counting and telemetry owners. Families without an adapter remain effectively off and show the `/experts` placeholder. V4/GLM 5.3/ML4 `--skip-routed-experts` and MiMo `--skip-experts` disable telemetry before admission, even with `on`. Direct engine/plan CLI uses `--expert-usage=false` to opt out. Shared adapters extend coverage. Each canonical producer GPU owns counters; one worker merges their aggregate snapshots into one bounded engine history. `/experts` uses the console's admin cookie. Rejected speculative verification rows count; GLM 5.3/ML4 startup warm-up and graph-capture replays are excluded; graph padding and TP2 replicas do not. Window and benchmark phase boundaries are approximate asynchronous samples; short phases can share a sample. A 5 s snapshot-completion timeout logs an error and leaves history stale without stopping serving. Totals saturate at u64 limits; merged-counter overflow stops sampling with an error. No disk persistence. |
| `CONSOLE_ASSETS` | `off` | `on`, `off` | Serve console pages from a checkout; `on` needs `--wip`. The `/memory` dashboard uses the same admin console cookie as `/experts`; its read-only `/console/memory` snapshot combines all accepted Growth owner forms (including deferred owners), reconciled worker workspace and future-ring owners, background allocation inventory and published KV/prefix/graph counters. Named-owner bars use exact capacity scales; the domain selector expands non-weight/non-KV owners on a subtotal scale. It polls each second and retains five minutes only while open. Allocation inventory refreshes every 10 s; reconciliation remains stage-sampled, with Spark launch-relative figures diagnostic only. Missing fields are unavailable, never zero. |
| `CONSOLE_TEXT` | `on` | `on`, `off` | Live console token text for unlocked viewers. |
| `HOST_WAIT` | `block` | `spin`, `block` | Serving host-thread wait mode. |
| `IDLE_PARK_SECONDS` | `30` | positive number | Park polling threads after this idle time. |
| `THERMAL` | `on` | `on`, `off` | NVML thermal sampling for stats/bench reports. |
| `LOOP_ABORT_RUN`, `LOOP_ABORT_COPIES`, `LOOP_FAST_SPAN`, `LOOP_FAST_COPIES` | 512, 8, 1024, 8 | integers | Loop-abort rule guards; 0 disables. |
| `PROBE_DUMP_ROOT` | unset | existing absolute dir | Remote bench probe dump root. |
| `COORDINATOR_DOCKER_INFERENCE`, `COORDINATOR_DOCKER_DEV`, `SPARK_EXPERT_DOCKER_*` | release/dev image names | image ref | Image selection. |

## Profiling

| Key | Default | Values | Notes |
| --- | --- | --- | --- |
| `PROFILE` | unset | comma list of `census`, `transport-timing`, `expert-timing`, `expert-routes`, `draft-timing`, `nsys`, `ncu` | Wraps the coordinator for one launch; nsys and ncu are exclusive. |
| `PROFILE_DIR` | unset | absolute dir | Report destination; also runner-owned for `wip-cards --probe memory` without a profiler. |
| `PROFILE_NCU_KERNEL` | unset | kernel regex | Required by `PROFILE=ncu`; each match is replayed. |
| `PROFILE_NCU_COUNT`, `PROFILE_NCU_SET`, `PROFILE_NCU_SKIP` | 1, `full`, 0 | 1–9999 / section set / integer | ncu replay count, section set, warm-up matches skipped. |
| `PROFILE_NSYS_GPU_METRICS`, `PROFILE_NSYS_CPU`, `PROFILE_NSYS_GRAPH_TRACE` | unset | GPU list / `on`/`off` / `node`/`graph` | nsys extras. `PROFILE_NSYS_CPU=on` adds CPU sampling and context switches; the profiled container alone gets `--cap-add PERFMON` (seccomp denies `perf_event_open` without it), and it refuses to start, printing nsys's environment check, if nsys cannot sample (#1700). Graph trace defaults to `node` (each kernel inside a graph replay is timed). A layout that captures graphs with kernels on both coordinator GPUs (V4.1 on two GPUs with TP2 dSpark experts) resolves to `graph` and refuses `node` when the coordinator's nsys is older than 2026.5.2 or its version cannot be read: older nsys node tracing faults when such a graph is instantiated (#2094; 2026.2.1 and 2026.3.2 fault, 2026.5.2 passes). Dev and release images pin Nsight Systems 2026.6.1, so their default remains `node`; the fallback stays active for older custom images. The launch logs the detected version. Under the fallback, each graph replay is one span; eager kernels keep per-kernel timing. |
| `PROFILE_NSYS_ROOT` | unset | absolute dir | Dev and profiling only: run `DIR/nsys` (mounted read-only at its own path) instead of the image's Nsight Systems, to try another version without rebuilding the dev image. Requires `PROFILE=nsys`; coordinator only. |
| `PROFILE_EXPERT_INPUT_SAMPLE`, `PROFILE_EXPERT_ROUTE_WEIGHTS` | unset | row count / `on`/`off` | Require `PROFILE=expert-routes`. |
| `PROFILE_SPARK`, `PROFILE_SPARK_RANKS` | unset | `nsys`, `ncu` / rank list | Profile Spark workers instead of the coordinator. |
| `PROFILE_SPARK_NCU_*`, `PROFILE_SPARK_NSYS_GPU_METRICS` | unset | as above | Spark-side profiling. |

The memory probe publishes measured per-logical-GPU sampler context bytes in
`PROFILE_DIR/observer-context.json`. The launcher declares that file to the
coordinator only; startup waits at most 30 seconds for the measurement and fails
if it is missing or invalid. Logical memory samples exclude those bytes from
used memory and from physical capacity, so the context owner is production-only
and an uncapped census leaves exactly the observer's bytes out of the pool.
An explicit smaller `GPU_MEMORY_CAP` uses existing physical headroom instead.
Raw physical samples and `memory.jsonl` remain unchanged. No observer declaration
means no adjustment.

`CUTEAFD_DIAG_KIMI_STARTUP_MEMORY` is a private coordinator-only Kimi
process environment diagnostic, not a launch configuration key. Pass it through
`wip-cards --arm-env`; unset or `off` does not sample or synchronize.
`pre-capture` samples in-process raw CUDA memory and native backing changes at
at most 64 synchronized phase boundaries, logs signed used/tracked/unclassified
deltas, then intentionally exits before target graph capture or Ready. It adds
no external observer, request probes, admission adjustment or automatic allowance.

`CUTEAFD_KIMI_DSA_DUMP`, `CUTEAFD_KIMI_DSA_SCORES` and `CUTEAFD_KIMI_DSA_KEYS`
are private coordinator-only Kimi K3.1 diagnostics for the DSA index route, not launch
configuration keys; pass them through `golden-run --arm-env`. Only golden runs
attach them: a serving engine logs a warning and ignores a stray variable.
`CUTEAFD_KIMI_DSA_DUMP=DIR` writes every launched layer's post-selector token
indices to `DIR/layer-NN.bin` (`KDSA0001`, one record per batch row),
and `CUTEAFD_KIMI_DSA_DUMP_FROM=N` restricts records to engine positions at or
after `N`. `CUTEAFD_KIMI_DSA_SCORES=DIR` additionally launches the scored
selector after the plain one on the same stream and writes the per-slot scores
aligned with those ids (`KDSAS001`, descending, `-1` tails preserved);
`CUTEAFD_KIMI_DSA_SCORE_LAYERS=3,59` restricts which layers pay the extra
launch and `CUTEAFD_KIMI_DSA_SCORE_FROM=N` the positions. The scored launch
must select exactly the plain route's set; otherwise it fails with the layer
and position and writes no score file. `CUTEAFD_KIMI_DSA_KEYS=DIR` reads the
stored index key the selector actually scores: for each token in the required
`CUTEAFD_KIMI_DSA_KEY_TOKENS=...` list it resolves the physical page through the
same live page table row and writes that page's E4M3 codes and FP32 UE8M0
scales (`KDSK0001` records, `KDSKP001` content-interned pages), with
`CUTEAFD_KIMI_DSA_KEY_LAYERS=3,59` and `CUTEAFD_KIMI_DSA_KEY_FROM=N` restricting
the readback. All three diagnostics refuse attached CUDA graphs and startup
capture, so run them on eager arms only. Unset: no readback, no extra launch,
no stream drain.

`CUTEAFD_DIAG_SELECTION_DUMP=DIR` is a private coordinator-only diagnostic for the
sparse-selecting families (GLM 5.3, GLM Flash, V4.1, MiniMax M3.1, Kimi K3.1), not a
launch configuration key; pass it through `wip-cards --arm-env`. Eager decode drains
the selecting stream after each selector and appends every decode row's selection to
`DIR/layer-NNN.bin` (`SSEL0001`, one record per row and step, repeated positions kept)
at the family's native unit, as ascending logical ids: GLM 5.3 and Kimi tokens (GLM
records only full-indexer layers), GLM Flash 4-token pools, V4.1 compressed rows (2
tokens at layers 2/8/14, 1 from 20), MiniMax 128-token blocks per KV head (`block * 4 +
head`). `CUTEAFD_DIAG_SELECTION_DUMP_FROM=N` drops rows before engine position `N`.
It refuses attached CUDA graphs and context attention placement (use `heads` or
whole). GLM 5.3 and GLM Flash skip startup graph prewarm and Kimi serves without
startup graphs while it is set; V4.1 needs `GPU_GRAPH_RESERVE=0` to keep its index
selection eager (GLM Flash refuses that key, #2586). `scripts/bench/sparse_selection_study.py` reads the files (#2576).
Unset: no drain, no copy, no behavior change.

`CUTEAFD_DIAG_PEER_WAIT=1` is a private coordinator-only diagnostic for every
family with a two-GPU head split, not a launch configuration key. Pass it through
`wip-cards --arm-env ARM:CUTEAFD_DIAG_PEER_WAIT=1`; the launcher forwards every
`CUTEAFD_DIAG_*` variable to the coordinator unchanged. Each peer exchange then
keeps one 48-byte progress record per flag on each GPU (`max(256, 48 × (slots + 1))`
bytes per exchange per GPU). Every push and wait logs its host issue, and a wait
that times out dumps both ranks' per-flag frontiers before ending the engine.
GLM 5.3, Step-5 and Qwen4 admission charge those records as transport bytes.
GLM Flash admission does not: its records, a few KiB per GPU over the target,
route, context and draft exchanges, come out of the release memory padding
(`MEMORY_PADDING`, 100 MiB by default). The per-issue host logging slows decode,
so leave it off for measurements.

`CUTEAFD_DIAG_STACK_ON_TIMEOUT=1` is a private coordinator-only diagnostic for any
family, not a launch configuration key; pass it through
`wip-cards --arm-env ARM:CUTEAFD_DIAG_STACK_ON_TIMEOUT=1`. A watcher thread dumps every
thread's stack to the coordinator log (`stall stack` lines) in two cases: when a
submission has issued no peer push or wait for 20 s while it is in flight, which
catches a blocked host thread while the stall is still live, and the first time a
peer wait times out on the device. Each line gives the thread's name, scheduler state
and kernel wait channel, plus its return addresses as `module+offset`. Resolve them
offline with `addr2line -f -C -e <module> <offset>` against the sealed slot binaries.
Collection uses a `SIGUSR1` handler that writes into pre-allocated slots; no ptrace
is involved. Unset or any value other than `1`: no thread, no handler, no counters.

## Family support: option values

Attention placement (`ATTENTION_PLACEMENT`; `auto` resolves through the
family's planner and runtime policy):

| Value | V4.1 | V4 | GLM 5.3 / ML4 | GLM Flash | MiMo | Qwen4 | Qwen dense | Kimi K3 | Step-5 | Muse | Inkling | Inkling Mini | Hy4-flash | MiniMax |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `auto` (default) | ✓ layers¹ | ✓ context | ✓ context | ✓ context | ✓ heads | ✓ layers | ✓ layers² | ✓ context⁴ | ✓ layers | ✓ heads | ✓ heads | ✓ heads | ✓ context⁵ | ✓ heads |
| `heads` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓⁴ | ✓ | ✓ | ✓ | ✓ | ✓⁵ | ✓ |
| `context` | ✗ | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | ✓⁴ | ✗ | ✗ | ✗ | ✗ | ✓⁵ | ✗ |
| `layers` | ✗ | ✗ | ✓ (explicit; auto stays context)³ | ✓ (explicit; auto stays context) | ✗ | ✓ | ✓ (two RTX) | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ (two RTX) |

`✓` means the launch path accepts the value; `✗` means it stops with the
family's attention-placement error.

¹ V4.1 is `layers` on two GPUs and `heads` on one; an explicit `heads` is
accepted on either. ² Dense Qwen `auto` retains one whole owner on one RTX;
on two RTX it selects solver-admitted contiguous whole-layer ranges, as does
explicit `layers`. Per-rank native arenas, GDN/KV, prefix marks and lazy decode
graphs remain local; the shared residual-hop mechanism joins the ranges. This
primarily adds memory capacity, not parallel layer compute. Explicit `heads`
and `context` name the missing native sharded exports (#2133), without fallback.
Single-rank graph traces/pins are never borrowed by the two-rank layout. ³ Native GLM 5.3 EXLR8 runs `context` (the auto default) and
explicit `heads` or two-GPU `layers` (#1305, #1308); the planner refuses a
mode whose image lacks its programs. ⁴ Kimi K3 accepts explicit `heads`,
`context` and `layers` for MLA layers on two GPUs; `auto` selects `context`
on two GPUs and retains whole-owned layers on one. KDA stays head-split on
two GPUs, independent of the MLA placement. ⁵ Hy4-flash `auto` selects
`context` (ContextSplit, parity-owned KV/index) for MLA/DSA layers and
head-split for SWA-512 layers on two GPUs, and retains whole-owned layers on
one; explicit `heads` opts into replicated-KV head-split across all layers.

Token embedding placement (`EMBEDDING`; `cuteafd plan --embedding-placement`):

| Value | V4.1 | V4 | GLM 5.3 / ML4 | GLM Flash | MiMo | Qwen4 | Qwen dense | Kimi K3 | Step-5 | Muse | Inkling | Inkling Mini | Hy4-flash | MiniMax |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `gpu` (default) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `host` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |

Host placement additionally requires an untied embedding/head. Kimi and Muse
keep their admitted GPU embedding: host offload would save 2.19 GiB and
3.08 GiB respectively, but add mapped-PCIe gathers to every token and prefill;
no runtime benefit is qualified for their current fitting layouts. Both plan
and serve refuse it rather than silently planning a GPU table. Step-5 likewise
requires its embedding on the GPU entry owner. The launch-key table records
whether a key is accepted at all, not whether all its values are supported.

`EXPERT_USAGE` has a live adapter only for V4.1, V4, GLM 5.3 / ML4, MiMo,
Kimi, Hy4-flash and MiniMax. Other families reject an explicit key instead of accepting a no-op;
the `/experts` placeholder remains available without telemetry.

Speculator (`SPECULATOR`):

| Value | V4.1 | V4 | GLM 5.3 / ML4 | GLM Flash | MiMo | Qwen4 | Qwen dense | Kimi K3 | Step-5 | Muse | Inkling | Inkling Mini | Hy4-flash | MiniMax |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `off` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ (default) | ✓ | ✓ (default) | ✓ |
| `dflash2` | ✗ | ✗ | ✓ (default) | ✓ (default) | ✓ | ✗ | ✗ | ✓ (default) | ✗ | ✓ (default) | ✗ | ✗ | ✗ | ✗ |
| `dspark` | ✓ (default) | ✓ (default) | ✓ (Mistral Large 4 only) | ✓ | ✗ | ✗ | ✓ (default unless hidden is 5120) | ✓ (content-paired K3.1 default in every view/layout; otherwise opt-in) | ✗ | ✗ | ✗ | ✓ (default) | ✗ | ✓ (default) |
| `mtp` | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ (default) | ✓ (local EXL3/NVFP4 default, depth 3) | ✗ | ✓ (default) | ✗ | ✗ | ✗ | ✗ | ✗ |
| `dfly2` | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ |

Drafter tensor parallelism (`TP2_DFLASH`) on active two-RTX DFlash2 / Hy4-flash DFly2 layouts:

| Layout | GLM 5.3 / ML4 | GLM Flash | MiMo | Kimi K3 | Muse | Hy4-flash |
| --- | --- | --- | --- | --- | --- | --- |
| Two RTX with DFlash2 / DFly2 | `on` | `on` | `on` | checkpoint geometry | `on` | `on` |
| One RTX or no DFlash2 / DFly2 | `off` | `off` | `off` | `off` | `off` | `off` |

Qwen4 unset/`auto` selects native MTP whenever the checkpoint declares its MTP
layer, independent of expert quantization, core view, repository revision or
admitted Spark expert world. The one native layer recursively drafts three
tokens by default; `SPECULATOR_DEPTH` overrides that depth and explicit
`SPECULATOR=off` disables drafting. Multiple checkpoint MTP layers require an
unimplemented executor and refuse precisely; unsupported tensors or native
exports still refuse through the ordinary structural contract. Media-only
Spark layouts still require an available worker and complete encoder admission.
Qwen4 keeps the exact pre-allocation-admitted expert set resident, with no runtime paging.
`RTX_EXPERT_LAYERS=auto` is pool-first on one or two RTX devices, including
Spark-backed layouts: whole routed layers on one RTX, TP2 halves on two, and
the remaining layers on the Sparks. `all` and `N` are exact fit-or-refuse pins;
`--rtx-expert-layers` does not require `--split-device`. Zero onboarded layers
and all-local layouts use the same per-layer executor. Two-RTX attention owns
whole layer ranges (`ATTENTION_PLACEMENT=layers`), independent of expert homes.

KV precision (`KV_CACHE`; `SWA_KV_CACHE` is MiMo only):

| Value | V4.1 | V4 | GLM 5.3 / ML4 | GLM Flash | MiMo | Qwen4 | Qwen dense | Kimi K3 | Step-5 | Muse | Inkling | Inkling Mini | Hy4-flash | MiniMax |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `bf16` | n/a (fixed) | n/a | n/a | n/a | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ (default) | ✓ | ✓ | ✓ | n/a |
| `fp8` | n/a | n/a | n/a | n/a | ✗ | ✓ (default) | ✓ (default) | ✓ (default) | ✓ (default) | ✓ (opt-in) | ✓ (default) | ✓ (default) | ✓ (default) | n/a (FP8 fixed) |
| `int8` | n/a | n/a | n/a | n/a | ✓ (default) | ✗ | ✗ | ✗ | ✗ | n/a | n/a | n/a | n/a | n/a |
| `SWA_KV_CACHE` `fp8` | n/a | n/a | n/a | n/a | ✓ | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

V4.1, V4, GLM 5.3/ML4 and GLM Flash fix their target cache formats. Muse
accepts `KV_CACHE=bf16` (default) or experimental `fp8`: native E4M3 payloads
with per-token/per-head FP32 K/V scales on both full and sliding layers.
Muse FP8 passes absolute fidelity but fails paired quality and C1 default bars;
it remains opt-in (measurements in the performance-levers table). GLM families and
Muse, MiMo and Kimi block dSpark expose independent `DRAFT_KV_CACHE`
(`bf16`/`fp8`, default `fp8`) for drafter rings; it never changes target KV.

Inkling defaults to `KV_CACHE=fp8`; `bf16` is an explicit option for global layers. FP8 records hold 2,048 E4M3 payload bytes
plus 16 FP32 per-token/head K/V scales: exactly 2,112 bytes per global layer
per pool token, versus 4,096 bytes in BF16. Local rings stay BF16 and
convolution marks stay FP32. Tail copies include the scales byte-exactly.

Media:

| Key | V4.1 | V4 | GLM 5.3 / ML4 | GLM Flash | MiMo | Qwen4 | Qwen dense | Kimi K3 | Step-5 | Muse | Inkling | Inkling Mini | Hy4-flash | MiniMax |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `VISION` | ✓ (`rtx`/`spark`; local is `rtx:0`) | ✓ | ✓ | ✓ (`auto` needs a template that can mark images) | ✓ | ✓ | ✓ (`auto`/`off`; auto starts none) | ✓ (`auto`/`off`; auto starts none) | ✓ (`auto`/`off`; auto starts none) | ✓ | `auto`/`off` (auto starts none) | ✓ (`auto`/`rtx`/`off`; bundled PE-Lang) | ✓ | ✓ (`auto`/`off`; auto starts none) |
| `AUDIO` | `off` (default; no tower) | `auto`/`off` | `auto`/`off` | `auto`/`off` | `auto`/`off` | `auto`/`off` | `auto`/`off` (auto starts none) | `auto`/`off` (auto starts none) | `auto`/`off` (auto starts none) | `auto`/`off` | `auto`/`off` (auto starts none) | `auto`/`off` (auto starts none) | `auto`/`off` (auto starts none) | `auto`/`off` (auto starts none) |

Native EXLR8 (`EXLR8_MANIFEST`): GLM 5.3/ML4 and GLM Flash, the only
architectures EXLR8 checkpoints declare. Placement is 1–2 RTX plus 1–8 Sparks; GLM Flash TP2 also
allows 0 Sparks. The speculator must be `off` or `dflash2`. GLM needs its
attention heads or a two-GPU context/layers placement; native GLM EXLR8 takes
context (auto), heads and two-GPU layers (#1305, #1308).

## Family support: launch keys

Every key the launcher knows, checked against each launch path's real
accept/refuse behaviour. `✓` means the family's config validation accepts it;
`✗` means the launch stops with `unused configuration key: KEY` (exit 2).
`SPARK_<rank>_*` covers ranks 0–7. In the combined GLM 5.3 / ML4 column,
`CORE_VIEW` is accepted for both families. ML4 original keeps its native E4M3
weights and F32 block-128 scales; only `original` is accepted because its
original core is already FP8, not an MC publication.

| Key | V4.1 | V4 | GLM 5.3 / ML4 | GLM Flash | MiMo | Qwen4 | Qwen dense | Kimi K3 | Step-5 | Muse | Inkling | Inkling Mini | Hy4-flash | MiniMax |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Identity** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `MODEL_ID` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `MODEL_REVISION` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `SERVED_MODEL_ID` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `INSTANCE` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `WIP_INSTANCE` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `WIP_ROOT` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `WIP_BUILD_SEED` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| **Placement** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `RTX_GPUS` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `COORDINATOR_GPUS` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `COORDINATOR_SPLIT` | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `SPARK_COUNT` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `SPARK_HOSTS` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `SPARK_<rank>_HOST` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `SPARK_<rank>_LANE_A` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `SPARK_<rank>_LANE_B` | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| `EXPERT_BACKEND` | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `RTX_EXPERT_LAYERS` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ |
| `RTX_EXPERT_PEER` | ✗ | ✓ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| `EMBEDDING` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `EXPERT_PORT` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `HOST_ISA` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `RDMA_BOND_BALANCE` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `MAPPED_SCALE_PRELOAD_GIB` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `ENGRAM_WARM` | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| `CUDA_MODULE_LOADING` | ✓ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| `SPARK_INTAKE` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `MIMO_WEIGHT_POLICY` | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| `MIMO_PREFIX_DRAFT` | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| `MIMO_SNAPSHOT_WAIT` | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| `MIMO_PREFILL_CHUNK_S` | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| **Memory and pools** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `KV_TARGET_TOKENS` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `KV_MIN_TOKENS` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `KV_MAX_TOKENS` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `GPU_MEMORY_CAP` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `SPARK_MEMORY_CAP` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `GPU_RESERVE` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `MEMORY_PADDING` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `GRAPH_NODE_PADDING` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `SPARK_RESERVE` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `SPARK_CARVEOUT` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `GRAPH_DISABLE_LOW_MEM` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `GRAPH_PREWARM` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `GRAPH_DONOR_ALWAYS` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `GRAPH_DONOR_COLD_LAG` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `GPU_GRAPH_RESERVE` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `GPU_GRAPH_RESERVE_EXTRA` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `GPU_GRAPH_RESERVE_EXTRA_CAP` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `SPARK_GRAPH_RESERVE` | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| `GRAPH_TRACE` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `FAMILY_RESERVE` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `KV_POOL_SIZE` | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| `POOL_TOKENS` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `HOST_CACHE_BYTES` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| **Attention and KV** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `CORE_VIEW` | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `ATTENTION_PLACEMENT` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `ATTENTION_PLAN_ROWS` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `ATTENTION_PRECISION` | ✓ (`fp8` resolves to `bf16`) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `MAX_CONTEXT_TOKENS` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `MAX_OUTPUT_TOKENS` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `KV_CACHE` | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ |
| `SWA_KV_CACHE` | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| `DRAFT_KV_CACHE` | ✗ | ✗ | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | ✗ | ✓ | ✓ | ✓ | ✗ | ✗ |
| `PREFIX_CACHE_ENTRIES` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `PREFIX_CACHE_MARK_MIB` | ✗ | ✗ | ✗ | ✓ | ✓ | ✗ | ✓ | ✗ | ✗ | ✗ | ✓ | ✓ | ✓ | ✓ |
| `PREFIX_PARTIAL` | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| `PREFIX_POINT_BOUNDARIES` | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| `GLM5_FLASH_PREFIX_MARKS` | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| `GLM5_FLASH_KDA_STATE` | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| `GLM5_FLASH_INDEX_CACHE` | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| `DECODE_ROWS` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `DECODE_LANES` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `DECODE_SHARE` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `DECODE_GRAPHS` | ✗ | ✗ | ✓ | ✗ | ✓ | ✗ | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ |
| `KV_DECODE_RESERVE_CHUNK` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `KV_RESERVE_DRAIN_BUFFER` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `KV_EARLY_TERMINATION_MIN_OUTPUT_TOKENS` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| **Speculation** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `SPECULATOR` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `SPECULATOR_MODEL_ID` | ✗ | ✗ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ |
| `SPECULATOR_MODEL_REVISION` | ✗ | ✗ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ |
| `SPECULATOR_DEPTH` | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ | ✓ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| `DRAFT_LENGTH` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `DRAFT_CONTEXT_WINDOW` | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| `SPECULATOR_FP8` | ✗ | ✗ | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ |
| `DRAFT_POLICY` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `DRAFT_CONTEXT_SLOTS` | ✗ | ✗ | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| `DRAFT_SEQUENCES` | ✗ | ✗ | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| `SPECULATION_TRACE` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `GLM5_FLASH_DRAFT_LINEAR` | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| **Prefill** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `PREFILL_PACKING` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `PREFILL_BATCH_TOKENS` | ✓ | ✓ | ✗ | ✓ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| `PREFILL_ROWS` | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `PREFILL_WAVES` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `PREFILL_SPAN_ROWS` | ✗ | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| `TARGET_HEAD` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `FP8_PREFILL` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `EXPERT_INPUT` | ✗ | ✓ | ✗ | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| `EXPERT_PARTIAL_BF16_MAX_ROWS` | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ |
| `GLM5_FLASH_EXL3_SCHEDULE` | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| `GLM5_FLASH_EXL3_WORKER_PATH` | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| `GLM5_FLASH_EXL3_ROUTE_DUMP` | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| `GLM5_FLASH_EXL3_ROUTE_DUMP_CALLS` | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| `GLM5_FLASH_REPLAY_RECORDS` | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| **Media** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `VISION` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `AUDIO` | ✓ (`off` only) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `VISION_REPLICAS` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `MEDIA_CACHE_BYTES` | ✗ | ✗ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `MEDIA_FEATURES_DIR` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `AUDIO_FEATURES_DIR` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `CHAT_TEMPLATE_FROM` | ✗ | ✓ | ✗ | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| **Precision** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `EXLR8_MANIFEST` | ✗ | ✗ | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| `EXLR8_PREFILL_ACTIVATIONS` | ✗ | ✗ | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| `V41_FP8_HEAD` | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| `FP8_SCALES` | ✗ | ✗ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| `L2_PREFETCH` | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| `QWEN_FP8_DECODE` | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| `QWEN_FP8_HEAD` | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| `MIMO_FP8_HEAD` | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| `MIMO_FP8_O_PROJ` | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| `GLM5_FLASH_FP8_HEAD` | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| `GLM5_FLASH_KDA_FP8` | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| `GLM5_FLASH_KDA_SPLIT` | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| `GLM5_FLASH_TARGET_HEAD` | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| `TP2_ATTENTION` | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| `TP2_QUERY_PROJECTION` | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| `TP2_OUTPUT_PROJECTION` | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| `TP2_DSPARK_EXPERTS` | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| `TP2_DFLASH` | ✗ | ✗ | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | ✗ | ✓ | ✗ | ✗ | ✓ | ✗ |
| **Concurrency and serving** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `ADDR` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `CONCURRENCY` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `HTTP_QUEUE_DEPTH` | ✓ | ✗ | ✓ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| `HTTP_QUEUE_WAIT_MS` | ✓ | ✗ | ✓ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| `API_KEY_FILE` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `ENABLE_BENCH` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `GATEWAY` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `USAGE` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `EXPERT_USAGE` | ✓ | ✓ | ✓ | ✗ | ✓ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ |
| `CONSOLE_ASSETS` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `CONSOLE_TEXT` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `HOST_WAIT` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `IDLE_PARK_SECONDS` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `THERMAL` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `LOOP_ABORT_RUN` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `LOOP_ABORT_COPIES` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `LOOP_FAST_SPAN` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `LOOP_FAST_COPIES` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `PROBE_DUMP_ROOT` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `COORDINATOR_DOCKER_INFERENCE` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `COORDINATOR_DOCKER_DEV` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `SPARK_EXPERT_DOCKER_INFERENCE` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `SPARK_EXPERT_DOCKER_DEV` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `SPARK_PREFILL_REPLY_FORMAT` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `SPARK_DECODE_REPLY_FORMAT` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| **Profiling** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `PROFILE` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `PROFILE_DIR` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `PROFILE_NCU_KERNEL` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `PROFILE_NCU_COUNT` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `PROFILE_NCU_SET` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `PROFILE_NCU_SKIP` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `PROFILE_NSYS_GPU_METRICS` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `PROFILE_NSYS_CPU` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `PROFILE_NSYS_GRAPH_TRACE` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `PROFILE_NSYS_ROOT` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `PROFILE_EXPERT_INPUT_SAMPLE` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `PROFILE_EXPERT_ROUTE_WEIGHTS` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `PROFILE_SPARK` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `PROFILE_SPARK_RANKS` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `PROFILE_SPARK_NCU_KERNEL` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `PROFILE_SPARK_NCU_COUNT` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `PROFILE_SPARK_NCU_SET` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `PROFILE_SPARK_NCU_SKIP` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `PROFILE_SPARK_NSYS_GPU_METRICS` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

MiniMax one-RTX accepted keys and retained qualification gates are listed in
[MiniMax Serving](#minimax-serving); its family column is listed separately until
two-RTX serving qualification completes (#1856).

`cuteafd plan --config FILE` and the launcher share deterministic GLM Flash index,
replay and draft-linear defaults. Plan reports `resolved_settings` with explicit,
config or auto-rule provenance, plus `planning_inputs` identifying simulated or
explicit device capacity and estimated runtime owners. The plan
`--coordinator-weight-budget-gib` default follows this capacity (including
`--gpu-memory-cap`); the launcher passes Qwen occupancy as an explicit adjusted
capacity. Explicit weight ceilings override that derivation.

Kimi DFlash2 planning resolves the configured external drafter snapshot and
defaults its weights to FP8, as the launcher does. MiMo defaults DFlash2 when
the checkpoint bundles a `dflash/` drafter or one is configured, including MC
views and renamed snapshots; otherwise it defaults off. Drafter packing validates
actual FP8 matrix alignment, not the target family name. Explicit MTP defaults
to one stage. These settings
are resolved in Rust for both config planning and launching. MiMo plan uses
`--mimo-kv-cache int8|bf16` for its full-attention records (separate from the
MLA `--kv-cache`), matching the launcher's `KV_CACHE`.
