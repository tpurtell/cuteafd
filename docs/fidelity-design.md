# Golden fidelity: metric, bar, set and harness (design, 2026-10-05)

## Dataset policy (2026-10-18)

A finished fidelity set contains own-model window text and locally measured
original-quality reference logits. Dataset creation never requires engine
qualification, paired serving runs, or engine-derived calibration. The
historical engine comparison measurements later in this document are
consumer-side evidence, not prerequisites for publishing a dataset.

- Text comes from the model's official API where available, otherwise our
  highest-fidelity local copy. Local generation is greedy BF16/checkpoint-native;
  hosted generation uses temperature 0 but is not claimed deterministic.
- Record endpoint/model/provider/date, reasoning/content/tool splits, recipe
  deviations, checkpoint/tokenizer/template hashes and cached response identity.
  Render fixed conversations with the checkpoint's own chat template; reconstructed
  generated spans must re-tokenize losslessly. Execute tools in the same fixture.
- Logits come from the original-quality model run locally, layer-streamed where
  necessary. An explicitly chosen quant of a very large model is an acceptable
  reference; record its exact checkpoint and arithmetic rather than implying BF16.
- Text checkpoint equals the scored model, or a declared higher-fidelity source
  of that same model (official API/original checkpoint with explicit evidence).
  Foreign-model borrowed text is rejected, not published provisionally.
- Reference finite-logit, snapshot/hash, byte-identical common-prefix and compact
  storage checks gate publication. No engine run is part of validate or finalize.
- New own-text sets use declared policy floors: top-1 >=90%, KL <=0.06 nat,
  confident top-1 >=96%, top-3 >=97%; paired loss margins remain 1 point and
  0.5 point respectively. These are policy, not values inferred from reference
  logits or calibrated against an engine. A compatibility audit of all 28 current
  cards and all 120 metric-bearing exports found no confident/top-3 failures at
  these floors; 98% confident would newly fail 13 previously passing cards.
  Unreplaced historical sets stay intact.
- Publish and validate the replacement, re-pin its immutable revision, then remove
  exactly the replaced borrowed-text config in a separate dataset commit. History
  retains the former config. Shared dataset index/card edits are coordinator-owned.

The numerical comparison design below describes how consumers can judge kernel
changes using the fixed dataset; its old calibration campaigns are historical.

## 0. Recommendation in one box

- **Hard bar (paired, same engine, same positions, decode-shaped scoring):**
  the candidate's top-1 agreement with the reference may be lower than the
  checkpoint-precision arm's by at most **0.5 point**, and its mean
  full-vocabulary KL(reference ‖ engine) higher by at most **0.005 nat**,
  both as one-sided 95% upper bounds of the paired difference (McNemar-style
  discordant pairs for top-1, window-block bootstrap for KL). Scored on
  assistant-generated positions only.
- **Absolute floor (regime check):** every arm, the BF16 arm included, keeps
  top-1 ≥ 90% and KL ≤ 0.06 nat against the family golden on the new set.
  90% is a floor the set is built to clear by several points, not the
  target; it fails only for broken kernels or a stale reference.
- **Tripwires (report, fail only when gross):** confident-top-1 (reference
  p₁ ≥ 0.5) and candidate argmax in reference top-3 use family/config
  expectations: round the minimum repeated baseline down to a whole
  percentage point, then subtract 2 points, using the least restrictive
  decode/prefill expectation. Fail below those calibrated expectations, or
  when the one-sided 95% lower bound of baseline minus candidate exceeds
  **1.0 point** for confident-top-1 or **0.5 point** for top-3. Always report
  raw baseline/candidate values, loss, and lower/upper bounds. V4.1's 98%/99%
  constants are legacy context, not universal floors; frozen legacy reports
  without calibrated fields retain their original constants and verdicts.
  NLL on human code stays within +0.01 nat of the baseline arm; zero invalid
  tool-call JSON in the agentic replay.
- **Set:** 64 windows, 32,768 scored positions per family, chat-templated
  agentic coding (reasoning + tool calls + diffs) 40%, long-context code
  reading 20%, plain repository code 15%, JSON/tool grammar 10%, prose and
  reasoning 15%; contexts 1K–16K (32K once the reference does chunked
  attention). Text from our own repository, our own agentic recordings and
  the family's own greedy continuations.
- **Two tiers:** quick (12 windows, 6,144 positions, ≈40 s engine time,
  margin 1.0 point; merge gates and Release smoke) and full (64 windows,
  two arms, ≈15 min including launches, margin 0.5 point; precision
  default decisions). Reference generation ≈15–30 min per family on one
  GPU, once per set version.
- **Keep:** the probe mechanism, `Reference::find`, compact top-K + tail
  scoring for the quick tier, the PLAN.md passage as a legacy window.
  **Change:** schema to multi-window with role masks and top-32, full-vocab
  f16 rows on sparknest for the full tier, a paired `compare` command with
  the statistics ported from Hugh's MIT `klgate.py`.

## 1. Why today's gate cannot carry a 0.5-point bar

Facts from the code and commits:

| Property | Today | Consequence |
|---|---|---|
| Positions | 512 (PLAN.md prose, `score_from` 1) | binomial SE at p = 0.88 is √(0.88·0.12/512) = **1.44 points**; the "regression" 87.11 → 86.52 is 3 flips, 88.48 → 86.91 is 8 flips. Neither is distinguishable from zero. The V4.1 FP8-head verdict (472 → 465 / 512) rests on 7 flips. |
| Text | our own markdown prose, reference NLL **3.45 nat** | high-entropy text has many near-ties; top-1 is capped near 88% for *any* faithful engine. Hugh's engine reads 95% on a 0.8-nat public panel. The absolute number is a property of the set, not the engine, so "90%" is meaningless until the set is fixed. |
| Content | no chat template, no code, no tool calls, no reasoning, no position beyond 512 | nothing stresses KV/attention numerics, FP8 KDA state over long spans, or the JSON/tool grammar the workload lives in. |
| KL | top-12 + tail bucket, KL(ref ‖ engine) | a coarse-graining that never exceeds the true KL and hides mass shifts inside the tail; fine as a quick proxy, not as the 0.005 bar. |
| Reference | transformers eager, FP32 routed sum, BF16 stream rounding per layer | a different arithmetic from the served kernels; the BF16 arm reads KL 0.043–0.045 to it, so ~0.04 nat of every absolute number is reference/engine mismatch, not quantization. MiMo notes: two valid references agree only 88–90% top-1 with each other. |
| Scoring path | decode-shaped (`probe::score`: prefill to `score_from`, then verify steps of ≤ 8 rows) | right for a C1-decode-first workload; prefill-path numerics (A8 GEMMs, two-lane chunking) are never scored. |
| Comparison | absolute `kl_max 0.15`, `top1_min 0.80` | no paired test; arms are compared by eye on aggregates. |

The design keeps what works (the probe, the golden harnesses, decode-shaped
scoring, compact references compiled into the bench) and fixes the set, the
sample size, the pairing and the KL estimator.

## 2. Metrics: what predicts agentic-coding loss

Agents run greedy (temperature 0) or near-greedy, multi-turn, on long
contexts, and emit tool-call JSON and code edits. A numerics change hurts
exactly when it changes a *confident* argmax, shifts mass onto an invalid
token in a grammar-constrained span, or drifts over a long context. Metrics,
judged on sensitivity to that, cost, and statistical tractability:

| Metric | Measures | Verdict |
|---|---|---|
| **Top-1 agreement** with the reference argmax, teacher-forced | per-step greedy divergence probability | **Carries the bar.** Direct model of greedy decoding. Noise from near-ties is removed by pairing (both arms see the same near-ties) and by the confident-top-1 tripwire. |
| Top-N containment, N ≥ 3 ("reference argmax in our top-N") | nothing about greedy output; ~99.9% for any sane engine | reject as a bar |
| Candidate argmax ∈ reference top-3 | whether our greedy pick is at least plausible to the reference | **Calibrated gross tripwire (§0)**: catches a broken kernel that keeps aggregate agreement; cheap (already have `top`); 99% is V4.1 legacy context. |
| Confident top-1 (positions with reference p₁ ≥ 0.5) | flips that cannot be near-ties | **Calibrated gross tripwire (§0)** and the first thing to read when the bar fails; 98% is V4.1 legacy context, not a universal floor. |
| **Full-vocab KL(ref ‖ engine)** | continuous distance; sensitive to mass shifts near-ties hide; paired differences are precise | **Carries the 0.005 bar** (full tier, full-vocab f16 rows). Quick tier keeps top-32 + tail as a lower-bound proxy. |
| NLL on the fixed text, paired | absolute competence, independent of the reference's arithmetic | **Supporting**: reported per role; tripwire +0.01 nat on human-written code. The one metric robust to reference mismatch (the MiMo lesson). |
| NLL on reference-sampled continuations | ≈ cross-entropy to the reference; redundant with KL | report only |
| Greedy continuation divergence length | chaotic; even two valid references diverge within tens of tokens | not a gate; byte-exactness stays a separate gate for cache restores and speculation |
| Tool-call / JSON validity on the agentic replay | the failure mode that actually breaks sessions | **Tripwire: zero invalid calls** over the recording; no power to see 0.5 points, so not the bar |
| Task pass rate (4 fixture tasks) | end-to-end | smoke only; 4 binary outcomes have no resolution |

**Against what.** Both, with different roles:

- *Paired against our checkpoint-precision arm* (same build, same engine
  path, same windows, same positions, drafts off) is the hard bar. It
  isolates the numerics change: reference/engine mismatch, text entropy and
  near-tie noise cancel because both arms meet them at the same positions.
  This is Hugh's `compare` design and it is what made his 0.002-nat
  decisions possible on 125 windows.
- *Absolute against the family golden* is a floor and a health check: the
  BF16 arm must clear it (engine correctness), the candidate must clear it
  (no regime change), and a drift of the BF16 arm's absolute number between
  releases flags a shared-path regression. It is not where a 0.5-point
  question is answered.

**Which arm is "checkpoint precision".** Weights as the checkpoint ships
them (official FP8 experts with their block scales, or the EXL3/NVFP4
quant under test), BF16 activations and BF16 KDA/head/KV, drafts off,
speculation off, prefix cache off (`cold`). Each family names it in its
reference manifest; the paired comparison refuses two runs whose manifests
name different checkpoints.

**"Top N".** The recommendation to TJ: keep N = 1 as the hard metric
(greedy is what runs), report confident-top-1 and top-3 containment as
tripwires. A top-3 bar at 90% would pass everything and decide nothing.

## 3. The bar, precisely

For arm A (candidate) and B (checkpoint precision) scored on the same set:

1. **Top-1 non-inferiority.** Let n₁₀ = positions where B agrees with the
   reference and A does not, n₀₁ the reverse, n the scored positions.
   δ̂ = (n₁₀ − n₀₁)/n, SE from a window-block bootstrap (B = 5,000) of the
   per-window δ_w. Pass when δ̂ + 1.645·SE < 0.005. McNemar's p is printed
   for the record.
2. **KL non-inferiority.** d_w = mean over window w of KL_A − KL_B (full
   vocabulary, float64 from f16 log-probs on the reference side and the
   engine's f32 rows). Pass when the one-sided 95% percentile-bootstrap
   upper bound of mean(d_w) is below 0.005 nat.
3. **Absolute floor** on each arm: top-1 ≥ 0.90 and mean KL ≤ 0.06 against
   the golden, over assistant-generated positions, every window within
   [0.80, 1.0] (a single window below 0.80 is a localized failure worth
   reading even when the mean passes).
4. **Tripwires**: §2. Failures are reported with the positions, so the
   reader can see whether a flip is a near-tie or an error.

The bar is evaluated on `gen` positions (assistant reasoning, content, tool
calls). `ctx` positions (system, user, tool results, pasted code) are scored
and reported per bucket; they measure context modeling and long-context
numerics but are not what the engine emits.

Both paths matter. The decode-shaped pass (verify steps of ≤ 8 rows, as
`probe::score` does today) is the primary bar because C1 decode is the
workload. The full tier adds a prefill-shaped pass (all rows' logits from
the prefill kernels at the engine's chunking; needs the engine change in
§10) and reports the same two statistics on it; a prefill-only regression
fails the bar too, since every long context is prefilled.

Speculation changes score the verify path with `verify_rows` set to the
draft width, drafts off, so verify-kernel numerics are measured without
acceptance noise; byte-exactness against plain decode remains the separate
gate PLAN.md already specifies.

## 4. The verification set

Per family (tokens differ by tokenizer and by the family's own
continuations), one set version, 64 windows, 512 scored positions per
window = 32,768 positions. A window is a token sequence with a role mask
(`gen`/`ctx`) and a context-length bucket; its scored positions are the last
512 `gen` tokens (fewer plus trailing `ctx` tokens when a turn is short).

| Block | Windows | Context at first scored position | Content and source |
|---|---:|---|---|
| A. Agentic coding turns | 26 | 1K–16K (8 ≤ 2K, 12 in 2K–8K, 6 in 8K–16K) | chat-templated transcripts from `bench-agentic-session.py record` on the fixture repo (and 4–6 added tasks): system prompt, tools, prior turns with reasoning echoed, tool results; scored span = one assistant turn (reasoning, tool-call JSON, diff/content) generated by the model's official API or highest-fidelity local copy, thinking on |
| B. Long-context code reading | 12 | 8K–16K (4 at 16K–32K once the reference supports it) | 1–3 files from this repository (Rust, Python, CUDA) as a user turn, the model asked to explain or modify; scored span = its answer |
| C. Plain repository code | 10 | 0.5K–4K | human-written files from this repository, teacher-forced; all positions `ctx` but reported as the code-NLL tripwire; the only block that measures competence on text the model did not write |
| D. JSON and tool grammar | 6 | 1K–4K | structured-output and tool-selection prompts (our `structured` and `tool_eval` panels' prompts, not the pinned benchmark's answers); scored span = the model's JSON |
| E. Prose and reasoning | 10 | 0.5K–4K | the legacy PLAN.md passage (window 0, 512 positions, kept for continuity), 4 math/reasoning prompts with the model's reasoning scored, 5 model-written explanations of repository design |

Totals: ≈ 400K tokens through the reference (dominated by the long
contexts), 32,768 logit rows kept. Positions beyond 8K: ≈ 9,000 (blocks A
and B), enough to show a long-context-only regression of ~1 point on its
own bucket, and the bucket is printed separately (0–2K, 2–8K, 8–16K, 16K+).

Own-model spans represent the model's reasoning, tool JSON and code distribution,
not another model's output. Official API output need not be exactly greedy or
repeatable: text generation and reference scoring are separate steps. Once the
text is frozen, downstream teacher-forced reference scoring is deterministic.
The fidelity metric remains agreement with the reference argmax, not merely
agreement with the generated text token.

Generation recipe (pinned in the manifest): temperature 0, local greedy decoding,
drafts off, thinking at the family's default, `max_tokens` 1,024 per turn, seed 0
where supported, the agentic loop's tools executed by the fixture runner. The prompt
side of every window is fixed text; only the model's spans differ per
family. Window token ids are pinned in the reference file; the set is
rebuilt only when the manifest version changes.

## 5. Sample size

Notation: p ≈ 0.94 projected agreement, d = discordance rate (positions
where exactly one arm agrees with the reference). Hugh measured d = 4.1%
(974 / 23,625) for an FP8-weight change on one engine; our V4.1 FP8-head
run had 14 / 512 = 2.7%. Use d = 0.04.

- **Unpaired, absolute.** SE = √(p(1−p)/n): n = 512 → 1.05 points
  (0.88 → 1.44); 4,096 → 0.37; 32,768 → 0.13. Two independent arms at
  0.5-point resolution would need ≈ 2·(1.645+1.282)²·p(1−p)/0.005² ≈
  39,000 positions per arm before clustering. Not the design.
- **Paired.** Var(δ̂) ≈ d/n (the (p₁₀−p₀₁)² term is negligible). SE: n = 512 →
  0.88 points; 4,096 → 0.31; 6,144 → 0.26; 16,384 → 0.16; 32,768 → 0.11.
- **Non-inferiority at margin m, one-sided α = 0.05, power 0.90 when the
  true difference is 0:** n = d·(1.645+1.282)²/m² = 8.57·d/m².
  m = 0.5 point → 13,700 positions; m = 1.0 → 3,430. Positions within a
  window are correlated (same turn, same topic): Hugh's clustered design
  effect ran 1.5–2.5. At DE = 2.4, m = 0.5 needs 32,900 → **64 windows ×
  512 = 32,768** (the full tier). At DE = 1.5, m = 1.0 needs 5,100 →
  **12 × 512 = 6,144** (the quick tier, margin 1.0 point).
- **KL.** Hugh's paired per-window SD was 0.009 nat (125 windows) with
  per-row correlation 0.85 between arms. SE = 0.009/√W: W = 64 → 0.0011;
  the bound at a true 0 sits at +0.0019, far inside 0.005. 90% power at
  m = 0.005 needs W ≥ 8.57·0.009²/0.005² ≈ 28 windows; 64 is comfortable,
  12 (quick) gives SE 0.0026 and a bound of +0.0043: marginal, so the quick
  tier's KL margin is 0.01.
- **Power to catch a real 0.5-point loss** at the full tier is the same 90%
  by symmetry; a 1-point loss is caught with > 99.9%.

Fix the panel before scoring. Growing it after an inconclusive result is a
second look the bound does not account for (Hugh's rule; keep it).

## 6. Text provenance and licences

| Source | Used for | Licence position |
|---|---|---|
| This repository's files and history (Rust, Python, CUDA, PLAN.md) | blocks B, C, E | ours |
| `scripts/fixtures/agentic-repo` and the agentic bench's tasks and system prompt | block A prompts | ours |
| The family model's own greedy continuations | all `gen` spans | model licences (GLM, DeepSeek, MiMo: MIT; Qwen: Apache-2.0) permit use and redistribution of outputs; the set can be published |
| Our `structured`/`tool_eval` panel prompts | block D | ours; the pinned `tool-eval-bench` answers are not copied into the set |
| Claude Code / Codex transcripts under `~/.claude/projects`, `~/.codex/sessions` (0.8 GB) | **not used** | third-party model output, local paths and possible secrets; not worth the scrub for a set we can generate |
| `brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits` (licence "other", 31.7 GB panel, raw packed text, no chat template) | GLM Flash absolute cross-check only, 25 windows × 189 rows (2.9 GB) via Hugh's `klgate_fetch.py` (MIT) | read locally, never redistributed; it answers "are we in the published regime" (K4 ≈ 0.025 nat, top-1 0.95) and quantifies our golden's mismatch component. Not available for other families. |
| Hugh's `harness/klgate.py` statistics (block bootstrap, clustered SE, McNemar, `compare`) | ported into the bench | MIT; credit repo, tag v1.1.0 and file in the commit, per PLAN.md |

No public web corpora are needed. If TJ wants a human-written code block
larger than our repository, The Stack v2 permissive subset or
`bigcode/the-stack-smol` is the fallback; not required for v1 of the set.

## 7. Cost

**Reference generation** (one GPU, layer at a time). Per-layer time is
dominated by loading and dequantizing that layer's experts (MiMo MOPD
golden: 70 layers in 295 s for 1.5K tokens, 4.2 s/layer average, 7–10 s on
MoE layers), so running every window of the set inside each layer visit is
nearly free: expert compute for 400K tokens × 8 experts is ≈ 160 TFLOP per
layer (≈ 1–2 s on an RTX PRO 6000), eager attention per window is small
only with bounded query blocks: at 16K, eager [H,T,T] scores plus FP32
softmax temporaries already exhaust GB10 unified memory. Goldens call the
unchanged official eager function on at most 1024 queries against all keys,
slicing the query mask and retaining row-wise arithmetic; unused attention
weights are discarded. Official DeepSeek sparse hooks use 128-query blocks
before their KV gather. GLM KDA keeps its official 64-token recurrent chunks
and bounds independent head groups instead. Small CPU byte-exact gates and
the actual common-prefix qualification precede every full panel. Hidden
streams for 400K tokens x 4 x 4096 x 2 B = 13 GB stay on CPU between layers.

Read each layer once through `/mnt/sparknest`; do not replicate weights for
a one-time golden. GLM Flash BF16 is spread over Spark NVMe and streams via
RoCE (~5 GB/s, ~3 s per ~13 GB layer, about two minutes extra total). Qwen
BF16 is in the scratch archive (~500 MB/s, about 20 minutes extra total).
These are planning estimates, not measurements. Replication is for serving,
which rereads weights at each launch.
The eager per-expert Python loop (288 experts, `index_add_`) is the real
cost: budget 10–30 s per MoE layer at 400K tokens. Estimate: **GLM Flash
(45 layers) 15–30 min, MiMo V2.6 Pro (70 layers) 30–45 min, DeepSeek V4.1
similar**; plus the LM head on 32,768 rows (≈ 50 TFLOP, seconds) and
writing 10 GB of f16 log-probs. Once per family per set version. Measure on
the first run and record it in the commit; batching tokens per expert
across windows is the first optimization if it runs long.

Reference roots use vendor BF16 originals when published: GLM 5.3 Flash
`zai-org/GLM-5.3-Flash-BF16`, Qwen 3.8 `Qwen/Qwen3.8-Flash-Next`, and GLM 5.3
`zai-org/GLM-5.3-BF16`. V4.1, V4 Flash/Pro and MiMo MOPD keep their official
FP8/MXFP4 releases because no vendor BF16 master is published. Manifests pin
root checkpoint, precision, snapshot/config/tokenizer hashes separately
from the text-generation arm. FP8-serve-generated GLM text may remain;
Qwen FP8 serve is wanted, with explicitly labelled EXL3 generation retained
only as an interim fallback. A root change creates a new set/reference hash
and requires fresh prefix qualification, never reuse of an old proof.

The full tier ships reference top-1024 token ids u32 and log-probs f16,
plus f32 tail log-mass and next-token log-prob in safetensors, one HF dataset
config per family/set-version in `wrldsuksgo2mars/cuteafd-fidelity`. The support is
reference-fixed; normalize the 1024 values plus aggregate tail bin together
and evaluate the engine on the same support with one aggregate engine tail.
This is a coarse-grained KL, not mathematically identical full-vocabulary
KL. Saved V4.1 FP8-head decode/prefill paired deltas and upper95 bounds agree
with full-vocabulary results within 1.18e-7 nat (required <=1e-4), with both
PASS verdicts unchanged; f16 entries and f32 tail were included. Full-vocab
rows remain local validation evidence, not a required download. The bench
defaults the full tier (without local `--reference`/`--rows`) to
`wrldsuksgo2mars/cuteafd-fidelity`, config `deepseek_v41-v2_20261005`, revision
`01a0948a62a478a0a9355dd5b57fe4a49bc6cca0`. Explicit `--dataset`,
`--dataset-config FAMILY-VERSION` and `--dataset-revision IMMUTABLE_40_HEX_COMMIT`
override that source; branches and tags are rejected. Future family configs
are added to the same repository and require their own verified revision.
The revision/config/repository and manifest SHA are retained
in run results and must match across arms. It checks the index, manifest,
window, qualification and safetensors hashes before scoring. Engine full-row
dumps remain local and provide the actual omitted mass, avoiding subtraction
of rounded top probabilities. Actual Rust absolute and tripwire gates have
also revalidated the saved compact pair, with both PASS verdicts unchanged.
Publication provenance carries artifact hashes/fork pins and rank topology,
not host paths, addresses or credentials; decoded source text is audited too.
The coordinator handles the public upload. The pinned publication passed
immutable Hub readback of all 72 files (212,253,166 bytes), including the
Hub-added `.gitattributes`; every frozen-file checksum and the manifest hash
chain matched. Manifest SHA256:
`6f2b22f3ed4882765c759b565baab960f7e1563a7fe2be2bbd05540ac4f2f70c`. Set/reference hashes, file hashes,
root/generation provenance and per-checkpoint licence terms accompany each
config. Text is ours, with source-file licence obligations preserved; logits
derive from the named official checkpoint and do not erase its terms.
Quick tier remains top-32 plus tail compiled into `cuteafd-bench`.

**Engine scoring** (figures from GLM Flash on 1 RTX + 2 Sparks: 8K prefill
≈ 5,100 tok/s, decode step ≈ 40 ms):

| Tier | Prefill tokens | Scored positions, decode-shaped | Prefill-shaped pass | Engine time per arm | Wall per decision |
|---|---:|---:|---|---|---|
| Quick | ≈ 30K | 6,144 (768 steps of 8) | no | ≈ 6 s + 31 s ≈ **40 s** | one arm, inside the ≤ 5 min Release smoke |
| Full | ≈ 400K | 32,768 (4,096 steps) | yes (≈ 80 s) | 80 s + 164 s + 80 s ≈ **5.5 min** | two arms + two launches ≈ **15 min** |

Scoring the full tier against 10 GB of reference rows: read 10 GB (sparknest
NVMe or RoCE at 5 GB/s: 2–3 s) plus float64 KL over 32,768 × 155K values;
numpy or Rust, well under a minute. Hugh's pure-Python 37 ms/row is the
ceiling we must beat, not the design.

## 8. Reference mismatch and what the 0.04 nat is

Our BF16 arm reads KL 0.043–0.045 and 87–88.5% to our golden on PLAN.md
prose; Hugh's engine reads 0.025 and 95% to a BF16 teacher on low-entropy
packed text. Two effects mix: text entropy (3.45 vs 0.8 nat) and arithmetic
(eager BF16 attention with an FP32 routed sum vs the served kernels, plus
official FP8 experts vs EXL3). The first run of the new set separates them:
the BF16 arm's absolute number on the model's own text, using only references
we generate from official checkpoints (no external teacher, §11.7). The
initial expectation was KL 0.02–0.04 and top-1 93–96%; if the BF16 arm sits below 92%
on its own greedy text, the golden itself is suspect (routing or norm
differences), and that is a finding about the reference, to fix before any
precision decision.

Absolute agreement also contains a reference-noise component. In the first
V4.1 investigation, the same 576-token prefix evaluated alone versus with a
64-token suffix changed 39/512 argmax rows (7.6171875%). The original
640-token run reproduced exactly, as did the serial versus multi-window
576-token run. The first difference was 12 values in layer-0 `attn.wq_a`
(max 1.52587890625e-5), before compressor/indexer state: sequence-shaped
GEMM rounding amplified through the model, not a demonstrated causal-mask
failure. This does not justify relaxing the absolute floors or the paired bar.

Reference arithmetic now fixes row-wise GEMMs to M=128 with zero-padded
final chunks, including quantized-kernel adapters, HC linears and LM heads;
V4.1 grouped projection/attention einsums use the same fixed query geometry.
Before generating a full family panel, its actual layer-major golden must
pass a fail-closed 576/640-token common-prefix check: all 512 scored f32
vocabulary rows finite and bit-identical. A passing proof is bound to the
family, snapshot identity and pinned set hash, and required by the schema-2
converter. This is a necessary sampled arithmetic/state gate, not a proof
for all sequence lengths or GPU architectures. Other families must pass on
the architecture that actually generates their references; a shared hook
alone does not qualify their attention or recurrent kernels.

The cancelled V4.1 baseline quick decode measured 96.9877% generated-position
top-1, compact KL 0.0111367 nat and 56.3449 s scoring (2,689 generated of
6,144 scored positions, 1 RTX + 4 Sparks, BF16 head, verify width 8). It used
the unqualified shape-sensitive reference and is informational only, not a
calibration or precision gate. Full decode failed on a missing dump parent;
no candidate, full-prefill or paired discordance result exists from that run.

### Qualified V4.1 calibration (2026-10-05)

The first qualified head-off calibration uses the pinned 64-window set
`16f94cfc43ad1c59879b497194cfa6ddeb793cb98f8225206dc0f747ea33cc91`,
checkpoint revision `dba1be0a40aa45a94ad051997016db3960a90277`, daemon
`090a5c3`, one RTX GPU0 and ostrich/dodo/emu/kiwi TP4. Drafts and prefix
cache are off; the decode verification width is 8. "Head off" is the
checkpoint-precision arm, not a claim that the family's native FP4 KV or
FP8 SWA/expert formats are all BF16. The SM120 reference passed its finite,
bit-exact common-prefix gate; full generation took 3203.17 s including
qualification, below the 60-minute stop bar.

All numbers here score actual generated positions, not context padding.
Quick KL is top-32-plus-tail; full KL is full-vocabulary. Scoring times
include requests and row scoring, not model loading or reference generation.

| Tier / shape | Generated positions | Top-1 | KL (nat) | Scoring (s) |
| --- | ---: | ---: | ---: | ---: |
| Quick / decode | 2,689 | 97.1737% | 0.010847 | 55.56 |
| Full / decode | 17,656 | 96.9529% | 0.008547 | 346.31 |
| Full / prefill | 17,656 | 97.0888% | 0.008573 | 262.75 |

| Generated block | Quick top-1 / KL | Full decode top-1 / KL | Full prefill top-1 / KL |
| --- | ---: | ---: | ---: |
| A, agentic | 97.8349% / 0.004431 | 98.4574% / 0.004352 | 98.4392% / 0.004283 |
| B, code reading | 96.8750% / 0.007576 | 96.4030% / 0.008658 | 96.5495% / 0.008471 |
| D, JSON/tool | 96.1353% / 0.031786 | 97.8479% / 0.013373 | 97.2023% / 0.016861 |
| E, prose/reasoning | 97.6562% / 0.009719 | 95.6163% / 0.011955 | 96.1589% / 0.011331 |

Block C has no generated positions. Its context top-1/KL is 95.3125% /
0.014620 quick, 97.9688% / 0.008652 full decode, and 97.7734% / 0.008343
full prefill. Across all context positions, KL is 0.040108 quick, 0.096022
full decode and 0.053880 full prefill. D's appended context dominates this
mismatch (0.605894 / 0.268443 nat on the two full shapes); it is reported
separately and does not become assistant text or enter the generated bar.
D also has the highest generated-span KL in every tier/shape.

Every generated aggregate clears the 92% stop bar, the confident-position
98% and reference-top-3 99% tripwires pass, and the worst generated window
is 94.3359% on either full path. The old 90% floor is comfortably met on
agentic text. Applying §10's whole-percentage-point round-down minus two
literally gives 95% for quick and full prefill, but **94% for full decode**
(96.9529% rounds down to 96%, not up to 97%). V4.1's published `expect`,
shared across its scoring paths, therefore uses `top1_min = 0.94`;
per-path calibration retains 0.95/0.94/0.95. The KL ceiling uses conservative
upward 0.01-nat rounding plus 0.02 nat, capped at 0.06: 0.04 quick, 0.03 on
each full path; V4.1's shared `expect` uses `kl_max = 0.04`. These are
config-specific sanity gates calibrated from V4.1, not cross-family floors.
The common floor remains top-1 >=0.90 / KL <=0.06; the precision-decision
bar remains paired 0.005 top-1 / 0.005 nat on both full scoring shapes.

### First paired V4.1 FP8 vocabulary-head decision (2026-10-05)

The FP8-head candidate uses the same immutable reference, build, checkpoint,
resolved nonprecision settings and GPU0 + TP4 layout as calibration. The
**full tier passes on both shapes**; all absolute gates and statistical
tripwires pass. The quick tier is **inconclusive**, not a demonstrated
regression: its top-1 upper bound exceeds the quick margin, while the
prespecified full-tier bounds comfortably clear the precision bar.

| Tier / shape | Generated positions | Baseline-only / candidate-only agreements | Discordance | Top-1 loss upper 95% | KL delta upper 95% (nat) | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Quick / decode | 2,689 | 56 / 37 | 3.4585% | 0.012199 | 0.001923 | Inconclusive |
| Full / decode | 17,656 | 250 / 269 | 2.9395% | 0.000937 | 0.000846 | Pass |
| Full / prefill | 17,656 | 252 / 255 | 2.8715% | 0.001854 | 0.000305 | Pass |

Candidate scoring takes 55.10 / 346.43 / 255.64 s for quick decode / full
decode / full prefill. Independent agentic replay validity remains an
unfulfilled separate gate; this statistical verdict does not promote a
default. Baseline decode versus prefill has 524/17,656 differing agreement
indicators and 547/17,656 argmax differences; those describe shape
sensitivity, not precision-arm discordance.

The paired bar remains the precision decision rule at any absolute level,
subject to the independent absolute gates and reference qualification.

## 9. Migration: what evolves, what stays

| Piece | Today | Change |
|---|---|---|
| `python/reference/families/*/golden.py` | one text, every layer's streams saved, logits for all rows | `--windows manifest.json`: loop over windows inside each layer visit; logits only for scored rows; f16 log-softmax output; query-chunked eager attention for T > 16K (phase 2); streams saved only with `--layers` (unchanged flag) |
| `scripts/bench/make-fidelity-reference.py` | one golden → `reference/1` JSON, top-12, 512 positions | emits `reference/2`: windows with tokens, role mask, bucket, `score_from`, top-32 ids + f16 log-probs + tail + next; writes the quick subset into `references/<family>.json` and the full rows + manifest to sparknest |
| new `scripts/bench/fidelity-set.py` | — | builds the set manifest: renders prompts through the served BF16 arm (probe returns `prompt_ids` and generated ids), applies the block recipe of §4, pins tokens and roles; `--family`, `--version` |
| `rust/crates/cuteafd-bench/src/reference.rs` | `Reference` (single window), `score` → `Fidelity` | `Reference` gains `windows: Vec<Window>` (schema 2; schema 1 read as one legacy window); `score` per window and per role/bucket; new fields: `top3_contained`, `confident_top1`, `agree_text`, per-position records retained for pairing; full-vocab KL when the sparknest rows are present (`CUTEAFD_FIDELITY_ROWS` dir) |
| `baseline.rs::fidelity` | one probe request, absolute pass/fail | quick tier: iterate windows, same probe spec per window (`cold`, `score_from`, `want` top-32 + token), absolute floor + tripwires; `expect` from the reference file |
| new `cuteafd bench fidelity` | — | `run --tier quick|full --arm NAME --out run.json` (every position's agree/KL/NLL, engine line, checkpoint id, path shape); `compare A.json B.json [--top1-margin 0.005 --kl-margin 0.005]` with the §3 statistics; exit 0 pass / 3 fail / 1 error |
| engine probe (`cuteafd-api::probe`, `cuteafd-daemon::shared::probe`) | rows carry top-k + wanted ids | add `dump_rows: Option<PathBuf>` (f32 or f16 log-softmax rows to a safetensors file, streamed per step, for full-vocab KL); add `verify_rows` override for speculation-width scoring; phase 2: `prefill_rows_logits: bool` returning every row's logits from prefill chunks (the LM head on all rows of the chunk, as Hugh's `score_each`) |
| `references/*.json` (7 families) | 512-position PLAN.md | regenerated as schema 2 with the legacy passage as window 0; the old thresholds (`kl_max 0.15`, `top1_min 0.80`) become per-window sanity bounds, the new floors live in `expect` |
| `release-smoke` | quick quality = this check | unchanged shape; the quick tier is what runs |
| External teacher | not used | no external dataset; references generated from official checkpoints only (§11.7) |

The legacy converter remains byte-compatible for identical input logits
(same tokens, positions, top-12 subset of top-32): converting the old raw
640-token golden reproduces the shipped JSON. This proves the converter,
not the old reference arithmetic. The old V4.1 JSON scores a 576-token
prefix of a 640-token run; evaluating those same tokens at length 576
changes 39/512 argmax rows, and the qualified fixed-M128 legacy differs
from the shipped JSON at 48/512 rows. The old reference is arithmetically
unqualified. Requiring a shape-invariant golden to reproduce it byte for
byte would preserve the defect, so that regeneration-equality gate is
retired, without relaxing the absolute floors or paired decision rule.

Replacement reference gates are: (1) the actual finite, bit-exact
common-prefix proof bound to the family, snapshot and set; (2) conversion
of the old raw logits reproduces the old JSON, retaining converter
compatibility; and (3) a newly generated qualified legacy window is
published as window 0 of schema 2. Publish schema-2 replacements after
reference validation, without engine calibration. Regenerate the old schema-1 V4.1
JSON from qualified logits when the bench switches over; retain the old
raw evidence for diagnosis, not as a certification target.

## 10. Historical implementation plan (2026-10-05)

This records the original implementation campaign, not the current publication
workflow. The 2026-10-18 policy supersedes its engine-calibration steps. Current
workflow: generate own text, build original-quality logits, convert, validate/seal,
finalize using declared policy floors, review, publish and re-pin. No engine run.

Branch `work/fidelity-v2` from `origin/work/p0`, worktree, no GPU until step 5. Follow
AGENTS.md (builds under `~/.cache/cuteafd/builds/fidelity-v2`, locks around
hardware runs, small commits, tables in commit messages).

1. **Schema and scorer (host only).** `reference.rs`: schema 2 types, legacy
   loader, per-window/per-role scoring, new metrics, position records.
   Port block bootstrap, clustered SE, McNemar and `compare` from
   `~/Developer/hugh/glm53f-afd/harness/klgate.py` (MIT, v1.1.0; credit in
   the commit). Unit tests: perfect copy → KL 0, agreement 1; shifted copy
   fails; `compare` of identical runs passes with δ̂ = 0; a synthetic 1-point
   regression on 32,768 positions fails, on 512 passes (documents the
   power). Gate: `cargo test -p cuteafd-bench`.
2. **Set builder.** `scripts/bench/fidelity-set.py` per §4: reads the
   agentic recordings and the task list, repository file lists, panel
   prompts; needs a served BF16 arm (uses the probe for `prompt_ids`,
   greedy generation, drafts off); writes `set/<family>/<version>/
   windows.json` (tokens, roles, buckets, recipe, sha256). Script tests
   with a fake server fixture. Gate: `pytest -q scripts/tests` adds none
   failing.
3. **Reference generation.** `golden.py` multi-window loop and scored-row
   logits for V4.1 Flash first (TJ's anchor; the V4.1 FP8-head decision is
   pending), then MiMo V2.6 Pro, MiMo V2.6 Flash MOPD, Qwen (GLM Flash is
   Hugh Madden's now); `make-fidelity-reference.py` schema 2 and the
   sparknest manifest. Reference gates: finite bit-exact common-prefix
   qualification; old raw logits still reproduce the shipped legacy JSON
   (converter compatibility); regenerate and publish the qualified legacy
   window in schema 2. Old-versus-new golden equality is retired for the
   arithmetic reason and measured evidence in §9.
4. **Engine probe.** `dump_rows`, `verify_rows` override; family `serve.rs`
   call sites pass them through (`deepseek_v4`, `glm5`, `glm5_flash`,
   `mimo_v2`, `qwen4`). Gate: loopback test writes rows whose top-k equals
   the in-band `ProbeRow.top`.
5. **First hardware run (V4.1 Flash, 1 RTX + 4 Sparks, `sparks.lock` then
   `gpu1.lock`).** Serve the BF16 arm, build the set, generate the
   reference (record the per-layer times), score BF16 and the single-GPU
   FP8 KDA/head arm on both tiers, run `compare`. Deliverables in the commit
   message: the absolute numbers per block and bucket, δ̂ and bounds for
   both paths, the discordance rate d actually observed (feeds §5). No
   external teacher cross-check (TJ, 2026-10-05). Each family/config derives
   its own per-path `expect` from repeated default-precision baselines:
   top-1 minimum = agreement rounded down to a whole percentage point
   minus 2 points, never below 0.90; KL maximum = KL rounded up to
   0.01 nat plus 0.02 nat, capped at 0.06. Use the lower repeated top-1
   and higher repeated KL for each path, and the least restrictive of
   those path-specific gates for the config's shared `expect`. Label it
   "calibrated from this config's baselines". Keep the common 0.90/0.06
   floor and paired 0.005/0.005 decision bar separate. Repeatability
   qualification of a reference is not a precision-default verdict.
6. **Wire the tiers.** Quick tier in `baseline.rs`; `cuteafd bench
   fidelity run/compare`; Release smoke unchanged in shape. Re-run the
   current `release-smoke` matrix entry for GLM Flash to show the ≤ 5 min
   budget holds.
7. **Phase 2 (separate brief).** Prefill-path all-rows logits in each
   family's prefill (Hugh's `score_each` pattern: LM head on the requested
   rows of the chunk), query-chunked eager attention in the goldens for
   16K–32K windows, the 4 long windows of block B.

Stop bars for the agent: if the BF16 arm's absolute top-1 on its own greedy
text is below 92%, stop and report (reference problem, §8); if reference
generation exceeds 60 min for V4.1 Flash, stop and report the per-layer
profile before optimizing.

## 11. Decisions (TJ, 2026-10-05)

1. Yes: precision defaults use the paired 0.5-point / 0.005-nat
   non-inferiority bounds; the 90% is an absolute floor calibrated on the
   new set.
2. Both paths in the full tier; decode-shaped alone in the quick tier.
3. Yes: greedy spans generated by our BF16 arm.
4. Cap the first set at 16K; 32K windows in phase 2.
5. Publish `set/` and the compact references with the repo.
6. Yes: re-run the V4.1 FP8-head decision and the GLM Flash split-FP8
   default on the full tier before v2 freezes precision defaults.
7. No external teacher: `brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits` is
   brandonmusic's dataset with licence "other"; we use only references we
   generate from official checkpoints.

## Original open questions (answered above)

1. **The bar is paired.** Agree that the hard 90% is an absolute *floor*
   calibrated on the new set, and that precision defaults are decided by the
   paired 0.5-point / 0.005-nat non-inferiority bounds? (The alternative,
   an absolute 90% on the current PLAN.md set, cannot be met by the BF16
   arm and cannot resolve the question.)
2. **Decode-shaped only, or both paths, for the hard bar?** Recommended:
   both in the full tier, decode-shaped alone in the quick tier.
3. **Greedy spans from our BF16 arm.** Acceptable as the generator of the
   model-written text (the bar is on reference-argmax agreement, symmetric
   between arms)? The alternative, a full-model transformers run, does not
   fit our hosts.
4. **Long context.** Cap the first set at 16K and add 32K windows in phase
   2 (chunked eager attention in the goldens), or hold the set until 32K
   is in?
5. **Publishing the set.** The set is clean to publish (our text, model
   outputs). Publish `set/` and the compact references with the repo, or
   keep them local like benchmark runs?
6. **Re-litigating past decisions.** The GLM Flash split-FP8 "fail" (8 flips
   of 512) and the V4.1 FP8-head "fail" (7 flips) are within noise on the
   old set. Re-run both on the full tier before the release's precision
   defaults are frozen?
7. **Hugh's teacher** carries licence "other". Fine for a local, unpublished
   cross-check of GLM Flash only?
