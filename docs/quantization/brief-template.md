# MoE quantization brief template: direct MC, MUL1, rolling recovery

**Every quant brief starts from this file.** To write one, copy it whole, keep every section, and append a **Model addendum** (§14) with only what's specific to the checkpoint. The addendum never restates or weakens a template rule. If a model truly needs an exception, the addendum names the rule and gives the reason.

These rules come from past runs: Inkling 2, Kimi K3/K3.1, GLM 5.3/Flash, ML4, Muse, Step-5, MiniMax, V4.1 and Hy4. Each **Why** names the run that paid for it. Most were found mid-run, and cost a stopped quant, a restart or a thrown-away WIP. **They're the required design, not later optimizations:** build them in from the first line of code. Survey evidence and conflicts: the #2511-era quant-learnings sweep (2026-11-19).

**What not to copy.** `quantization/inkling2/output_io.py`, `io_resume.py` and the expert-output parts of `inkling2/run.py` (≈ 311–366) carry an "ASCII dragon" warning from TJ: *wrong design, do not copy* (`4a5cdcaf17`, on `quant/hy4-flash-k4-mc`). They're a compatibility shim for a frozen run.
- **Safe to copy:** Inkling's route spool and recovery (`storage.py`, `run.py` `router()`), its activation segments with XXH3 (`2a95c7c8af`), the MC exporter (`export.py`), and the metadata builders.
- **Not safe to copy:** its expert-output spool.

## 0. Operating context (fill in per run)
- **Who runs it:** TJ, in Codex on maxq. Run from the engine checkout `~/Developer/cuteafd` at the latest **origin/dev**, with recursive submodules, or a task worktree of it. Local `dev` checkouts go stale, so always fetch. Quant code lives under `quantization/<family>/`.
- **Quantizer:** GPTQModel is the submodule `third_party/gptqmodel` (remote `tpurtell/GPTQModel`).
  - Make quantizer changes there, merge them to **fork main** and push, then bump the engine's submodule pin and `third_party/gptqmodel.lock.json`, and encode from that.
  - Production refuses a dirty fork, or a fork that differs from the pin.
  - Check fork main with `git ls-remote`, not a local clone.
- **Upload:** approved in the brief, with the full lifecycle (§10).
- **Decisions go to TJ, not to main.** Report anything that changes numerics before acting on it. When TJ says stop testing, run only the production arm, with no extra A/Bs.
- **Re-verify every architecture fact** in the brief against the complete download before starting. **Why:** briefs written mid-download have been wrong: the Hy4 drafter KV layers, its Engram and LatentMoE naming, and Inkling's 2× expert-size estimate.

## 1. Recipe: direct MC, one causal pass
- **Package:** one MC package, schema `cuteafd-mc-v1`. The authoritative spec is `docs/mc-checkpoints.md`; read all of it first. It holds shared routed experts plus three indexed views:
  - `original`: the source core byte for byte, in its native dtype;
  - `k8-compact`;
  - `k6-compact`.
- **Every MC family must serve original, k8, k6 and fp8-compact uniformly.** A view the engine refuses is a release gap.
- **Routed experts:** EXL3 at the addendum's bit width (K4 by default), encoded once and shared by every view.
- **Core: two encodes from one Hessian.** Each core module is quantized to **both K8 and K6 from the same Hessian**.
  - Downstream layers, experts and the drafter see the **K8 reconstruction**. K6 is a second encode with no separate pass.
  - Record `upstream_activations: k8-compact` in `quantize_config.k6.json`.
  - Core widths are exactly {4, 6, 8}.
- **Bundled drafter** (MTP, DFlash, dSpark or DFly): it rides the core views, calibrated in the same pass from the K8-core target's hidden states, exactly as the reference runs it.
  - `original` restores the native drafter **byte for byte**. **Why:** published original views once carried a K4 MTP and needed repair (`c5b1e4aeab`).
- **Write the MC package directly into one output directory.** No intermediate plain packages, staging copies or server-side parent copies.
- **FP8-origin sources aren't MC** and never get a quantized core. Start from the BF16 original whenever one exists.
- **MLA attention: which projections are core.** In every core view (K8 and K6), quantize **`q_a_proj`, `q_b_proj`, `kv_a_proj_with_mqa` and `o_proj`** (plus any gate projection the architecture has, such as `linear_gate`). **`q_b_proj` is a core module: quantize it.** The only MLA projection kept native is the absorbed **`kv_b_proj`**. **Why:** the two `_b` projections sit side by side, and a past run read "absorbed `kv_b_proj`" as covering `q_b_proj` too and left it native. The DSA indexer's own `wq_b` is a different tensor: it belongs to the indexer and stays native with it.
- **Keep native, byte for byte, in every view:** embeddings, router weights and biases, all norms, sinks, the DSA/compressed indexer (including its `wq_b`, `wk`, `weights_proj` and `k_norm`), the absorbed MLA `kv_b_proj` (**only `kv_b_proj`; `q_b_proj` is quantized as above**), recurrent/KDA pieces, vision towers, the drafter's special heads, and any lookup tables (see the addendum).

## 2. EXL3 contract: MUL1 codebook
- **Codebook (TJ 2026-11-19):** **MUL1** (multiplier `0x83DCD12D`) for every EXL3 tensor, experts and core K8/K6 alike.
  - Write `.trellis`/`.suh`/`.svh`/`.mul1`, with `out_scales: auto`.
  - Whole projections record side 1.
  - A padded head records logical vs stored N, padded to 128, with zero padding scales.
- **Quantizer support:** GPTQModel already handles MUL1 (`gptqmodel/exllamav3/modules/quant/exl3_lib/quantize.py`, `codebook_mul1_mult`, `mul1` in `quant_args`).
- **Record the codebook** in `quantize_config*.json`, `cuteafd-mc.json` and the model card's recipe table.
- **Engine status:** the engine serves MUL1 for both the core and the routed experts (routed-expert MUL1 since #2538, with the SM120/SM121 decode duel winners from fork #420/#421 pinned in #2581). The loader reads the codebook from the per-tensor `mcg`/`mul1` marker (`cuteafd-loader/src/plan/format.rs`).
- **Don't use an E4M3-rounded codebook at K6.** **Why:** it failed all 20 pilot tensors at 2.28–2.39× NMSE.

## 3. Calibration data
- **Single source of truth (TJ 2026-11-28):** `quantization/panel_corpus.py` owns the pinned panel (repo, revision) pairs, component order, multi-turn renderer, and corpus loader. New quants (new families, QMLA, and future re-quants) call `quantization/panel_corpus.py` directly (`prepare_panel_corpus`, `load_token_batches`, `load_component_records`) rather than duplicating panel pins or corpus code.
  - **Frozen loaders stay untouched:** completed/in-flight runs whose frozen identities hash their own corpus modules — Hy4, Hy4-flash, ERNIE 5.5 Flash (PR #2816), and all v1 families (`kimi_k3`, `glm5`, `glm5_flash`, `inkling2`, `step5`, `muse_spark`, `mistral_large4`, `corpus_inputs.py`) — keep their own loaders untouched for reproducibility.
  - **Corpus for new quants: `wrldsuksgo2mars/agentic-quant-panel-v3@e3ac6ee6126f469e37ddf879a6cb2dccc10c6c99`** (`V3_PANEL`, the default), all three components. Its `agentic` and `exl3_standard` files are byte-identical to v2's; it adds `long_context`. A full run (`full=True`) on v3 with only two components fails closed.
  - **Past runs: `wrldsuksgo2mars/agentic-quant-panel-v2@54683731eb01c7c2fa5b801a0888e029a0e1b1c7`** (`V2_PANEL`, `agentic` + `exl3_standard` only) stays pinned for frozen reproductions; pass `panel_revision=V2_PANEL.revision` and existing v2 manifests still load and verify.
- **Corpus components and canonical order:** every full future quant (MC, QMLA, or other) calibrates on all three components from that single pinned revision, **in this exact order**:
  1. **`agentic`**: the v1 panel, byte for byte (`calibration.jsonl`, `screening.jsonl`, `heldout.jsonl`, `manifest.json`, the parquets). That's 1,426 prompts, about 1.08M tokens, records of 512–1,024 tokens covering code/agentic, multilingual wiki, math, reasoning-termination and structured/function-call traffic.
     - **Rendering:** single-turn chat-rendered (`add_generation_prompt=True`), one unpadded prompt per batch, as below.
     - **Splits:** calibration, screening and heldout, disjoint by group and prompt hash.
     - **Frozen loaders:** earlier v1 and v2 family loaders (`kimi_k3/prepare_corpus.py`, `hy4_flash/prepare_corpus.py`, etc.) keep their frozen reproduction paths untouched; new runs call `quantization/panel_corpus.py` directly.
  2. **`exl3_standard`**: the ExLlamaV3 standard calibration data (C4, code, multilingual, technical, wiki, tiny stories, plus random-token rows), pre-split into ordered segments (`exl3_standard/segments.jsonl`, `data/exl3_standard.parquet`).
     - **Packing:** pack it with this checkpoint's tokenizer using `reproduction/pack_exl3_standard.py` (contract `exl3-standard-pack-v1`), at the defaults of 250 rows × 2,048 tokens, about 512K tokens. That reproduces ExLlamaV3's `get_default_calibration` exactly: source order, row counts, alternating BOS/EOS article packing, raw-stream slices and seed-0 random rows.
     - **Raw text, never chat-rendered.** The packed rows are fixed-width, one row per batch, so §3's no-packing rule applies to `agentic` and `long_context` (whose windows never pack across records).
     - **Calibration only:** it has no screening or heldout split. Random rows aren't language, so they never go into held-out metrics or diagnostics.
  3. **`long_context`**: 16K and 32K multi-turn **agentic traces** (with tool calls, tool results, and optional reasoning) plus long-document QA and repository audits (`long_context/16k.jsonl`, `long_context/32k.jsonl`, `long_context/manifest.json`, schema `agentic-quant-panel-v3-long-context-v1`). About 16 records, about 352K tokens total: 12 calibration records (~256K tokens: 8 × 16K + 4 × 32K) and 4 heldout records (~96K tokens: 2 × 16K + 2 × 32K). Records store model-neutral `messages` (`{role, content, reasoning_content?, tool_calls?, tool_call_id?, name?}`) plus optional per-record `tools` schema, never token IDs.
     - **Multi-turn rendering (`panel_corpus.render_long_context_record`):** render in **one pass** with **this checkpoint's own** tokenizer and chat template (`tokenizer.apply_chat_template(messages, tools=tools, tokenize=True, add_generation_prompt=False, padding=False, truncation=False, ...)`), never re-encoding rendered text.
     - **Fail closed on tool support:** if the checkpoint's template cannot express `tools`, `tool_calls`, or `role="tool"` results (or silently drops them), fail rather than flattening them into plain text.
     - **Reasoning in history turns:** pass `reasoning_content` untouched so the checkpoint's own template applies its exact native behavior (most drop history reasoning before the final user turn and keep the final turn's). Expose the thinking/effort mode via the same chat kwargs as `agentic`.
     - **Lengths:** record each record's rendered token count under the checkpoint's tokenizer. Fail if a record falls outside its declared length class (`16k`: 11,468–22,118; `32k`: 22,937–44,236) or exceeds the model's `max_position_embeddings` / context limit.
     - **Weight quants use 2,048-token windows (`long_context_mode="windowed"`, the default of `prepare_panel_corpus` and the loaders).** Each record is rendered once as above, then its rendered token stream is cut into consecutive 2,048-token windows (like `exl3_standard`'s 2,048-token rows). A final partial window of at least 256 tokens is kept; a shorter one is dropped and recorded as `dropped_tail_tokens`. One unpadded window per batch, never packed across records. Windows keep their record's split, so heldout windows never enter calibration. **Why:** full-length records exceed the activation store's 256 MiB record cap, Hy4's 2,048-row runner cap and eager T×T attention in Kimi/Inkling/GLM; windows fit every family pipeline unchanged.
     - **QMLA long-context evaluation uses full-length records** (`long_context_mode="full"`): one unpadded record per batch. A prepared corpus loads only in the mode it was prepared in.
     - **Splits:** honor `calibration` and `heldout` splits strictly (disjoint by group, source identity, and content hash, including against `agentic`). Heldout never enters calibration.
  - **Totals:** `agentic` (~1.08M) + `exl3_standard` (~0.51M) = ~1.59M, plus `long_context` (~0.35M including heldout; ~0.26M calibration, slightly less after dropped tails) ≈ **1.85M calibration tokens**. Activation and Hessian accumulation time and host-bank memory scale with the total.
  - **Frozen identity fields:** record one panel revision (`dataset.repo_id`, `dataset.revision`, `dataset.sha256sums_sha256`), `component_order`, `calibration_mode`, and per-component `count`, `tokens`, rendered `input_ids_sha256`, and `token_stream_sha256` (plus `long_context.heldout` counts/hashes when active) in `manifest.json` and the frozen run identity. `long_context` also records `mode`, `window_tokens` and per-record `sources` (rendered length, `input_ids_sha256`, window count, `dropped_tail_tokens`); its `input_ids_sha256` is over the windowed stream.
- **Rendering (`agentic`):** use **this checkpoint's own** tokenizer and chat template (`add_generation_prompt=True`). One unpadded prompt per batch, in dataset order. No truncation, padding, sorting or packing. Freeze the template date.
  - **Never re-encode rendered text.** Re-encoding erases special-token permissions (`kimi_k3/prepare_corpus.py:143`).
  - Never reuse another model's token IDs.
- **Thinking/effort mode:** state it explicitly in the addendum, as a value the checkpoint's template actually accepts. Calibrate in the mode the engine serves by default (`agentic` and `long_context` both receive it). Kimi, GLM and Inkling used effort `max`; Hy4 used `high`.
- **Long-context coverage and parity witnesses:**
  - Before `long_context` is published (2-component runs), the longest calibration prompt is ~2,048 tokens (`exl3_standard`), so long attention paths aren't exercised in the Hessians and separate `hessian_eligible=False` long-context witnesses verify parity past the SWA window or sparse top-k cutoff.
  - Once `long_context` is published (3-component runs), its traces enter the calibration stream as 2,048-token windows: real multi-turn agentic content, but each window is still a 2,048-token row, so long attention paths stay as in 2-component runs. Keep the `hessian_eligible=False` long-context witnesses for parity past the SWA window or sparse top-k cutoff.
- **Pilot slices** use real prompt **tails**, not prefixes, because prefixes are mostly the shared template preamble (`f48ad65c5b`). A pilot may run a subset of components (`--subset`), while the full run requires all three components once published.
- **Transformers:** pin the family-matched version, or the reference PR head, and record it in provenance.
- **Preflight:** the calibration forward matches the reference implementation on a few prompts, covering every special path, before any Hessian is captured.

## 4. Routed-expert coverage: low-row recovery (required)
**Every routed expert gets at least 1024 Hessian observations.**

**Why:** at 384–512 experts per layer, some experts get few or **zero** natural rows. Hy4's run **stopped at layer 26, on an expert with no natural calibration rows** (`"Hy4 layer 26 expert 456 has no natural calibration rows"`), and its first brief left this out. Inkling handled it: of 18,432 experts, 327 needed candidates and 134 needed the identity fallback.

**Never raise on, or silently skip, a low-row expert.** Fill it in this order:
1. **Natural rows:** the reference router's native top-k routes, in corpus order.
2. **Near-expert candidates:** if it's still short of 1024, add rows where this expert ranks **k+1..2k** by the reference router's own ranking score. The bias and correction terms affect *ranking* exactly as the reference applies them.
   - Fill rank-major, then corpus order, until the expert reaches 1024. A row never appears at two recovery ranks.
   - **Weight each candidate with renormalized hypothetical gamma:** substitute it for the last native route (rank k), recompute the reference's routing normalization over that hypothetical set, then apply the routed scaling factor.
3. **Identity fallback:** if it's still short, add `missing × I` to the **raw XᵀX, before count normalization**. That's contract `raw-XTX-missing-times-I-equivalent-to-normalized-2I` (`inkling2/storage.py:240`). It keeps count above 0, so GPTQModel's uncalibrated `q_fallback` path (`exl3_lib/quantize.py`, `count==0` or `diag_mean<1e-20`) can never trigger for a routed expert.

- **Reference implementation (copy and generalize):**
  - `quantization/inkling2/run.py` `router()` (≈ 230–246);
  - `quantization/inkling2/storage.py` `RouteSpool` (≈ 47–112, 213–247), with `minimum=1024` and `CANDIDATE_GAMMA_CONTRACT = native-last-top8-substitution-renormalized-v1`.
  - **The Inkling code hardcodes top-8:** `ids[:, :7]`, the 8-wide sum, and `_routes_valid` shape `[:,8]`. Generalize it to the model's top-k and routing function (softmax or sigmoid, biases, scaling).
- **Recovery ranks by top-k, from past recipes:** k=8 uses ranks 9–16 (GLM, GLM Flash, Step-5, Muse), k=16 uses 17–32 (Kimi), and k=4 uses 5–8 (ML4).
- **Provenance per expert:** natural rows, augmented rows with their ranks, identity rows, effective rows, and a selection hash. Record it in the receipts and the per-module CSV. The pilot reports the distribution before the full run.
- **Routed experts only.** Dense FFNs and shared experts use `core_identity_rows=0`.
- **Known disagreement:** for down-projection Hessians, V4.1 used unit route weight, and Kimi used none. Use the renormalized candidate gamma above unless the addendum justifies otherwise.

## 5. Numerics
- **Covariance:** FP32 products bounded to 128 rows, a persistent **FP64** outer sum, and FP32 storage (`fp32-products-k128-fp64-sum-fp32-storage-v1`). **Why:** Muse's FP32 outer sum gave 7.97e-6 matrix error and an indefinite Hessian; fixed to 2.85e-8 by `4ac29e1e2c`.
- **Factorization:** checked. **On failure, fix the input. Never retry with more damping.** `sigma_reg=0.025` (× `diag_mean`) is the only regularizer, with `seed=787`. The Hadamard congruence runs in FP64.
- **TF32 off,** including NGC's environment overrides: `NVIDIA_TF32_OVERRIDE=0`, `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0` (`3977fc5671`).
- **Propagate the serialized reconstruction:** the packed trellis/suh/svh decoded to BF16, **never the search's raw `weight_q`**, which differs by 6.2e-4. Down Hessians see the serialized gate and up.
- **Fail closed** on non-finite values at every stage boundary. Stop any K6 encode whose training error exceeds 1e-3 before installing it.
- **Finiteness checks stay on the GPU, batched per activation chunk,** with one device-to-host scalar per chunk. **Never** a GPU→CPU boolean per product, module or element. **Why:** Hy4 synced per product until the fork's `d606eaccc` (TJ: "we don't keep the check on device, we download it to check over and over?"). The fix was 1.08–1.16× faster and bitwise-identical.

## 6. Activation and expert-output dataflow (required design)
- **Accumulate routed-expert outputs on the GPU.** The weighted top-k combine is computed on the device as an ordered FP32 reduction. Only the **summed** per-layer output feeds the next layer or the causal checkpoint.
  - **Never persist unsummed per-expert outputs.**
  - **Why:** Hy4 copied Inkling's unsummed-output spool. That was **100.5 GiB written plus 100.5 GiB read back per routed layer**, and it hung maxq. `495873eaa` replaced it with about 12.6 GiB of resident buffers.
  - On resume, replay the sealed packed experts in ascending expert order. Don't reload stored outputs.
- **Keep routed inputs resident** for the layer when they fit, otherwise stream bounded chunks. Never bounce through host memory per expert.
- **Discrete GPU (maxq, RTX): when the calibration inputs don't fit on the GPU, use one host bank and one bounded GPU staging buffer.** Keep the layer's calibration inputs once in host RAM, as a single FP32 bank that the CPU route selection indexes. Gather each expert's rows into **one reusable pinned host chunk**, and copy them into **one reusable GPU row bank** sized to a fixed budget (8 GiB for full Hy4, `HostExpertStaging` in `quantization/hy4_flash/storage.py`). An exceptionally hot expert streams the same ordered chunks again. The ordered top-k sum stays on the GPU.
  - **Don't allocate per expert,** per chunk or per held-out pass: no fresh `torch.empty`, no new pinned buffers, no `.to(device)` copies of the whole bank.
  - **Don't keep a second device copy of inputs the host bank already holds.**
  - **Why:** full Hy4 (256 experts, hidden 6144) first ran with device-resident inputs plus per-pass temporaries. It wasted memory and regressed badly in speed until #2551 moved the 36.84 GiB FP32 input bank to host RAM with an 8 GiB staging bank (`b1704cd406`, `0e1f90263b`).
- **Unified memory (GB10 Sparks): no host bank and no staging.** On GB10 the CPU and GPU share one DRAM pool, so a host bank plus a GPU staging copy just holds the same rows twice and adds a copy pass. Instead:
  - **One device-allocated input bank.** Allocate the layer's calibration inputs once as a CUDA tensor (`cudaMalloc`, i.e. an ordinary `device="cuda"` tensor) and gather expert rows from it directly on the GPU. The calibration wavefront, the routed-row selection and the ordered top-k accumulation all stay resident together, with no host round trip.
  - **Never pinned or host-mapped memory for gathered rows.** `cudaHostAlloc`/`pin_memory` buffers use small-page coherent mappings, and a scattered row gather from them measured **4.9× slower** than from `cudaMalloc` on the same GB10 DRAM (rhea, 2026-10-29). Pinned memory is only for a bulk transfer, never a gather source.
  - **Budget by `MemAvailable`, not `cudaMemGetInfo` or `nvidia-smi`.** CUDA allocations reclaim clean page cache, so `cudaMemGetInfo` under-reports and `nvidia-smi` shows N/A. Count the input bank, the output accumulator, the live Hessians, the current layer's expert weights and the workspace against `MemAvailable`, leaving about 10 GiB for the OS and the source page cache. Never drop caches.
  - **If the whole working set doesn't fit,** shrink the live set before reaching for staging: fewer experts' Hessians resident at once, or BF16 inputs with FP32 accumulation, if the pilot proves that's exact against the FP32 path. Don't reintroduce a host↔device copy; on GB10 it's the same memory.
  - **CPU work:** GB10 has 10 performance and 10 efficiency cores. Size CPU-side parallelism to about 10, detected at runtime, not `nproc`.
- **Reuse workspaces across held-out and gate passes.** Held-out projections and gate checks reuse preallocated temporary buffers with **the same GEMM shapes and FP32 operation order** as the original pass, so results stay bit-identical. Never re-allocate them per pass.
  - **Why:** in Hy4, the held-out workspace alone added 288 MiB per 2,048 rows. Reuse cut it to 112 MiB, with no slowdown (4.12 → 4.05 ms per warmed sample) and identical gate evidence (`0e1f90263b`, `22c105992a`).
- **Prove a memory change is exact before production uses it.** Host-bank placement, staging and workspace reuse must match the resident CUDA path **byte for byte**: raw Hessians, serialized K4 bytes, recovery replay, held-out sums and gate evidence (`9b0308f1c6`). Bind the change to the frozen run with a recovery receipt (`host_input_recovery.py`, `529bd6fa06`), never as a silent change.
- **Record memory as the run goes.** Log CUDA allocated and reserved bytes and the trellis workspace count at each expert-group completion. A run that only reports wall time hides a memory regression until it OOMs or slows down.
- **Read each source weight once,** and rely on the page cache. Source reads go through SparkNest at about 1 GB/s on maxq's 10G link, so don't be wasteful.
- **Caching:** cache only the native non-expert weights for the current phase, and clear the cache at phase boundaries. Retire stages and stores after the downstream commit.
- **Huge lookup tables** (Engram, PLE): mmapped and reclaimable, gathering only the rows the calibration tokens hash to. **Never materialize them.** Watch anonymous memory, not just RSS.
- **Three-component panel sizing:** weight quants see `long_context` as 2,048-token windows (§3), so no family pipeline needs 32K support; per-record activation, attention and `StageStore` segment sizes match `exl3_standard` rows. Budget host-bank + GPU `combined` buffers for ~1.85M total calibration tokens on maxq (80 GiB container / 96 GiB VRAM) and cassowary (~118 GiB `MemAvailable`). Only a pipeline that runs full-length records (QMLA long-context evaluation) needs tiled query-row or SDPA/flash attention rather than dense eager `[H, S, S]` (`32 × 32K² × 4 B = 128 GiB`).

## 7. Recovery and I/O (required)
- **Log-style recovery:** append records to **bounded sequential segments**. The fork uses 32 batches per segment, capped at 256 MiB (`9820e963a`); `long_context` windows are 2,048-token records like `exl3_standard` rows, so they fit that cap. **Fsync each segment once when it closes,** then atomically publish a small per-phase or per-layer manifest. On resume, replay the incomplete tail.
  - **Never scatter writes** into a big preallocated or mmapped file. **Why:** Inkling's 90 GiB token×rank scatter memmap was flushed and fsynced after every expert, under an 80 GiB container cap, at **107 s per expert**. Sequential staging made it 4.8 s (`0220f41abc`). That shim is compatibility-only, so don't copy it.
  - **Never fsync per record, expert or tensor.** **Why:** `2a95c7c8af` cut payload fsyncs from 12 to 3 on a 12-record fixture, with grouped activation segments.
- **Cheap hashing:** XXH3-128, **streamed in the write pass,** for every scratch and recovery receipt. **Never read a file back just to hash it.** **Why:** XXH3 runs at 21.75 GiB/s against SHA256's 2.11 GiB/s (`cb9ed4c2ee`, `8eb5a68a5b`).
- **SHA256 only where a format requires it:** the `cuteafd-mc.json` digests, the Hub/LFS identity and the source identity. Compute those in the write pass too.
- **Frozen identity:** cover the source, corpus, numerical-file hashes and fork pin. The numerical files are committed, and the fork is clean and equal to the pin.
  - Refuse resume on any mismatch.
  - Require a diagnosis before resuming a failed producer.
  - Never restart a completed job.
- **Mid-run changes that don't touch numerics** (I/O, scheduling) go in as reviewed recovery receipts bound to the frozen identity, never as silent numeric changes. Hy4's `resident-expert-recovery.json`, `batched-hessian-recovery.json` and `paced-trellis-recovery.json` are examples.
  - Build §5–§8 in from the start, so you don't need a chain of them.
- **Infra faults:** diagnose, then resume. Never restart from scratch. **Why:** SparkNest EIO hit Inkling at layers 20 and 24, and both resumed.

## 8. GPU, host and throughput on maxq
- **maxq's GPU is also the WSL display GPU.**
  - Don't flood the PCIe/driver queue with huge host↔device ping-pong. During Hy4, the host hung (screen, keyboard, ssh, gpu-powerd) with `dxgkio_escape` floods in `dmesg`.
  - Structure work like Kimi's: bounded spans, resident buffers, few syncs.
  - **No sleep-based pacing.** Hy4's `GPTQMODEL_EXL3_SPAN_PAUSE_MS=1` (`43180de6c`) had no measurable benefit; only a host reboot cleared the hang (TJ 2026-11-19). Don't add pacing, and remove that knob from the fork. If the host wedges, stop cleanly, report it, and let TJ reboot; then resume.
- **Memory and GPU access:**
  - **Memory:** 94 GiB RAM with no swap, so cap containers at about 80G. Page cache from mmapped scratch counts against the cap.
  - **GPU access:** request GPUs via CDI, not `--gpus`, because `daemon-reload` kills `--gpus` containers. Keep timeouts in the supervisor's process group.
- **Hub upload throughput:**
  - **Publish over HTTP LFS (`HF_HUB_DISABLE_XET=1`), through the shared resumable publisher (`quantization/hub_lfs.py`).** Xet is not used for publication: `hub_requirements.py` keeps the hub 1.33.0 pin and refuses to publish unless Xet is disabled, and the publish units set `HF_HUB_DISABLE_XET=1`.
  - **Tuning:** start at 16 workers (`--upload-workers`, files uploading at once) and batches of 32–64 files (`--upload-batch-files`, default 48), and tune toward a sustained 900–950 Mbps on the gigabit link. Use the CLI flags, not env vars. Hub 1.33.0 parallelizes across files, but each file's multipart parts go up serially. A batch therefore ends when its largest file does, and a few very large shards (Hy4-flash's 29.5 GiB Engram tables, say) leave the tail on that many streams. The publisher starts the largest shards first.
  - **Measured:** Xet from Asia topped out at about 300 Mbps (`HF_XET_FIXED_UPLOAD_CONCURRENCY=32` gave 299 Mbps). Parallel LFS at 8 workers gave about 800 Mbps (Hy4 K4 MC, maxq, 2026-11-25).
- **Don't repeat known dead ends:** for example, page-cache hint trials (released 65 GiB, no speedup).
- **Progress reporting:** an update every 30 min via a durable timer, with remaining time and an Asia/Taipei ETA. Exclude the interrupted layer from averages, and allow separately for head/MTP, upload and placement.

## 9. Output and validation
- **MC contract:** follow `docs/mc-checkpoints.md` exactly. Earlier packages needed post-publication repairs (`quantization/mc_repair_*.py`), because of:
  - incomplete `tensor_storage`: use the shared builder, with quant format, bpw, codebook multiplier, in/out/stored features and stored tensors (`85d080f37f`);
  - view declarations not mirrored into the metadata (`05d890fbcf`);
  - nested text-block metadata not rewritten per view (`0fd3199183`, Kimi K3.1 #2098);
  - integer base bits lost in mixed experts (`e0f0901430`);
  - mixed core and expert shards blocking clean copies (`5a4a6e0e4b`);
  - per-view config not referencing its own index;
  - unsorted or non-unique `core_tensors`;
  - mismatched shared-inventory digests;
  - K6 names leaking into the K8 index;
  - the original index naming trellis tensors.
- **Readers and writers:** tolerant readers ignore unknown fields, and writers and validators stay strict. Anything that changes how weights are interpreted goes in `required_features`, or bumps the schema. Advisory sections (like Hy4's `engram_fp8`) don't.
- **Derived alternates** (precomputed tables, for example): put them in dedicated shards that no standard view indexes, and verify every row bit-exact against the vendor's own expression, including ±0.
- **Reuse:**
  - `quantization/mc_package.py` (writer and validator);
  - `core_export.py`, `core_launch.py`;
  - `mc_publish.py`, `mc_postpublish.py`;
  - `quantization/tests/test_mc_*.py`;
  - the Inkling MC-direct exporter `quantization/inkling2/export.py`, with `9e71bb5320` and `05d890fbcf`.
- **Model card (`README.md`):** generated by the exporter.
  - **Template:** the **GLM 5.3 MC card** (`wrldsuksgo2mars/GLM-5.3-EXL3-K4-MC`), not Inkling's, which still has boilerplate TJ rejected.
  - **Front matter:** `library_name: cuteafd`, the source's real license with the LICENSE file kept, `base_model`, and tags `exl3`, `mixture-of-experts`.
  - **Sections:**
    - "What multi-core means", with a view table of sizes and when to use each view;
    - "How it was made";
    - a Recipe table: codebook and multiplier, bits, seed, sigma, Hessian accumulation, ordering, replay, calibration count, low-row recovery policy;
    - "Running it";
    - "License".
  - **Describe the panel accurately:** "token-budgeted real source code, worked mathematics, function-call schemas, multilingual text, ExLlamaV3 standard raw text, and (when calibrated on the three-component revision) 16K/32K multi-turn agentic traces and long-document QA, as 2,048-token windows".
  - **Leave out** repair narratives and "no quality claim" boilerplate.
- **Validate before upload** with the loader's own checks: discovery, every view's index, metadata hashes, fingerprints, shared-digest equality, and original-view core bytes equal to the source. Also run a plain-loader check of `original`.

## 10. Publishing lifecycle
- **Before publishing:** release the GPU. Run publication as a separate, CPU-only unit (`af7528e1b7`).
- **Sealing:** predeclare `.gitattributes` in the sealed package as a regular file, because the Hub appends LFS rules. **Why:** Inkling's publication omitted it, and its staging used a symlink.
- **The lifecycle:**
  1. **Upload** as `wrldsuksgo2mars/<Model>-EXL3-K<bits>-MC` over HTTP LFS (`HF_HUB_DISABLE_XET=1`) through the shared resumable publisher (`quantization/hub_lfs.py`; §8 has the tuning). Each batch is one parent-guarded commit, verified against the Hub before the next one starts. An interrupted transfer resumes automatically from the verified remote tree and never sends a committed file again. A sealed-receipt size or SHA256 mismatch, or a source that changes mid-batch, refuses the publication.
  2. **Verify** by size plus the LFS sha256 the Hub already reports. **Reuse the export's SHA256s:** the writer's streaming receipts supply the LFS object ids, so nothing rereads weights for hashing. No extra hash passes, no hashing over 10G, and no re-downloading your own quant. (Older briefs said otherwise, and this supersedes them.)
  3. **Stage** a hard-link HF cache with the Hub OIDs as blob names, then run `nest hf import --offline --no-copy`. Update `refs/main`. Pinned revisions get `refs/keep/<name>`, written with `printf %s`, with no trailing newline.
  4. **Place it, before deleting anything.** Use SparkNest only: `nest plan --place hf:<repo> --host raptor:d2 --host raptor:d3 --host ostrich --host dodo --host emu --host kiwi --host rhea --host moa --factor 1/8`, then `nest plan show` (check for removals), then one `nest plan apply --wait`.
     - Random-read tables get a pin rule to raptor first.
     - Never `nest replicate --hosts`.
     - Check coverage with `nest ls --json`.
  5. **Clean up:** delete the source export, staging and intermediates. Keep receipts, logs and the per-module CSV.
- **Hub folder limits:** bundle large diagnostic sets to stay under them (`5b15c6d5d3`).

## 11. Preflight and pilot gates (report before the full run)
*(The addendum may replace the separate pilot with production gating, as Hy4 does. When it does, the preflight below still runs, and the full run checks the same bars on its first layers of each kind, stopping on any miss.)*
- **Preflight:** a full header and source audit over the complete download, with a fail-closed inventory and byte totals. Then the reference-forward match on a few prompts, covering every special path.
- **Pilot:** representative layers (first dense, first MoE, a mid layer of each attention type, the last layer, and the drafter target). Arms per core module are K8, K6 and keep-native; experts run at the addendum's bits. Keep it under about 15–30 minutes of GPU time. The pilot may use a subset of corpus components/records (`--subset`), while **the full run must use all three components (`agentic`, `exl3_standard`, `long_context`) once published** (`full=True` fails closed on two components). Report:
  - held-out NMSE per arm;
  - scale-inclusive bpw;
  - view sizes;
  - a TP2 projection;
  - the §4 recovery distribution;
  - anything odd, with a keep-native option.
- **Bars:** held-out NMSE K8 ≤ 1e-4, K6 ≤ 1e-3, expert ≤ 1e-2, all finite (`inkling2/pilot.py:21`). For reference, Inkling's worst values were K8 1.41e-5, K6 1.86e-4 and K4 4.55e-3.

## 11a. QMLA cache calibration (MLA-capable families only; non-MLA families skip)
*(Normative format contract: [`docs/quantization/qmla-format.md`](qmla-format.md), `cuteafd-qmla-v3`. Design & runtime contract: [`docs/quantization/qmla-design.md`](qmla-design.md), #2909. Non-MLA families—GQA, GDN, MSA, CSA/HCA—skip this section completely. Existing MC weight encoding, core views, `0:0` cache behavior, and publication rules stay unchanged.)*

- **Reference implementation to copy:** Do not write a standalone QMLA pipeline. Reuse the shared family-agnostic library in **`quantization/qmla/`** (`capture.py`, `quantizer.py`, `fit.py`, `measurement.py`, `sidecar.py`, `pipeline.py`, `long_eval.py`, `publish.py`) and copy the thin family adapter pattern from **`quantization/hy4_flash/`** (`qmla_capture.py`, `qmla_run.py`, `qmla_publish.py`). A new MLA family only needs its own `QmlaCaptureAdapter` (layer geometry, RoPE pairing / M-RoPE / NoPE convention, and exact attention capture hooks).
- **Eligible layers & default recent-token protection:** Full-history MLA layers only. **SWA rings, KDA/GDN recurrence, HCA/CSA layers, and DSA indexer keys are untouched by QMLA.** Recent-token protection (`sink4_local128`: first 4 sink + last 128 recent tokens exact) is **enabled by default** during both calibration and serving on full-history MLA layers only (SWA untouched; `--protection none` is only the measured alternative if protection later proves unnecessary). Pure-NoPE MLA families (`qk_rope_dim=0`) calibrate content profiles (`C:0`) only and never emit RoPE profiles.
- **Corpus (panel_corpus pin, never Standard Fidelity):** `quantization/panel_corpus.py` owns the pin (§3): `wrldsuksgo2mars/agentic-quant-panel-v3@e3ac6ee6126f469e37ddf879a6cb2dccc10c6c99`. **QMLA fits on all calibration rows of `agentic`, `exl3_standard` and windowed `long_context`, split 60 / 30 / 10 train / val / test. Heldout full-length `long_context` records are the separate `--long-eval` set.**
- **Capture, fitting, selection, and validation contract:** follow `docs/quantization/qmla-incremental-template.md` and `docs/quantization/qmla-format.md`. In short:
  - capture once on the pinned MC checkpoint’s `original` view through the MC loader, using its own core and stored experts (plain checkpoints use their own weights). Capture the native weighted cache latent $c$ (or exact RMSNorm operands), `q_nope`, post-RoPE `q_rope`/`k_rope`, exact attention masks (causal / sparse DSA top-k), output gates, learnable sinks, `post_proj_norm`, and post-`o_proj` FP32 `0:0` reference outputs, persisted per layer with streamed XXH3-128 receipts (`save_layer_capture`). Reuse captures across widths and release each layer's activations before advancing;
  - fit and quantize the cached latent $c=\mathrm{BF16}(u\Gamma)$ at a native BF16 boundary. Apply the active kind's transform after weighted RMSNorm (`identity`: no-op; `scale_only`: $c \odot s_C$; `full`: $c R_C S_C$) and compensate `kv_b_proj` accordingly on every view;
  - for sparse replay on pure-NoPE MLA models, keep the full native key dimension in each query tile and verify the exact selection mask;
  - **Fitting budget & candidate persistence:** fit **both `full` and `scale_only`** candidates at every calibrated width (`c2, c4, c8` and `r4, r8`, including 8-bit) alongside step-0 SmoothQuant inits and `identity`. Persist every candidate (`full`, `scale_only`, `smoothquant`) under `<output>/fit-checkpoints/` (`cuteafd-qmla-fit-checkpoint-v3`, XXH3-128 receipts) plus the full val/test score table and training-width provenance in `<output>/candidate-table.json` (`cuteafd-qmla-candidate-table-v3`) so interrupted runs resume automatically per layer and selection can be rerun via `reselect_from_persisted_candidates` before publishing. After verified publication and SparkNest import/pin, **delete** unselected candidate fits, capture banks, and intermediate scratch, retaining only the small JSON receipts and logs;
  - **Per-width selection & explicit kinds:** select one transform per `(layer, path, serving_width)` on `val` (`select_per_width`), classify every winner into `identity` | `scale_only` | `full` (`1e-6` rotation/scale collapse tolerances), store zero tensors for `identity` and `.s`-only for `scale_only`, and alias byte-identical F32 transforms across widths per `(path, layer)` (`qmla-format.md` §3);
  - pilot and full run measure all nine combinations including `2:0` (pure-NoPE: `8:0, 4:0, 2:0`);
  - calibrations are fitted and validated on the MC `original` view only and apply to every core view of that checkpoint (`applicable_views: ["all"]`);
  - full-length `--long-eval` runs in memory-bounded record groups (sized from CUDA free memory on discrete GPUs or `MemAvailable` on unified-memory GPUs; defaulting to 1 record per group when unsure) before the final headline table is assembled;
  - every reported number is measured on `test` / full-length `long_context` only, **leading with ONE combined short-vs-long headline table** for deployment modes **`4:0`, `2:4`, `8:0`, `8:8`, `4:8`** (one joint end-to-end post-`o_proj` measurement per mode showing Short Identity / Short Shipped / Short Impr. %, Long Identity / Long Shipped / Long Impr. %, Long/Short improvement ratio, and short per-layer spread median/best/worst, with `:0` caption and explicit `n/a` + reason for NoPE models; per-position-bucket breakdown `0-2k` through `32k+` stays in `long_context_eval` below the headline), followed by the full candidate cross-table, identity (RTN), and fixed Hadamard (measurement baseline only, never shipped);
  - every fitted combination is supported with no quality cutoff gate; each carries `attn_rel_err` (mean relative post-`o_proj` attention-output error against `0:0`) so serving recommendations can be chosen from the reported numbers.
- **Unattended end-to-end execution:** Run in ONE `systemd-run --user` resume-on-restart unit chaining preflight → capture → fit → select → sidecar → validation → `--long-eval` → headline table → dry-run publish check → live publish (approved in the brief) → SparkNest import/pin → post-publish cleanup → final report (combined short-vs-long headline table, new revision SHA, sidecar size, wall time, cleanup result). Every phase gates automatically and stops only on a failed bar or integrity violation.
- **Outputs and MC packaging (single sidecar file, no `mla_calibration.json`):**
  - Write **only** `mla_calibration.safetensors` (F32 transforms + self-describing `__metadata__["cuteafd_qmla"]` header, `schema: "cuteafd-qmla-v3"`, `artifact_status: "calibrated_unqualified"` until matched runtime fidelity/perf gates pass, `protection_policy: "sink4_local128"`, `protection_scope: "full_history_mla_only"`) at the package root, **never indexed** in any view's `model*.safetensors.index.json`. There is **no separate `mla_calibration.json`**.
  - For MC checkpoints, embed the full `"qmla"` manifest section inside `cuteafd-mc.json` via `attach_qmla_to_mc` (**never** in `required_features`; `shared_inventory_sha256` and all view fingerprints stay identical).
  - Validate with `qmla.validate_sidecar` (schema `cuteafd-qmla-v3`, payload + whole-file SHA256 digests, agreement between `cuteafd-mc.json["qmla"]` and `__metadata__["cuteafd_qmla"]`, zero orphan tensors, canonical kind consistency, `width_fingerprint`/`profile_fingerprint`/`transform_fingerprint`, geometry, orthogonality $\|R_C^\top R_C - I\|_\infty \le 10^{-5}$, invertibility, byte-identical quantization-off identity replay, and expanded K/V algebraic relative L2 $< 10^{-5}$) and `mc_manifest.validate`. Include the model-card note and combined short-vs-long headline table from `docs/quantization/qmla-format.md` §9 in `README.md`.
- **Adding QMLA to an already-published checkpoint:** Use the standalone incremental workflow in `docs/quantization/qmla-incremental-template.md` (same-repo new revision, zero shard re-uploads).

## 12. Hardware and run hygiene (TJ's runs)
- **No cluster locks or leases:** no `hwlock.py`, `build-slot.sh`, `maxq.lock`, `gpu*.lock` or seed leases. (Inkling ran under a maxq lease; this rule supersedes that.)
- **Durability:** `systemd-run --user --expand-environment=no --setenv=PATH --setenv=HOME`, detached. `nohup setsid` dies with the session's cgroup. Chained units need a failure path that wakes the owner, and resume-after-reboot units are gated on `boot_id`.
- **venv:** use a task venv under the run root. **Never use or modify `~/Developer/cuteafd/.venv` on maxq,** not even by symlinking it into the worktree.
- **Storage:** work storage on `~/Developer`, never `/mnt/scratch`. Stream the source from the HF cache.
- **Secrets:** the HF token goes over stdin only.
- **Process hygiene:** no `pkill -f`, no `git stash`, and never print full process argv.
- **Off limits:** don't touch power settings, raptor's GPUs or the serving Sparks.

## 13. Code delivery and report
- **Nothing lives only on maxq.** Push every green step. Code that exists only in a local checkout or a run directory is lost work: the next quant can't reuse it, and main can't review it.
- **Engine code:** `quantization/<family>/`, `mc_package.py` changes, docs and tests go on a branch, with a PR against **cuteafd `dev`** (`scripts/dev/pr-sync init`). Main reviews and merges; the quant agent never merges.
  - **Merge incrementally.** Once a part is proven (its tests pass and the run uses it), main merges it to dev during the run. Shared fixes (I/O, recovery, numerics) especially shouldn't wait for the quant to finish. The final PR carries what's left: the export, the card and the receipts.
- **GPTQModel changes:** merged to **`tpurtell/GPTQModel` main** and pushed *before* the encode that uses them. The engine branch bumps the `third_party/gptqmodel` pin and `third_party/gptqmodel.lock.json` to that main commit in the same PR. Production refuses anything else.
  - **Removals count too.** Deleting a knob or a dead path is a fork-main commit plus a pin bump, like any other change.
- **Before the final report,** list the open PRs, the fork-main commits and the pin state: dev's pin vs. fork main, from `git ls-remote`.
- **Report** on a GitHub issue:
  - per-layer time and total;
  - sizes per view;
  - the per-module error table (K8/K6);
  - the §4 recovery distribution;
  - repo@sha.

---

## 14. Model addendum
*(Append: source and pinned revision; reference code in authority order and the pinned Transformers version; architecture; size estimate; top-k and routing function (for §4); thinking mode (for §3); core module list (for MLA, state explicitly that `q_b_proj` is quantized); keep-native list; special tensors and their formats; model-specific calibration traffic; pilot layers; extra bars; the published name; and the engine follow-ups this checkpoint needs.)*
