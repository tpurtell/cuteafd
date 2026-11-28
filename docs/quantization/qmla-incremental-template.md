# Incremental QMLA calibration template: add calibrated MLA sidecar to an existing checkpoint

**Every incremental QMLA calibration brief starts from this file.** Given a `MODEL_ID` and an immutable `SOURCE_REVISION` (MC or plain/non-MC), it adds a validated `cuteafd-qmla-v3` sidecar (**one** file: `mla_calibration.safetensors`, with **no separate `mla_calibration.json`**) as a **new revision of the same Hugging Face repository**, preserving every existing weight shard, view, index, tokenizer, and asset with **zero weight shard re-uploads**.

Copy this file whole, keep every section, and fill in **§9 Model addendum**. Never weaken a template rule.

**Authority & reference code to copy:**
- Normative tensor, math & sidecar format contract: [`docs/quantization/qmla-format.md`](qmla-format.md) (`cuteafd-qmla-v3`).
- Design rationale & runtime architecture: [`docs/quantization/qmla-design.md`](qmla-design.md) (confirmed 2026-11-28, [#2909](https://github.com/tpurtell/cuteafd/issues/2909)).
- Full MC quant rules (corpus, recovery, I/O, hub hygiene): `docs/quantization/brief-template.md`.
- **Shared family-agnostic QMLA library (reuse directly, do not duplicate):** `quantization/qmla/` (`capture.py`, `quantizer.py`, `fit.py`, `measurement.py`, `sidecar.py`, `pipeline.py`, `long_eval.py`, `transfer.py`, `publish.py`).
- **Working reference pipeline to copy/adapt:** `quantization/hy4_flash/qmla_capture.py`, `quantization/hy4_flash/qmla_run.py`, `quantization/hy4_flash/qmla_transfer.py`, `quantization/hy4_flash/qmla_publish.py`.
  - For a new MLA family (starting with GLM 5.3 Flash), write only a thin family adapter implementing `QmlaCaptureAdapter` (layer geometries, RoPE/NoPE/M-RoPE convention, exact attention capture hook) plus thin CLI entrypoints copying `hy4_flash/qmla_{run,transfer,publish}.py`.

---

## 0. Operating context and scope
- **Target:** Existing checkpoint `MODEL_ID` at immutable 40-char `SOURCE_REVISION` (MC or plain/non-MC — **do not convert a plain checkpoint to MC**).
- **Same repository only:** Publish as a **new parent-guarded commit on `MODEL_ID`**. Never create a new HF repo for a calibration-only update.
- **Single sidecar file, no `mla_calibration.json`:**
  - **MC checkpoints:** Rewrites `cuteafd-mc.json` with the added optional `"qmla"` section (never in `required_features`; existing `variants` and `shared_inventory_sha256` unchanged) and adds `mla_calibration.safetensors`.
  - **Non-MC (plain/official) checkpoints:** Adds **only** `mla_calibration.safetensors` (self-describing via `__metadata__["cuteafd_qmla"]`; never creates `cuteafd-mc.json`).
- **Zero re-quantization / zero shard re-uploads:** Existing routed experts, dense cores, view indexes, configs, tokenizers, vision towers, drafters, and native assets stay **byte-for-byte identical** (`reuploaded_shard_bytes == 0`). Never re-encode cores or experts.
- **No `GPTQModel` fork changes:** All QMLA code lives in `cuteafd` under `quantization/qmla/` and `quantization/<family>/`.
- **`0:0` and `--kv-cache` compatibility:** Users not passing `--quant-mla` / `QUANT_MLA` see zero behavioral change. `--kv-cache` and `QUANT_MLA` compose: `--kv-cache` sets the default format for every cache (SWA rings, indexer keys, MLA), and `QUANT_MLA` overrides only the full-history MLA path(s) set nonzero.
- **SWA / KDA / HCA / indexer exclusion & default `sink4_local128` protection:** QMLA targets **full-history MLA layers only**. SWA rings (e.g., Hy4-flash's 30 SWA-512 layers), KDA/GDN recurrence, HCA/CSA layers, and DSA indexer keys are untouched. Recent-token protection (`sink4_local128`: first 4 sink + last 128 recent tokens exact) is **enabled by default** in both calibration and serving on full-history MLA layers only (never SWA); `--protection none` is only the measured alternative if protection later proves unnecessary.
- **Calibrated vs. runtime-qualified artifacts:** Until runtime Phases 3–5 qualify the checkpoint on matched Standard fidelity + long-context + performance tests, the manifest **must** record `"artifact_status": "calibrated_unqualified"`. Never label an untested calibration artifact as runtime-qualified.

---

## 1. Preflight: inspect checkpoint architecture and inventory
Before capturing any activation:
1. Verify local snapshot at `MODEL_ID@SOURCE_REVISION` (`config.json` SHA256, `cuteafd-mc.json` if MC, `model*.safetensors.index.json` headers).
2. Enumerate and verify:
   - Exact eligible full-history MLA layer IDs (excluding SWA, KDA, GDN, HCA, CSA layers).
   - Per-layer geometry: `heads`, `kv_lora_rank`, `qk_nope_dim`, `qk_rope_dim`, `v_head_dim`, `rms_norm_eps`, `attention_scale`.
   - RoPE convention: `interleaved` (Hy4-flash, Hy4, GLM 5.3, Mistral Large 4, ERNIE 5.5 Flash MLA-DSA), `split_half`, `unrotated_pe` (Kimi K3/K3.1), or `none` (`qk_rope_dim=0`: GLM 5.3 Flash all MLA layers; Mistral Large 4's 18 NoPE layers).
   - Multimodal RoPE axes (`mrope_section`, e.g. `[12, 10, 10]` on Hy4/Hy4-flash) and YaRN / LeRoPE / scalable-softmax parameters.
   - Cross-layer KV latent sharing (`kv_source_layers` + `shared_consumers`, e.g. Mistral Large 4 layer 35) so shared-KV consumers optimize one source `(R_C, S_C)` jointly.
   - Drafter cache ownership (target-KV-sharing DFly2 vs independent DFlash2 / `mtp_dspark`).
3. Verify the family's `QmlaCaptureAdapter` reproduces the pinned reference attention forward on synthetic/preflight prompts, including 3D M-RoPE positions (`T != H != W`) for VL models and sparse DSA top-k pruning past `index_topk`. Require exact masks and reference relative L2 below `2e-5` for ordinary preflight prompts. For the 2,080-token layer-3 GLM Flash sparse case, tile queries across the full key dimension and compare both native and replay against the same FP64 sparse attention. Replay must be at least as close to FP64 as native, and its relative difference from native must be below 1% of the measured 8-bit error.

---

## 2. Calibration corpus (pinned panel revision + fixed known component set, never Standard Fidelity)
`quantization/panel_corpus.py` owns the panel pin, component order, renderer and loader (`brief-template.md` §3). QMLA resolves the panel through `panel_corpus.resolve_panel_policy` (full = not `--pilot`) and never pins its own revision. Never use the held-out Standard Fidelity panel.
1. **Default (full) calibration: `wrldsuksgo2mars/agentic-quant-panel-v3@e3ac6ee6126f469e37ddf879a6cb2dccc10c6c99`, all three components in order:**
   - `agentic`: rendered with **this checkpoint's own** tokenizer and chat template.
   - `exl3_standard`: packed with this checkpoint's tokenizer (`exl3-standard-pack-v1`).
   - `long_context`: 16K/32K multi-turn traces rendered once with the checkpoint's template, cut into 2,048-token windows (`long_context_mode="windowed"`); heldout records never enter fitting.
   - `--corpus-dir` is a `panel_corpus.prepare_panel_corpus` output (`python quantization/panel_corpus.py ...`), loaded and verified through `panel_corpus.load_component_records`. A full run with fewer than all three components fails closed.
2. **Pilot:** `--pilot` may pass a subset with `--components` (always including `agentic` and `exl3_standard`).
3. **Past runs:** `wrldsuksgo2mars/agentic-quant-panel-v2@54683731eb01c7c2fa5b801a0888e029a0e1b1c7` (two components) stays pinned for frozen reproductions via `--panel-revision 54683731eb01c7c2fa5b801a0888e029a0e1b1c7`.
- The exact `dataset_repo`, `dataset_revision`, and components used are recorded explicitly in `calibration_provenance`.

---

## 3. Pilot first (required before full run)
Always run a short pilot first and report its results to TJ before the full run. The pilot uses a few representative MLA layers and a small corpus slice (16 / 4 / 4 train / val / test), with default `--protection sink4_local128`. It is for understanding, so it fits every width and measures all nine combinations, just like the full run (pure-NoPE families: `8:0,4:0,2:0`).

**Reference pilot command (Hy4-flash):**
```bash
MIMALLOC_PURGE_DELAY=0 PYTHONPATH="$PWD/quantization:$PWD/third_party/transformers/src" \
  "$VENV/bin/python" -m hy4_flash.qmla_run \
  --checkpoint "$SNAPSHOT_DIR" \
  --source-repo "$MODEL_ID" \
  --source-revision "$SOURCE_REVISION" \
  --view original \
  --corpus-dir "$CORPUS_DIR" \
  --output "$WORK_ROOT/pilot" \
  --pilot
# --pilot defaults to: --layers 0,21,41 --modes 8:8,4:8,4:4,2:4,8:0,4:0,2:0,0:8,0:4 --protection sink4_local128
#   --train-sequences 16 --val-sequences 4 --test-sequences 4 --steps 100
```

**Pilot gates (must pass before full run):**
- `validate_sidecar` passes with `status == "passed"` and `protection_policy == "sink4_local128"`.
- Every pilot layer satisfies orthogonality $\|R_C^\top R_C - I\|_\infty < 10^{-5}$, strictly positive finite scales, and quantization-off identity replay ($R_C=I$, $S_C=I$) byte-identical to captured native post-`o_proj` output. Require matched FP32 expanded K/V relative L2 below `1e-5` for learned content transforms. Record native attention-output drift separately.
- Report the full test cross-table (every per-width fit and SmoothQuant init against every combination), the shipped transforms' test error per combination against RTN, fixed Hadamard and SmoothQuant†, and which candidate each layer shipped.
- Wall time per layer and peak GPU/host RAM logged to forecast full-run ETA (Asia/Taipei).

---

## 4. Full calibration (unattended, streaming per-layer, resumable)
Run the full calibration in **ONE `systemd-run --user` resume-on-restart unit** (`Restart=on-failure`, `RestartSec=30`, non-restarting exit code on hard gate failures) chaining **preflight → capture → fit → select → validate → `--long-eval` → K8 transfer → headline table & sidecar emit → dry-run publish audit**, with **automatic gates** and **one confirmation pause just before live publish**. Once confirmed, chain live publish → SparkNest import/pin → post-publish cleanup → final report.

**Reference full command (`hy4_flash` pattern):**
```bash
MIMALLOC_PURGE_DELAY=0 PYTHONPATH="$PWD/quantization:$PWD/third_party/transformers/src" \
  "$VENV/bin/python" -m hy4_flash.qmla_run \
  --checkpoint "$SNAPSHOT_DIR" \
  --source-repo "$MODEL_ID" \
  --source-revision "$SOURCE_REVISION" \
  --view original \
  --corpus-dir "$CORPUS_DIR" \
  --output "$WORK_ROOT/full" \
  --layers all-dsa \
  --modes 8:8,4:8,4:4,2:4,4:0,0:4,8:0,0:8,2:0 \
  --protection sink4_local128 \
  --train-sequences 128 \
  --val-sequences 16 \
  --test-sequences 16 \
  --steps 300 \
  --long-eval
```
- **Pure-NoPE models (e.g., GLM 5.3 Flash):** pass `--modes 8:0,4:0,2:0`. Any `R > 0` mode is omitted and recorded under `unsupported_modes_reason`.
- **Recovery & I/O (`brief-template.md` §7):** Layer captures (`captures/layer-XXXX.{safetensors,json}`) and per-layer candidate checkpoints (`fit-checkpoints/*.{safetensors,json}`) are written sequentially with streamed XXH3-128 receipts bound to the frozen config/source/corpus identity so a crash resumes automatically on restart without manual intervention.

---

### Capture, fitting and validation contract (both templates)
- **Primary forward.** Load the pinned MC checkpoint’s `original` view through the MC runtime. Use the core and all experts from that view. The upstream BF16 repo in `cuteafd-mc.json["source"]` is provenance only. For a plain checkpoint, load its own weights. Record the source revision and view. Use this forward’s own `0:0` outputs as the reference for fitting, validation and test.
- **Content coordinates:** fit and quantize the checkpoint's native weighted cache latent $c$ ($c=\mathrm{BF16}(u\Gamma)$ at a BF16 output boundary). Use $z_C=c R_C S_C$ (or $c \odot s_C$ for `scale_only`, or $c$ for `identity`) on writes and the matching inverse in consumers, including target-sharing drafters. Every view applies the transform after weighted RMSNorm.
- **Three disjoint splits:** `train` fits, `val` selects fit steps and per-width transforms, `test` produces every reported number (full 128 / 16 / 16, pilot 16 / 4 / 4). Split IDs and hashes are recorded in provenance and the fit identity.
- On PyTorch builds with mimalloc, set `MIMALLOC_PURGE_DELAY=0` before the Python process starts so freeing a capture returns its anonymous pages to the OS. With Docker, pass `-e MIMALLOC_PURGE_DELAY=0` to the container. Log `qmla.capture.layer_memory_bytes(device)` after releasing each layer. On GB10, use `system_total - system_available` to track unified memory pressure alongside process RSS and CUDA allocations.
- **Capture once:** capture each sequence once on the MC `original` view and reuse its activations across every bit width.
- **Fitting budget & candidate persistence (kept during the run, deleted after publish):**
  - Fit **both `full` (rotation + scales) and `scale_only` (fixed identity rotation / zero angles, log-scales only)** for **every** calibrated width (`c2, c4, c8` and `r4, r8`, including 8-bit).
  - Persist **every** fitted candidate (`full`, `scale_only`, and step-0 `smoothquant` per width and layer) under `<output>/fit-checkpoints/` (`cuteafd-qmla-fit-checkpoint-v3`, XXH3-128 receipts) and write the full val/test score table + training-width provenance to `<output>/candidate-table.json` (`cuteafd-qmla-candidate-table-v3`). Interrupted runs resume automatically from these checkpoints, and selection policies can be rerun offline via `reselect_from_persisted_candidates` with zero refitting before publishing.
- **Per-width selection with explicit kinds (`cuteafd-qmla-v3`):** on `val`, score `identity`, full fits, scale-only fits, and SmoothQuant inits across all widths at every requested combination, and pick the best transform per `(layer, path, serving_width)` (`select_per_width`). Classify every selection into `identity` | `scale_only` | `full` (`1e-6` rotation/scale collapse tolerances; `identity` emits zero tensors, `scale_only` emits `.s` only, identical non-identity F32 transforms alias per `(path, layer)` across widths — see [`qmla-format.md`](qmla-format.md)).
- **Combinations:** the pilot and the full run both evaluate **all nine** (pure-NoPE families: `8:0,4:0,2:0`):
  - **both paths active:** `8:8`, `4:8`, `4:4`, `2:4`;
  - **single path:** `8:0`, `4:0`, `2:0`, `0:8`, `0:4`. A `0` keeps that path's original cache format.

  **Always evaluate and report `2:0`.**
- **Error measurement:** every reported number is measured on the **held-out test sequences only**, against **identity (RTN)** and **fixed Hadamard** (measurement baseline only, never shipped), with `sink4_local128` protection on.
- **K8 transfer check (MC checkpoints):** replay **only the held-out test sequences** through `k8-compact`. Apply the per-width `original`-fitted transforms and compare each combination against **K8's own `0:0` output**. Reuse those K8 captures across all nine combinations. Training and validation use only `original`.
- **Long-context evaluation (`--long-eval`, 16K and 32K):** fitting stays at the current calibration sequence lengths. Run `--long-eval` after a fit, or standalone against an existing sidecar (`--long-eval --sidecar <sidecar-dir>`), over full-length 16K/32K `long_context` records rendered from the pinned panel snapshot in the local HF cache through `quantization/panel_corpus.py` (`long_context_mode="full"`, `heldout` then `calibration` splits, same chat kwargs as the corpus). Use the same calibration view and its own `0:0` output at full sequence length. Evaluation streams forward-only through the same 128-row tiled `eval_attention_output` (`tiled_sink_attention` with per-query `sink4_local128` protection) and compact DSA `SparseSelection` masks, recording `long_context_eval` (`transform_scope: "per_width_selected_transforms"`) in `qmla-validation-report.json`: leading with the **same headline table** (`4:0`, `2:4`, `8:0`, `8:8`, `4:8` with identity error, shipped error, ratio, `%` reduction, per-layer median/best/worst spread, `:0` caption, and `n/a` + reason for unsupported/NoPE modes) both **overall** (`headline_table` / `headline_table_markdown`) and **per position bucket** (`headline_table_by_position_bucket` / `headline_table_by_position_bucket_markdown`: `0-2k`, `2k-4k`, `4k-8k`, `8k-16k`, `16k-32k`, `32k+`), followed by full per-combination `attn_rel_err` alongside RTN and Hadamard, plus DSA selection drift.
- **Reporting (lead with the headline table):**
  - **Headline table first** (`headline_table` / `headline_table_markdown` in `qmla-validation-report.json`, `long_context_eval` overall & per position bucket, and the final report): one end-to-end joint post-`o_proj` measurement (never per-path errors summed) of the final selected per-layer transforms on held-out test for deployment modes **`4:0`, `2:4`, `8:0`, `8:8`, `4:8`** — reporting per mode: identity (RTN) attention error, shipped attention error, improvement as ratio (`ratio_to_rtn`) and `%` (`error_reduction_vs_identity_pct`), and **per-layer spread (median, best, worst)**. Pure-NoPE models emit `2:4`, `8:8`, `4:8` explicitly as `n/a` with the structural `qk_rope_dim=0` reason, with `4:0` and `8:0` as the measured headline rows. Always include the table caption stating `:0` means RoPE is kept unquantized (native format);
  - **Final report summary:** headline table (`4:0`, `2:4`, `8:0`, `8:8`, `4:8`, plus long-eval), new HF revision SHA, total sidecar size, wall time, and cleanup result;
  - supporting detail in the report JSON: fitting time per layer/path/width/mode, full held-out test cross-table and per-layer candidate scores (`cross_table`, `cells`, `candidate-table.json`), per-width selected candidate and explicit kind (`identity` | `scale_only` | `full`) per layer, and K8 transfer.

  **Keep every measured result in the report, including baseline misses.**
- **Every fitted combination is supported. There's no acceptance gate.** `supported_modes` lists every combination with calibrated serving widths; only structural limits are unsupported (INT2 RoPE; RoPE modes on pure-NoPE models). The quality cutoff for what to recommend is decided later, from the metrics below.
- **One summary error number per combination, recorded in `qmla-validation-report.json` and the manifest's per-combination entry:** `attn_rel_err` of the per-width selected transforms, the mean over layers and held-out test tokens of ‖o_q − o_ref‖² / ‖o_ref‖², where o is the post-`o_proj` attention output, o_ref is the 0:0 output of the same view, and o_q is the quantized output with protection on. Report it per combination (alongside identity/RTN and fixed Hadamard), for K8 transfer against K8's own 0:0, and per layer as supporting detail.

---

## 5. Validation and cross-view transfer
Before any publication step:
1. Run `qmla.validate_sidecar(sidecar_dir, expected_source=..., expected_geometries=..., banks_by_layer=..., strict=True)` to verify:
   - Single sidecar file `mla_calibration.safetensors` (no `mla_calibration.json`), self-describing `__metadata__["cuteafd_qmla"]`.
   - Schema (`cuteafd-qmla-v3`), canonical conventions, kind tolerances (`1e-6`), aliasing (`exact_per_layer_f32_dedup`), quantizer recipe, `artifact_status == "calibrated_unqualified"`, `protection_policy == "sink4_local128"`, `protection_scope == "full_history_mla_only"`.
   - `tensors_payload_sha256`, `tensors_payload_bytes`, zero orphan F32 tensors, canonical kind consistency (`identity` has no tensors; `scale_only` has `.s` only with $\|s-1\|_\infty > 10^{-6}$; `full` has rotation $> 10^{-6}$), `width_fingerprint`, `profile_fingerprint`, and `transform_fingerprint`.
   - Orthogonality $\|R_C^\top R_C - I\|_\infty \le 10^{-5}$, positive finite scales, identity replay byte-identical to the captured output with quantization disabled, and learned-transform algebraic relative L2 below `1e-5` for expanded content K/V or RoPE positional QK. Check the stored content convention (`post_norm`) and fit identity before reuse.
2. **Cross-view transfer (MC checkpoints):** the required check is the K8 transfer above: the held-out test sequences through `k8-compact`, each combination against K8's own 0:0. Other views (`k6-compact`, `fp8-compact`) are optional extra evidence on the same test slice. Record validated views in `applicable_views` and `view_transfer_validation`. Share one sidecar across views whenever transfer validates; create a view-scoped profile only if measured necessary.
3. **MC manifest check (MC checkpoints):** Confirm `attach_qmla_to_mc` + `validate_sidecar(..., mc_manifest=updated_mc)` + `mc_manifest.validate` passes with `required_features` unchanged, `shared_inventory_sha256` identical, all view fingerprints identical, exact agreement between `cuteafd-mc.json["qmla"]` and `__metadata__["cuteafd_qmla"]` (plus whole-file `tensors_sha256`/`tensors_bytes`), and `mla_calibration.safetensors` absent from every view's `weight_map`.

---

## 6. Same-repo incremental HF publication (zero shard re-upload)
Release the GPU first; run publication CPU-only over HTTP LFS (`HF_HUB_DISABLE_XET=1`, `huggingface-hub==1.33.0`). Pass the HF write token over **stdin only**.

1. **Dry-run audit first (always):**
```bash
HF_HUB_DISABLE_XET=1 PYTHONPATH="$PWD/quantization:$PWD/scripts/lib" \
  "$VENV/bin/python" -m hy4_flash.qmla_publish \
  --repo-id "$MODEL_ID" \
  --source-revision "$SOURCE_REVISION" \
  --sidecar-dir "$WORK_ROOT/full/sidecar" \
  --state-root "$WORK_ROOT/publish-state" \
  --validation-report "$WORK_ROOT/full/qmla-validation-report.json" \
  --update-mc-manifest \
  --dry-run
# Omit --update-mc-manifest for non-MC (plain/official) checkpoints; only mla_calibration.safetensors is added.
```
Verify the dry-run receipt reports:
- `reuploaded_shard_bytes == 0`
- `preserved_shards_count` equals the exact model shard count at `SOURCE_REVISION`
- `uploaded_files` contains ONLY:
  - MC repo: `mla_calibration.safetensors`, rewritten `cuteafd-mc.json` (with `"qmla"`), and optional `qmla-validation-report.json` / `README.md`.
  - Non-MC repo: `mla_calibration.safetensors` (self-describing `__metadata__["cuteafd_qmla"]`), and optional `qmla-validation-report.json` / `README.md` (no `cuteafd-mc.json` created).

2. **Commit new revision (single confirmation gate before live publish):**
Stop cleanly after the dry-run audit to confirm with TJ before publishing. Once confirmed, re-run without `--dry-run`, piping the HF token over stdin. `publish_incremental_revision` guards on `parent_commit=SOURCE_REVISION` (refuses if head moved) and verifies after commit that every preserved shard blob and size at `new_revision` is byte-identical to `SOURCE_REVISION`.

3. **SparkNest import, pin, and post-publish cleanup (`brief-template.md` §§10 & 12):**
   - Stage a hard-link HF cache with the Hub OIDs as blob names, run `nest hf import --offline --no-copy`, update `refs/main` and `refs/keep/<name>` (`printf %s`, no trailing newline), place via `nest plan`, and verify `nest ls --json`.
   - **After verified publication and SparkNest import/pin, DELETE all heavy intermediates:**
     - every unselected candidate fit (the entire `<output>/fit-checkpoints/` directory);
     - the capture banks (`<output>/captures/`);
     - all other intermediate QMLA scratch (publish staging, temporary copies, local sidecar export copy once imported).
   - **KEEP only small receipts:** `qmla-validation-report.json`, `candidate-table.json` (the complete selection table with every candidate's val/test scores and provenance), `qmla-k8-transfer.json`, run/publish logs, and `incremental-publish-complete.json` (recording the published sidecar's SHA-256 and new revision). Persisting every candidate is for the duration of the run (so selection can be rerun before publishing); it is never kept afterward.

---

## 7. Post-runtime qualification (when Phases 3–5 runtime exists)
Calibration alone produces an artifact with `"artifact_status": "calibrated_unqualified"`. Do **not** flip `"artifact_status"` to `"runtime_qualified"` until the runtime family qualifies the checkpoint on untouched held-out tests:
- `0:0` byte/launch/memory parity unchanged.
- Matched **Standard generated-token fidelity** on untouched held-out windows: mean KL, top-1 agreement, NLL, context scores separately, reported as **deltas vs matched `0:0` baseline** (both FP8 and BF16 where applicable; report honestly if INT8 fails to beat FP8).
- Held-out long-context / needle retrieval and agentic multi-turn prefix-cache exact-restore sessions (with default `sink4_local128` protection active on full-history MLA layers, SWA untouched).
- Matched speed & physical memory (bytes/token including scales, indexers, and `sink4_local128` protection rows; load time; write-transform overhead; cold prefill; short & long decode; concurrency).
- Speculative decoding correctness (especially target-KV-sharing drafters such as DFly2) and zero hidden full-precision KV duplicate.

---

## 8. Hardware and run hygiene
Follow `docs/quantization/brief-template.md` §12:
- No cluster locks (`hwlock.py`, `build-slot.sh`, `gpu*.lock`) on standalone quant hosts.
- Detached durable resume-on-restart unit via `systemd-run --user --expand-environment=no --setenv=PATH --setenv=HOME`.
- Self-contained task venv under the run root (`~/.cache/cuteafd/builds/<task>/venv` or host work root); never modify `~/Developer/cuteafd/.venv`.
- HF token on stdin only; no `pkill -f`, no `git stash`, never print full process argv.

---

## 9. Model addendum (fill in per checkpoint)
- **MODEL_ID:** `<org/repo>` (e.g. `wrldsuksgo2mars/Hy4-flash-EXL3-K4-MC`)
- **SOURCE_REVISION:** `<immutable 40-char SHA>`
- **Checkpoint kind:** `mc` (`original`, `k8-compact`, `k6-compact`, `fp8-compact`; rewrites `cuteafd-mc.json` + adds `mla_calibration.safetensors`) | `plain` (adds `mla_calibration.safetensors` only)
- **Family adapter:** `quantization/<family>/qmla_{capture,run,publish}.py`
- **Eligible full-history MLA layers:** `<list>` (explicitly list untouched SWA / KDA / HCA layers)
- **RoPE / NoPE convention:** `<interleaved | split_half | unrotated_pe | none>`, `mrope_section=<...>`, `shared_consumers=<...>`
- **Target modes:** `<e.g. 8:8,4:8,4:4,2:4,4:0,0:4,8:0,0:8,2:0 or 8:0,4:0,2:0 for pure NoPE>`
- **Protection policy:** `sink4_local128` (default on full-history MLA layers; SWA untouched)
- **Pinned corpus panel:** from `quantization/panel_corpus.py`: `wrldsuksgo2mars/agentic-quant-panel-v3@e3ac6ee6126f469e37ddf879a6cb2dccc10c6c99` (`agentic`, `exl3_standard`, `long_context`; v2 only for past runs)
- **Pilot configuration:** `--pilot` (`--layers <subset>`, all nine modes, `--train-sequences 16 --val-sequences 4 --test-sequences 4 --steps 100`)
- **Artifact status target:** `calibrated_unqualified` (promote to `runtime_qualified` after runtime qualification in #2909 Phases 3–5)
