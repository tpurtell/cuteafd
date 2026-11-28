# Incremental QMLA calibration template: add calibrated MLA sidecar to an existing checkpoint

**Every incremental QMLA calibration brief starts from this file.** Given a `MODEL_ID` and an immutable `SOURCE_REVISION` (MC or plain/non-MC), it adds a validated `cuteafd-qmla-v1` sidecar (**one** file: `mla_calibration.safetensors`, with **no separate `mla_calibration.json`**) as a **new revision of the same Hugging Face repository**, preserving every existing weight shard, view, index, tokenizer, and asset with **zero weight shard re-uploads**.

Copy this file whole, keep every section, and fill in **§9 Model addendum**. Never weaken a template rule.

**Authority & reference code to copy:**
- Architecture & schema contract: `docs/quantization/qmla-design.md` (confirmed 2026-11-28, [#2909](https://github.com/tpurtell/cuteafd/issues/2909)).
- Full MC quant rules (corpus, recovery, I/O, hub hygiene): `docs/quantization/brief-template.md`.
- **Shared family-agnostic QMLA library (reuse directly, do not duplicate):** `quantization/qmla/` (`capture.py`, `quantizer.py`, `fit.py`, `sidecar.py`, `publish.py`).
- **Working reference pipeline to copy/adapt:** `quantization/hy4_flash/qmla_capture.py`, `quantization/hy4_flash/qmla_run.py`, `quantization/hy4_flash/qmla_publish.py`.
  - For a new MLA family (starting with GLM 5.3 Flash), write only a thin family adapter implementing `QmlaCaptureAdapter` (layer geometries, RoPE/NoPE/M-RoPE convention, exact attention capture hook) plus thin CLI entrypoints copying `hy4_flash/qmla_{run,publish}.py`.

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
3. Verify the family's `QmlaCaptureAdapter` reproduces the pinned reference attention forward (`eval_attention_output` vs reference `< 2e-5`) on synthetic/preflight prompts, including 3D M-RoPE positions (`T != H != W`) for VL models and sparse DSA top-k pruning past `index_topk`.

---

## 2. Calibration corpus (pinned panel revision + fixed known component set, never Standard Fidelity)
Calibration draws deterministically from a pinned revision of `wrldsuksgo2mars/agentic-quant-panel-v2` over the fixed known component set `{agentic, exl3_standard, long_context}` (no arbitrary pluggable component list; never use the held-out Standard Fidelity panel):
1. **Default pilot & full calibration (`agentic` + `exl3_standard` at `54683731eb01c7c2fa5b801a0888e029a0e1b1c7`):**
   - `agentic`: rendered with **this checkpoint's own** tokenizer and chat template (matching `quantization/hy4_flash/prepare_corpus.py` and `brief-template.md` §3).
   - `exl3_standard`: packed with this checkpoint's tokenizer (`exl3-standard-pack-v1`).
   - Runs directly on a `hy4_flash.prepare_corpus` output directory (`manifest.json` + `calibration.safetensors`) or per-component `<name>.safetensors` files — do **not** block on `long_context`.
2. **Later recalibration with `long_context`:**
   - Once the new dataset revision carrying `long_context` (16K/32K traces, ~128K tokens per length class, plus agentic traces and long articles) lands, pass `--panel-revision <NEW_40_CHAR_SHA> --components agentic,exl3_standard,long_context` to exercise DSA top-k selection past 2,048 tokens.
- The exact `dataset_repo`, `dataset_revision`, and components used are recorded explicitly in `calibration_provenance`.

---

## 3. Pilot first (required before full run)
Always run a short pilot first — a few representative MLA layers, a small corpus slice, and one principal mode (`4:4`, or `4:0` for pure-NoPE families) with default `--protection sink4_local128` — and report results to TJ before running all layers and modes.

**Reference pilot command (Hy4-flash):**
```bash
PYTHONPATH="$PWD/quantization:$PWD/third_party/transformers/src" \
  "$VENV/bin/python" -m hy4_flash.qmla_run \
  --checkpoint "$SNAPSHOT_DIR" \
  --source-repo "$MODEL_ID" \
  --source-revision "$SOURCE_REVISION" \
  --view original \
  --corpus-dir "$CORPUS_DIR" \
  --output "$WORK_ROOT/pilot" \
  --pilot
# --pilot defaults to: --layers 0,21,41 --modes 4:4 --protection sink4_local128 --train-sequences 16 --val-sequences 4 --steps 100
```

**Pilot gates (must pass before full run):**
- `validate_sidecar` passes with `status == "passed"` and `protection_policy == "sink4_local128"`.
- Every pilot layer satisfies orthogonality $\|R_C^\top R_C - I\|_\infty < 10^{-5}$, strictly positive finite scales, and quantization-off algebraic equivalence `< 1e-4` relative L2.
- Held-out validation $\mathcal{L}_C$ and $\mathcal{L}_P$ improve over step-0 RTN (and report comparison vs fixed Sylvester Hadamard and SmoothQuant† $\alpha$-search under the same protection policy).
- Wall time per layer and peak GPU/host RAM logged to forecast full-run ETA (Asia/Taipei).

---

## 4. Full calibration (streaming per-layer, resumable)
Once the pilot passes, run all eligible MLA layers and target modes (reusing verified XXH3-128 captures with `--reuse-captures` whenever the layer captures already exist):

**Reference full command (Hy4-flash):**
```bash
PYTHONPATH="$PWD/quantization:$PWD/third_party/transformers/src" \
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
  --val-sequences 32 \
  --steps 300
```
- **Pure-NoPE models (e.g., GLM 5.3 Flash):** pass `--modes 8:0,4:0,2:0`. Any `R > 0` mode is omitted and recorded under `unsupported_modes_reason`.
- **Recovery & I/O (`brief-template.md` §7):** Layer captures (`captures/layer-XXXX.{safetensors,json}`) and per-layer fit checkpoints (`fit-checkpoints/*.{safetensors,json}`) are written sequentially with streamed XXH3-128 receipts bound to the frozen config/source/corpus identity. Resume cleanly on interruption; never restart a completed job.

---

## 5. Validation and cross-view transfer
Before any publication step:
1. Run `qmla.validate_sidecar(sidecar_dir, expected_source=..., expected_geometries=..., banks_by_layer=..., strict=True)` to verify:
   - Single sidecar file `mla_calibration.safetensors` (no `mla_calibration.json`), self-describing `__metadata__["cuteafd_qmla"]`.
   - Schema (`cuteafd-qmla-v1`), canonical conventions & quantizer recipe, `artifact_status == "calibrated_unqualified"`, `protection_policy == "sink4_local128"`, `protection_scope == "full_history_mla_only"`.
   - `tensors_payload_sha256`, `tensors_payload_bytes`, exact F32 tensor inventory, `profile_fingerprint`, and `transform_fingerprint`.
   - Orthogonality $\|R_C^\top R_C - I\|_\infty < 10^{-5}$, positive finite scales, and quantization-off numerical equivalence `< 1e-4`.
2. **Cross-view transfer (MC checkpoints):** Validate the fitted sidecar on each available core view (`original`, `k8-compact`, `k6-compact`, `fp8-compact`) by checking quantization-off equivalence and quantized $\mathcal{L}_C/\mathcal{L}_P$ delta on the validation slice. Record validated views in `applicable_views` and `view_transfer_validation`. Share one sidecar across views whenever transfer validates; create a view-scoped profile only if measured necessary.
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

2. **Commit new revision (after TJ approves the validation report and dry-run plan):**
Re-run without `--dry-run`, piping the HF token over stdin. `publish_incremental_revision` guards on `parent_commit=SOURCE_REVISION` (refuses if head moved) and verifies after commit that every preserved shard blob and size at `new_revision` is byte-identical to `SOURCE_REVISION`.

3. **SparkNest refresh (only after verified publication):** Import the new revision cleanly without copying unchanged weight shards, verify `nest ls --json`, and delete temporary captures/fit-checkpoints while keeping logs, `qmla-validation-report.json`, and publication receipts.

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
- No cluster locks (`hwlock.py`, `build-slot.sh`, `gpu*.lock`) on TJ's standalone quant host (maxq / cassowary).
- Detached durable runs via `systemd-run --user --expand-environment=no --setenv=PATH --setenv=HOME`.
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
- **Pinned corpus panel:** `wrldsuksgo2mars/agentic-quant-panel-v2@54683731eb01c7c2fa5b801a0888e029a0e1b1c7` (`agentic`, `exl3_standard`; later recalibration adds `long_context` via `--panel-revision <NEW_SHA> --components agentic,exl3_standard,long_context`)
- **Pilot configuration:** `--layers <subset> --modes <one-mode> --protection sink4_local128 --train-sequences 16 --val-sequences 4 --steps 100`
- **Artifact status target:** `calibrated_unqualified` (promote to `runtime_qualified` after runtime qualification in #2909 Phases 3–5)
