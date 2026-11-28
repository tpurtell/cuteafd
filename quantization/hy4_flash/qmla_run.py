"""End-to-end Hy4-flash QMLA calibration driver (reference MLA pipeline to copy).

Supports both short pilot runs (``--pilot`` or explicit ``--layers``, ``--modes``,
``--train-sequences``, ``--val-sequences``, ``--test-sequences``, ``--steps``) and full
12-DSA-layer calibration. Both fit every width (content c2/c4/c8, RoPE r4/r8) and
measure all nine combinations by default.

Splits: ``train`` fits, ``val`` selects fit steps and the shipped transform, ``test``
produces every reported attention error. The sidecar ships ONE content and ONE RoPE
transform per layer, chosen on ``val`` across every combination; the report also keeps
the per-width fits' test errors and the full candidate cross-table for analysis.

Corpus contract (amended Correction 1): defaults to the TWO existing components
``agentic`` + ``exl3_standard`` from
``wrldsuksgo2mars/agentic-quant-panel-v2@54683731eb01c7c2fa5b801a0888e029a0e1b1c7``
(matching ``quantization/hy4_flash/prepare_corpus.py``), and accepts ``long_context``
later via ``--panel-revision <NEW_SHA> --components agentic,exl3_standard,long_context``
over the fixed known component set ``{agentic, exl3_standard, long_context}``
(no arbitrary pluggable component list). Works directly on ``prepare_corpus.py`` output
directories (``manifest.json`` + ``calibration.safetensors``) or per-component
``<component>.safetensors`` files.

Protection default (Correction 2): ``--protection sink4_local128`` is enabled by
default on full-history MLA layers only (Hy4-flash's 30 SWA-512 layers are untouched).

Outputs (Correction 3 — no separate ``mla_calibration.json``):
- ``<output>/sidecar/mla_calibration.safetensors`` (self-describing with embedded
  ``__metadata__["cuteafd_qmla"]``, ``artifact_status="calibrated_unqualified"``)
- ``<output>/qmla-validation-report.json``
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import resource
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qmla.capture import (
    SPLIT_NAMES,
    CapturedLayerBank,
    MlaLayerGeometry,
    _atomic_json,
    check_split_description,
    describe_splits,
    load_layer_capture,
    save_layer_capture,
    sequence_sha256,
    split_train_val_test,
)
from qmla.fit import (
    _require_unprotected_tokens,
    DEFAULT_PROTECTION_POLICY,
    VALID_PROTECTION_POLICIES,
    ContentFitResult,
    FitConfig,
    RopeFitResult,
    fit_config_digest,
    fit_content_layer,
    fit_rope_layer,
    load_fit_checkpoint,
    save_fit_checkpoint,
)
from qmla.long_eval import (
    DEFAULT_STAGED_LONG_CONTEXT_DIR,
    LongContextEvaluator,
    LongContextRecord,
    has_prepared_long_context,
    load_long_context_records,
)
from qmla.measurement import (
    SELECTION_OBJECTIVE,
    inverse_max_error,
    layer_candidates,
    layer_rows,
    measure_candidates,
    measure_shipped,
    select_shipped,
    shipped_summary,
    summarize_modes,
    with_equivalence,
)
from qmla.quantizer import parse_qmla_mode
from qmla.sidecar import (
    DEFAULT_QMLA_PANEL_REVISION,
    QMLA_BASE_COMPONENTS,
    QMLA_KNOWN_COMPONENTS,
    load_sidecar_transforms,
    resolve_qmla_panel,
    validate_sidecar,
    write_sidecar,
)

from . import forward as hy4_forward
from . import recipe as hy4_recipe
from .qmla_capture import Hy4FlashCaptureAdapter

DEFAULT_PILOT_LAYERS = (0, 21, 41)
DEFAULT_FULL_MODES = ("8:8", "4:8", "4:4", "2:4", "8:0", "4:0", "2:0", "0:8", "0:4")
# The pilot is for understanding: it fits every width and measures every combination.
DEFAULT_PILOT_MODES = DEFAULT_FULL_MODES
PILOT_SPLITS = (16, 4, 4)
FULL_SPLITS = (128, 16, 16)


def _git_head_sha(root: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        if len(out) == 40:
            return out
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def parse_modes_arg(spec: str) -> tuple[tuple[int, int], ...]:
    """Parse comma-separated QUANT_MLA mode list (e.g. ``4:4,2:4,4:0``)."""
    raw = [part.strip() for part in spec.split(",") if part.strip()]
    if not raw:
        raise ValueError("--modes must specify at least one nonzero QUANT_MLA mode")
    parsed: list[tuple[int, int]] = []
    for item in raw:
        c_bits, r_bits = parse_qmla_mode(item)
        if c_bits == 0 and r_bits == 0:
            raise ValueError("0:0 needs no QMLA calibration; omit 0:0 from --modes")
        if (c_bits, r_bits) not in parsed:
            parsed.append((c_bits, r_bits))
    return tuple(parsed)


def parse_components_arg(spec: str) -> tuple[str, ...]:
    """Parse comma-separated component names strictly against QMLA_KNOWN_COMPONENTS."""
    raw = [part.strip() for part in spec.split(",") if part.strip()]
    _, _, ordered = resolve_qmla_panel(components=raw)
    return ordered


def required_profile_bits(modes: Sequence[tuple[int, int]]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return sorted unique nonzero (content_bits, rope_bits) needed to serve ``modes``."""
    c_bits = tuple(sorted({c for c, _ in modes if c > 0}))
    r_bits = tuple(sorted({r for _, r in modes if r > 0}))
    return c_bits, r_bits


def parse_layers_arg(spec: str, available_layers: Sequence[int]) -> tuple[int, ...]:
    """Parse ``all-dsa`` or comma-separated layer indices against available MLA layers."""
    avail = set(available_layers)
    if spec.strip().lower() in ("all", "all-dsa"):
        return tuple(sorted(available_layers))
    layers: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if not part.isdigit():
            raise ValueError(f"invalid layer index {part!r} in --layers")
        lid = int(part)
        if lid not in avail:
            raise ValueError(f"layer {lid} is not an eligible Hy4-flash MLA layer (eligible: {sorted(avail)})")
        if lid not in layers:
            layers.append(lid)
    if not layers:
        raise ValueError("--layers selected no layers")
    return tuple(sorted(layers))


def _load_one_component_file(
    corpus_dir: Path,
    comp_name: str,
    *,
    max_sequences: int | None = None,
) -> tuple[list[tuple[str, torch.Tensor]], dict[str, Any]]:
    from safetensors.torch import load_file

    candidates = (
        corpus_dir / f"{comp_name}.safetensors",
        corpus_dir / comp_name / "calibration.safetensors",
    )
    st_path = next((p for p in candidates if p.is_file()), None)
    if st_path is None:
        raise ValueError(
            f"pinned QMLA corpus component {comp_name!r} missing under {corpus_dir} "
            f"(expected {comp_name}.safetensors or {comp_name}/calibration.safetensors)"
        )

    data = load_file(str(st_path))
    if "input_ids" not in data or "offsets" not in data:
        raise ValueError(f"corpus component {comp_name!r} must contain input_ids and offsets tensors")
    ids = data["input_ids"].to(torch.int64).contiguous()
    offsets = data["offsets"].to(torch.int64).contiguous().tolist()
    if len(offsets) < 2 or offsets[0] != 0 or offsets[-1] != int(ids.numel()):
        raise ValueError(f"corpus component {comp_name!r} has malformed offsets")

    records: list[tuple[str, torch.Tensor]] = []
    hasher = hashlib.sha256()
    limit = len(offsets) - 1 if max_sequences is None else min(len(offsets) - 1, max_sequences)
    total_tokens = 0
    for idx in range(limit):
        seq = ids[offsets[idx] : offsets[idx + 1]]
        if seq.numel() == 0:
            continue
        seq_id = f"{comp_name}:{idx:05d}"
        records.append((seq_id, seq))
        hasher.update(seq.numpy().tobytes())
        total_tokens += int(seq.numel())
    if not records:
        raise ValueError(f"corpus component {comp_name!r} produced zero non-empty sequences")

    return records, {
        "sequences": len(records),
        "tokens": total_tokens,
        "input_ids_sha256": hasher.hexdigest(),
    }


def _load_prepared_combined_corpus(
    corpus_dir: Path,
    *,
    dataset_repo: str,
    dataset_revision: str,
    component_names: Sequence[str],
    max_sequences_per_component: int | None = None,
) -> tuple[list[tuple[str, torch.Tensor]], dict[str, dict[str, Any]]]:
    """Load directly from a ``hy4_flash.prepare_corpus`` output directory (manifest.json + calibration.safetensors)."""
    from safetensors.torch import load_file

    manifest_path = corpus_dir / "manifest.json"
    cal_path = corpus_dir / "calibration.safetensors"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ds = manifest.get("dataset", {})
    if ds.get("repo_id") != dataset_repo or ds.get("revision") != dataset_revision:
        raise ValueError(
            f"prepared corpus dataset mismatch: found {ds.get('repo_id')}@{ds.get('revision')}, "
            f"expected {dataset_repo}@{dataset_revision}"
        )
    manifest_comps = manifest.get("components", {})
    for comp in component_names:
        if comp not in manifest_comps:
            raise ValueError(
                f"prepared corpus manifest at {corpus_dir} does not contain requested component {comp!r} "
                f"(present: {sorted(manifest_comps)})"
            )

    data = load_file(str(cal_path))
    ids = data["input_ids"].to(torch.int64).contiguous()
    offsets = data["offsets"].to(torch.int64).contiguous().tolist()
    total_rows = len(offsets) - 1
    if len(offsets) < 2 or offsets[0] != 0 or offsets[-1] != int(ids.numel()):
        raise ValueError(f"prepared corpus {cal_path} has malformed offsets")

    # Component spans in prepare_corpus.py order: agentic, exl3_standard, optional long_context.
    all_records: list[tuple[str, torch.Tensor]] = []
    components_meta: dict[str, dict[str, Any]] = {}
    cursor = 0
    for comp in QMLA_KNOWN_COMPONENTS:
        if comp not in manifest_comps:
            continue
        count = int(manifest_comps[comp]["count"])
        end = cursor + count
        if end > total_rows:
            raise ValueError(f"prepared corpus component {comp!r} count exceeds calibration.safetensors rows")
        if comp in component_names:
            limit = count if max_sequences_per_component is None else min(count, max_sequences_per_component)
            hasher = hashlib.sha256()
            comp_seqs = 0
            comp_tokens = 0
            for local_idx in range(limit):
                row_idx = cursor + local_idx
                seq = ids[offsets[row_idx] : offsets[row_idx + 1]]
                if seq.numel() == 0:
                    continue
                all_records.append((f"{comp}:{local_idx:05d}", seq))
                hasher.update(seq.numpy().tobytes())
                comp_seqs += 1
                comp_tokens += int(seq.numel())
            if comp_seqs == 0:
                raise ValueError(f"prepared corpus component {comp!r} produced zero non-empty sequences")
            components_meta[comp] = {
                "sequences": comp_seqs,
                "tokens": comp_tokens,
                "input_ids_sha256": hasher.hexdigest(),
            }
        cursor = end

    return all_records, components_meta


def load_pinned_qmla_corpus(
    corpus_dir: Path | str,
    *,
    panel_revision: str = DEFAULT_QMLA_PANEL_REVISION,
    components: Sequence[str] = QMLA_BASE_COMPONENTS,
    max_sequences_per_component: int | None = None,
) -> tuple[list[tuple[str, torch.Tensor]], dict[str, Any]]:
    """Load requested known components from the pinned ``agentic-quant-panel-v2`` revision.

    Defaults to ``agentic`` + ``exl3_standard`` at ``54683731eb01c7c2fa5b801a0888e029a0e1b1c7``
    (does not fail closed on missing ``long_context``). Accepts ``long_context`` when passed
    explicitly with the updated ``panel_revision``. Supports both ``hy4_flash.prepare_corpus``
    directories (``manifest.json`` + ``calibration.safetensors``) and per-component
    ``<component>.safetensors`` files.
    """
    dataset_repo, dataset_revision, component_names = resolve_qmla_panel(
        panel_revision=panel_revision,
        components=components,
    )
    corpus_dir = Path(corpus_dir)
    if not corpus_dir.is_dir():
        raise ValueError(f"prepared QMLA corpus directory missing: {corpus_dir}")

    has_per_component = all(
        (corpus_dir / f"{c}.safetensors").is_file() or (corpus_dir / c / "calibration.safetensors").is_file()
        for c in component_names
    )
    if has_per_component:
        all_records: list[tuple[str, torch.Tensor]] = []
        components_meta: dict[str, dict[str, Any]] = {}
        for comp_name in component_names:
            recs, meta = _load_one_component_file(
                corpus_dir, comp_name, max_sequences=max_sequences_per_component
            )
            all_records.extend(recs)
            components_meta[comp_name] = meta
    elif (corpus_dir / "manifest.json").is_file() and (corpus_dir / "calibration.safetensors").is_file():
        all_records, components_meta = _load_prepared_combined_corpus(
            corpus_dir,
            dataset_repo=dataset_repo,
            dataset_revision=dataset_revision,
            component_names=component_names,
            max_sequences_per_component=max_sequences_per_component,
        )
    else:
        # Delegate to per-component loader to report the exact missing component path.
        all_records = []
        components_meta = {}
        for comp_name in component_names:
            recs, meta = _load_one_component_file(
                corpus_dir, comp_name, max_sequences=max_sequences_per_component
            )
            all_records.extend(recs)
            components_meta[comp_name] = meta

    combined_sha = hashlib.sha256(
        "\n".join(f"{k}:{components_meta[k]['input_ids_sha256']}" for k in sorted(components_meta)).encode("utf-8")
    ).hexdigest()
    corpus_provenance = {
        "dataset_repo": dataset_repo,
        "dataset_revision": dataset_revision,
        "components": components_meta,
        "input_ids_sha256": combined_sha,
    }
    return all_records, corpus_provenance


def split_identity(banks: Mapping[int, CapturedLayerBank]) -> dict[str, Any]:
    """The one split description every layer bank must share (checked, not assumed)."""
    descriptions = [bank.provenance.get("splits") for bank in banks.values()]
    first = check_split_description(descriptions[0])
    if any(d != first for d in descriptions[1:]):
        raise ValueError("layer captures disagree on the train/val/test split")
    return first


def _shipped_result(candidates, record, bank, device):
    if record is None:
        return None
    pool = candidates.content if record["candidate"].split(":")[1][0] == "c" else candidates.rope
    return with_equivalence(pool[record["candidate"]], bank, device)


def calibrate_hy4_flash_from_banks(
    banks: Mapping[int, CapturedLayerBank],
    *,
    output_dir: Path | str,
    source: Mapping[str, Any],
    modes: Sequence[tuple[int, int]],
    corpus_provenance: Mapping[str, Any],
    fit_config: FitConfig = FitConfig(),
    applicable_views: Sequence[str] | None = None,
    transfer_banks: Mapping[str, Mapping[int, CapturedLayerBank]] | None = None,
    device: torch.device | str = "cpu",
    code_commit: str | None = None,
    thinking_mode: str = "high",
    renderer_template_sha256: str = "",
) -> dict[str, Any]:
    """Fit -> select -> single-file sidecar -> test-split report from captured layer banks.

    For each layer: fit every width independently (train fits, val picks the step), build
    candidates from every fit and its SmoothQuant init, score the full candidate table on
    val and on test, ship the val-selected single content and RoPE transform, and report
    test errors only. ``transfer_banks`` maps a view (e.g. ``k8-compact``) to layer banks
    whose test split replays the calibration test sequences through that view.

    Shared by the production CLI driver and CPU numeric oracle tests.
    """
    started = time.monotonic()
    if not banks:
        raise ValueError("calibrate_hy4_flash_from_banks requires at least one CapturedLayerBank")
    fit_config.validate()
    output_dir = Path(output_dir).resolve()
    ckpt_dir = output_dir / "fit-checkpoints"
    sidecar_dir = output_dir / "sidecar"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    sidecar_dir.mkdir(parents=True, exist_ok=True)

    c_bits_list, r_bits_list = required_profile_bits(modes)
    ordered_layers = sorted(banks)
    geometries: list[MlaLayerGeometry] = [banks[lid].geometry for lid in ordered_layers]
    splits = split_identity(banks)

    identity_key = fit_config_digest(
        fit_config,
        extra={
            "fit_semantics": "causal-protection-v3-three-way-split",
            "source": dict(source),
            "splits": splits,
            "seq_lens": {
                str(lid): {name: [seq.seq_len for seq in bank.split(name)] for name in SPLIT_NAMES}
                for lid, bank in banks.items()
            },
            "layers": ordered_layers,
            "corpus": {k: v for k, v in corpus_provenance.items() if k != "splits"},
        },
    )

    shipped_content = []
    shipped_rope = []
    selection_by_layer: dict[str, Any] = {}
    layer_measurements = {}
    layer_fit_details = {}
    for lid in ordered_layers:
        bank = banks[lid]
        bank.validate()
        _require_unprotected_tokens(
            bank.test, protection=fit_config.protection, sink_tokens=fit_config.sink_tokens,
            recent_tokens=fit_config.recent_tokens, split_name="test",
        )
        layer_start = time.monotonic()
        details: dict[str, Any] = {}
        fits: dict[str, dict[int, ContentFitResult | RopeFitResult]] = {"content": {}, "rope": {}}
        for path, widths, fit_layer in (
            ("content", c_bits_list, fit_content_layer),
            ("rope", r_bits_list if bank.geometry.has_rope else (), fit_rope_layer),
        ):
            for bits in widths:
                timing_file = ckpt_dir / f"{path}-{bits}-layer-{lid:04d}.time.json"
                result = load_fit_checkpoint(ckpt_dir, kind=path, bits=bits, layer_id=lid, identity_key=identity_key)
                reused = result is not None
                if reused:
                    timing = json.loads(timing_file.read_text()) if timing_file.is_file() else {}
                    seconds = timing.get("seconds") if timing.get("identity_key") == identity_key else None
                else:
                    fit_start = time.monotonic()
                    result = fit_layer(bank, bits, fit_config, device=device)
                    seconds = time.monotonic() - fit_start
                inverse_error = inverse_max_error(result, bank.geometry)
                errors = {"inverse_max_abs_err": inverse_error, "equiv_max_rel_err": result.equiv_max_rel_err}
                if isinstance(result, ContentFitResult):
                    errors["orth_max_err"] = result.orth_max_err
                for check, value in errors.items():
                    if not math.isfinite(value):
                        raise ValueError(f"nonfinite {path} {bits}-bit layer {lid} {check}")
                    if check in ("equiv_max_rel_err", "orth_max_err") and value >= 1e-5:
                        raise ValueError(f"{path} {bits}-bit layer {lid} {check} = {value} (requires <1e-5)")
                if not all(math.isfinite(value) for value in (result.best_val_loss, *result.baselines.values())):
                    raise ValueError(f"nonfinite {path} {bits}-bit layer {lid} validation loss")
                if not reused:
                    save_fit_checkpoint(ckpt_dir, result, identity_key=identity_key)
                    _atomic_json(timing_file, {"identity_key": identity_key, "seconds": seconds})
                fits[path][bits] = result
                details.setdefault(path, {})[str(bits)] = {
                    "fit_seconds": seconds,
                    "reused_fit": reused,
                    "best_step": result.best_step,
                    "fitting_beat_init_on_val": result.best_step > 0,
                    # Fit-objective losses (L_C or L_P) on the selection split; not reported errors.
                    "val_objective": {
                        "smoothquant_init": result.baselines["smoothquant_alpha_val"],
                        "fitted": result.best_val_loss,
                        **{k: v for k, v in result.baselines.items() if k != "smoothquant_alpha_val"},
                    },
                    **errors,
                }

        candidates = layer_candidates(bank, fits["content"], fits["rope"], fit_config)
        val_table = measure_candidates(bank, candidates, modes, fit_config, device, split="val")
        test_table = measure_candidates(bank, candidates, modes, fit_config, device, split="test")
        selection = select_shipped(val_table, candidates, bank.geometry)
        selection_by_layer[str(lid)] = selection
        for path, sink in (("content", shipped_content), ("rope", shipped_rope)):
            result = _shipped_result(candidates, selection[path], bank, device)
            if result is None:
                continue
            for check, value in (("equiv_max_rel_err", result.equiv_max_rel_err),
                                 ("inverse_max_abs_err", inverse_max_error(result, bank.geometry))):
                if not math.isfinite(value) or (check == "equiv_max_rel_err" and value >= 1e-5):
                    raise ValueError(f"shipped {path} layer {lid} {check} = {value}")
            sink.append(result)
        layer_measurements[lid] = layer_rows(test_table, val_table, selection, bank.geometry)
        details["fit_and_measure_wall_seconds"] = time.monotonic() - layer_start
        layer_fit_details[str(lid)] = details

    combinations = summarize_modes(layer_measurements)
    shipped_by_layer = {
        path: {r.layer_id: r for r in rows} for path, rows in (("content", shipped_content), ("rope", shipped_rope))
    }

    view_transfer_validation: dict[str, Any] = {}
    for view, view_banks in (transfer_banks or {}).items():
        by_layer = {}
        for lid in ordered_layers:
            tbank = view_banks.get(lid)
            if tbank is None:
                raise ValueError(f"{view} transfer is missing layer {lid}")
            tbank.validate()
            if [s.seq_id for s in tbank.test] != [s.seq_id for s in banks[lid].test]:
                raise ValueError(f"{view} transfer must replay the calibration test sequences at layer {lid}")
            by_layer[lid] = measure_shipped(
                tbank, shipped_by_layer["content"].get(lid), shipped_by_layer["rope"].get(lid),
                modes, fit_config, device,
            )
        per_mode: dict[str, Any] = {}
        for lid, rows in by_layer.items():
            for key, row in rows.items():
                per_mode.setdefault(key, {"per_layer": {}})["per_layer"][str(lid)] = row
        for entry in per_mode.values():
            rows = list(entry["per_layer"].values())
            tokens = sum(r["tokens"] for r in rows)
            entry["layer_tokens"] = tokens
            for field in ("attn_rel_err", "rtn_attn_rel_err", "hadamard_attn_rel_err"):
                entry[field] = sum(r[field] * r["tokens"] for r in rows) / tokens
            entry["ratio_to_rtn"] = entry["attn_rel_err"] / entry["rtn_attn_rel_err"] if entry["rtn_attn_rel_err"] else None
        view_transfer_validation[view] = {"split": "test", "reference": f"{view} 0:0", "combinations": per_mode}

    def path_selection(path: str) -> dict[str, Any] | None:
        per_layer = {lid: sel[path] for lid, sel in selection_by_layer.items() if sel[path] is not None}
        return {"objective": SELECTION_OBJECTIVE, "per_layer": per_layer} if per_layer else None

    sample_bank = banks[ordered_layers[0]]
    provenance: dict[str, Any] = {
        "dataset_repo": corpus_provenance["dataset_repo"],
        "dataset_revision": corpus_provenance["dataset_revision"],
        "components": corpus_provenance["components"],
        **{f"{name}_sequences": len(sample_bank.split(name)) for name in SPLIT_NAMES},
        "splits": splits,
        "input_ids_sha256": corpus_provenance["input_ids_sha256"],
        "renderer_template_sha256": renderer_template_sha256,
        "thinking_mode": thinking_mode,
        "measurement_split": "test",
        "combination_measurements": {key: shipped_summary(row) for key, row in combinations.items()},
        "optimizer": "adamw",
        "steps_per_path": fit_config.steps,
        "seed": fit_config.seed,
        "code_commit": code_commit or _git_head_sha(Path(__file__).resolve().parents[2]),
    }

    manifest = write_sidecar(
        sidecar_dir,
        source=source,
        geometries=geometries,
        content_profile=shipped_content,
        rope_profile=shipped_rope,
        content_bits=c_bits_list,
        rope_bits=r_bits_list if shipped_rope else (),
        selection={"content": path_selection("content"), "rope": path_selection("rope")},
        provenance=provenance,
        applicable_views=(
            applicable_views if applicable_views is not None
            else (source["calibrated_view"], *view_transfer_validation)
        ),
        view_transfer_validation=view_transfer_validation,
        artifact_status="calibrated_unqualified",
        protection_policy=fit_config.protection,
    )
    validation = validate_sidecar(
        sidecar_dir,
        mc_qmla=manifest,
        expected_source={"repo": source["repo"], "revision": source["revision"], "family": source["family"]},
        expected_geometries=geometries,
        banks_by_layer=banks,
        equiv_tol=1e-5,
        strict=True,
    )
    fitting_beat_init = {
        "val_objective": {
            path: {
                str(bits): sum(
                    bool(layer_fit_details[str(lid)].get(path, {}).get(str(bits), {}).get("fitting_beat_init_on_val"))
                    for lid in ordered_layers
                )
                for bits in widths
            }
            for path, widths in (("content", c_bits_list), ("rope", r_bits_list))
        },
        "test_attn_rel_err": {
            key: {"layers": row["layers"], "fitted_beats_smoothquant": row["fitted_beats_smoothquant_layers"]}
            for key, row in combinations.items()
        },
        "shipped_fitted_layers": {
            path: sum(
                1 for sel in selection_by_layer.values()
                if sel[path] is not None and sel[path]["chosen"] == "fitted"
            )
            for path in ("content", "rope")
        },
    }
    report = {
        "schema": "cuteafd-qmla-validation-report-v2",
        "artifact_status": manifest["artifact_status"],
        "protection_policy": manifest["protection_policy"],
        "protection_scope": manifest["protection_scope"],
        "source": manifest["source"],
        "layers_calibrated": ordered_layers,
        "splits": {name: splits[name]["count"] for name in SPLIT_NAMES},
        "measurement_split": "test",
        "selection_split": "val",
        "selection_objective": SELECTION_OBJECTIVE,
        "combinations": combinations,
        "selection_by_layer": selection_by_layer,
        "fitting_beat_init": fitting_beat_init,
        "view_transfer_validation": view_transfer_validation,
        "fit_by_layer": layer_fit_details,
        "sidecar_bytes": manifest["tensors_bytes"],
        "fit_measure_validate_seconds": time.monotonic() - started,
        "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "process_peak_cuda_allocated_bytes": (
            torch.cuda.max_memory_allocated(torch.device(device)) if torch.device(device).type == "cuda" else None
        ),
        "supported_modes": manifest["supported_modes"],
        "validation": validation,
        "shipped_profiles": {
            path: (None if prof is None else {
                "bits": prof["bits"],
                "layers": prof["layers"],
                "chosen": {lid: row["candidate"] for lid, row in prof["selection"]["per_layer"].items()},
                "equiv_max_rel_err": prof["validation"]["equiv_max_rel_err"],
            })
            for path, prof in manifest["profiles"].items()
        },
    }
    _atomic_json(output_dir / "qmla-validation-report.json", report)
    return {"manifest": manifest, "validation": validation, "report": report, "sidecar_dir": str(sidecar_dir)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Local HF snapshot directory")
    parser.add_argument("--teacher-checkpoint", type=Path,
                        help="Pinned BF16 source for an incremental MC calibration")
    parser.add_argument("--source-repo", default=hy4_recipe.TARGET_REPO, help="Target/source HF repo id")
    parser.add_argument("--source-revision", required=True, help="Pinned 40-char immutable source revision")
    parser.add_argument("--base-repo", default=hy4_recipe.BASE_REPO)
    parser.add_argument("--base-revision", default=hy4_recipe.BASE_REVISION)
    parser.add_argument("--checkpoint-kind", choices=("mc", "plain"), default="mc")
    parser.add_argument("--view", default="original", help="Core view used for forward capture")
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=None,
        help="Prepared corpus dir (hy4_flash.prepare_corpus output or per-component safetensors)",
    )
    parser.add_argument(
        "--panel-revision",
        default=DEFAULT_QMLA_PANEL_REVISION,
        help="Pinned 40-char revision of wrldsuksgo2mars/agentic-quant-panel-v2",
    )
    parser.add_argument(
        "--components",
        default=",".join(QMLA_BASE_COMPONENTS),
        help="Known panel components: agentic,exl3_standard (default) or agentic,exl3_standard,long_context",
    )
    parser.add_argument("--output", type=Path, required=True, help="Task output/work directory on root NVMe")
    parser.add_argument("--pilot", action="store_true",
                        help="Short pilot preset (layers 0,21,41; all nine modes; 16/4/4 train/val/test seqs; 100 steps)")
    parser.add_argument("--layers", default=None, help="Layer selection: 'all-dsa' or comma-separated IDs (e.g. 0,21,41)")
    parser.add_argument("--modes", default=None, help="Comma-separated QUANT_MLA modes (e.g. 4:4 or 8:8,4:8,4:4,2:4)")
    parser.add_argument(
        "--protection",
        choices=sorted(VALID_PROTECTION_POLICIES),
        default=DEFAULT_PROTECTION_POLICY,
        help="Recent-token protection policy on full-history MLA layers (default: sink4_local128; SWA untouched)",
    )
    parser.add_argument("--train-sequences", type=int, default=None)
    parser.add_argument("--val-sequences", type=int, default=None, help="Step and shipped-transform selection split")
    parser.add_argument("--test-sequences", type=int, default=None, help="Held-out split for every reported error")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--reuse-captures", type=Path, default=None, help="Reuse verified XXH3-128 layer captures")
    parser.add_argument(
        "--long-eval",
        action="store_true",
        help="Run 16K/32K long-context evaluation after fitting, or standalone with --sidecar",
    )
    parser.add_argument(
        "--sidecar",
        type=Path,
        default=None,
        help="Existing QMLA sidecar (mla_calibration.safetensors or directory) for standalone --long-eval",
    )
    return parser


def resolve_run_settings(args: argparse.Namespace, eligible_layers: Sequence[int]) -> dict[str, Any]:
    """Resolve explicit CLI flags on top of pilot or full defaults."""
    if args.pilot:
        default_layers = ",".join(str(x) for x in DEFAULT_PILOT_LAYERS if x in eligible_layers) or str(eligible_layers[0])
        default_modes = ",".join(DEFAULT_PILOT_MODES)
        (default_train, default_val, default_test), default_steps = PILOT_SPLITS, 100
    else:
        default_layers = "all-dsa"
        default_modes = ",".join(DEFAULT_FULL_MODES)
        (default_train, default_val, default_test), default_steps = FULL_SPLITS, 300

    layers = parse_layers_arg(args.layers or default_layers, eligible_layers)
    modes = parse_modes_arg(args.modes or default_modes)
    components = parse_components_arg(args.components)
    train_n = default_train if args.train_sequences is None else int(args.train_sequences)
    val_n = default_val if args.val_sequences is None else int(args.val_sequences)
    test_n = default_test if args.test_sequences is None else int(args.test_sequences)
    steps = default_steps if args.steps is None else int(args.steps)
    return {
        "layers": layers,
        "modes": modes,
        "panel_revision": args.panel_revision,
        "components": components,
        "protection": args.protection,
        "train_sequences": train_n,
        "val_sequences": val_n,
        "test_sequences": test_n,
        "steps": steps,
    }


def _routed_forward(model, layer, inputs):
    """Read each expert once for all sequences; keep the reference's expert-ID sum order."""
    latent, ids, gamma = model.moe_route(torch.cat(inputs), layer)
    routed = torch.zeros_like(latent)
    for expert in ids.unique(sorted=True).tolist():
        rows, weights = model.expert_rows(ids, gamma, expert)
        routed[rows] += model.expert(expert, latent[rows], layer) * weights[:, None]
    return routed.split([len(x) for x in inputs])


def corpus_split_description(
    token_records: Sequence[tuple[str, torch.Tensor]],
    *,
    train_count: int,
    val_count: int,
    test_count: int,
    seed: int,
) -> dict[str, Any]:
    """Describe the deterministic three-way split of corpus rows by ID and token-ID SHA256."""
    parts = split_train_val_test(
        token_records, train_count=train_count, val_count=val_count, test_count=test_count,
        key_fn=lambda r: r[0], seed=seed,
    )
    return describe_splits(
        {name: [(sid, sequence_sha256(tokens)) for sid, tokens in part] for name, part in zip(SPLIT_NAMES, parts)},
        seed=seed,
    )


def check_reused_captures(
    banks: Mapping[int, CapturedLayerBank],
    resolved: Mapping[str, Any],
    teacher_identity: Mapping[str, Any],
    *,
    seed: int,
    expected_splits: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Refuse reused captures whose teacher, panel, seed, split counts or split membership differ."""
    first = banks[min(banks)]
    corpus_provenance = dict(first.provenance)
    if any(corpus_provenance.get(key) != value for key, value in teacher_identity.items()):
        raise ValueError("capture teacher or checkpoint identity differs from the requested run")
    if (corpus_provenance["dataset_revision"] != resolved["panel_revision"]
            or set(corpus_provenance["components"]) != set(resolved["components"])):
        raise ValueError("capture panel revision or components differ from the requested run")
    splits = split_identity(banks)
    if splits["seed"] != seed:
        raise ValueError(f"capture split seed {splits['seed']} differs from the requested seed {seed}")
    for bank in banks.values():
        if bank.provenance != corpus_provenance:
            raise ValueError(f"capture provenance differs at layer {bank.geometry.layer_id}")
        for name in SPLIT_NAMES:
            if len(bank.split(name)) != resolved[f"{name}_sequences"]:
                raise ValueError(f"capture {name} split count differs at layer {bank.geometry.layer_id}")
    if expected_splits is not None and dict(expected_splits) != splits:
        raise ValueError("capture split membership differs from the split of the requested corpus")
    return corpus_provenance


@torch.no_grad()
def _stream_hy4_flash_captures(
    reader: hy4_forward.Reader,
    adapter: Hy4FlashCaptureAdapter,
    target_layers: Sequence[int],
    token_records: Sequence[tuple[str, torch.Tensor]],
    *,
    train_count: int,
    val_count: int,
    test_count: int,
    seed: int,
    capture_dir: Path,
    provenance: dict[str, Any],
) -> dict[int, CapturedLayerBank]:
    """Run one frozen-weight layer-major forward up to max(target_layers) and capture target MLA layers."""
    model = adapter.model
    max_layer = max(target_layers)
    target_set = set(target_layers)
    embedding = reader.raw(hy4_forward.reference.OUTER_NAMES[0])

    provenance = {**provenance, "splits": corpus_split_description(
        token_records, train_count=train_count, val_count=val_count, test_count=test_count, seed=seed
    )}
    train_items, val_items, test_items = split_train_val_test(
        token_records, train_count=train_count, val_count=val_count, test_count=test_count,
        key_fn=lambda r: r[0], seed=seed,
    )
    active = [*train_items, *val_items, *test_items]
    active = [(sid, tokens.to(reader.device)) for sid, tokens in active]
    states = [model.initial(tokens, embedding=embedding) for _, tokens in active]
    banks = {}
    for layer_idx in range(max_layer + 1):
        started = time.monotonic()
        layer = model.load_layer(layer_idx)
        captures, mlp_args = [], []
        for (seq_id, tokens), state in zip(active, states, strict=True):
            residual, attention_post, attn_in = model.attention_start(state, tokens, layer)
            if layer_idx in target_set:
                cap, attn_out, _ = adapter.capture_sequence(attn_in, layer, seq_id=seq_id)
                captures.append(cap)
            else:
                attn_out, _ = model.attention(attn_in, layer)
            if layer_idx < max_layer:
                mlp_args.append(model.mlp_start(residual, attention_post, attn_out, layer))

        if layer_idx in target_set:
            bank = CapturedLayerBank(adapter.geometry_for(layer_idx), adapter.layer_params(layer),
                                     tuple(captures[:train_count]),
                                     tuple(captures[train_count:train_count + val_count]),
                                     tuple(captures[train_count + val_count:]), provenance)
            bank.validate()
            save_layer_capture(capture_dir, bank)
            banks[layer_idx] = bank
        if mlp_args:
            routed = [None] * len(mlp_args)
            if model.config.mlp_layer_types[layer_idx] == "sparse":
                routed = _routed_forward(model, layer, [args[2] for args in mlp_args])
            states = [model.mlp_finish(*args, layer, result)
                      for args, result in zip(mlp_args, routed, strict=True)]
        print(json.dumps({"phase": "capture", "completed_layer": layer_idx,
                          "seconds": time.monotonic() - started}), flush=True)
        del layer
    return banks


@torch.no_grad()
def run_hy4_flash_long_eval(
    adapter: Hy4FlashCaptureAdapter,
    *,
    sidecar_path: Path | str,
    records: Sequence[LongContextRecord],
    corpus_meta: Mapping[str, Any],
    device: torch.device | str = "cpu",
    layers: Sequence[int] | None = None,
    modes: Sequence[tuple[int, int]] | None = None,
    expected_source: Mapping[str, Any] | None = None,
    embedding: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Forward-only streaming long-context (16K/32K) evaluation against a shipped QMLA sidecar.

    Streams one layer at a time matching ``_stream_hy4_flash_captures``, capturing compact DSA
    group IDs (never ``[T, T]`` bool masks), evaluating shipped / RTN / Hadamard per sequence
    at each target layer, and freeing every capture immediately after scoring.
    """
    if not records:
        raise ValueError("run_hy4_flash_long_eval requires at least one LongContextRecord")
    started = time.monotonic()
    dev = torch.device(device)
    model = adapter.model
    manifest, content_by_layer, rope_by_layer = load_sidecar_transforms(
        sidecar_path, expected_source=expected_source, strict=True,
    )
    sidecar_layers = sorted(set(content_by_layer) | set(rope_by_layer))
    target_layers = tuple(sorted(sidecar_layers if layers is None else layers))
    missing = [lid for lid in target_layers if lid not in set(sidecar_layers)]
    if missing:
        raise ValueError(f"sidecar is missing requested long-eval layers: {missing}")

    if modes is None:
        eval_modes = tuple(parse_qmla_mode(m) for m in manifest["supported_modes"])
    else:
        supported = set(manifest["supported_modes"])
        eval_modes = tuple(modes)
        unsupported = [f"{c}:{r}" for c, r in eval_modes if f"{c}:{r}" not in supported]
        if unsupported:
            raise ValueError(f"sidecar does not support requested long-eval modes: {unsupported}")

    evaluator = LongContextEvaluator(
        modes=eval_modes,
        content_transforms=content_by_layer,
        rope_transforms=rope_by_layer,
        protection_policy=manifest["protection_policy"],
        dsa_indexer_note=adapter.dsa_indexer_note,
    )

    max_layer = max(target_layers)
    target_set = set(target_layers)
    if embedding is None:
        embedding = model.reader.raw(hy4_forward.reference.OUTER_NAMES[0])
    active = [(rec, rec.input_ids.to(dev)) for rec in records]
    states = [model.initial(tokens, embedding=embedding) for _, tokens in active]

    for layer_idx in range(max_layer + 1):
        layer_start = time.monotonic()
        layer = model.load_layer(layer_idx)
        is_target = layer_idx in target_set
        params = adapter.layer_params(layer) if is_target else None
        geom = adapter.geometry_for(layer_idx) if is_target else None
        mlp_args = []
        for (rec, tokens), state in zip(active, states, strict=True):
            residual, attention_post, attn_in = model.attention_start(state, tokens, layer)
            if is_target:
                assert params is not None and geom is not None
                cap, attn_out, _ = adapter.capture_sequence(attn_in, layer, seq_id=rec.seq_id, compact_mask=True)
                evaluator.evaluate_layer_sequence(cap, params, geom, split=rec.split, device=dev)
                del cap
            else:
                attn_out, _ = model.attention(attn_in, layer)
            if layer_idx < max_layer:
                mlp_args.append(model.mlp_start(residual, attention_post, attn_out, layer))
        if mlp_args:
            routed = [None] * len(mlp_args)
            if model.config.mlp_layer_types[layer_idx] == "sparse":
                routed = _routed_forward(model, layer, [args[2] for args in mlp_args])
            states = [model.mlp_finish(*args, layer, result) for args, result in zip(mlp_args, routed, strict=True)]
        evaluator.release_layer_cache()
        print(
            json.dumps({"phase": "long_eval", "completed_layer": layer_idx, "evaluated": is_target,
                        "seconds": time.monotonic() - layer_start}),
            flush=True,
        )
        del layer

    return evaluator.finalize(
        corpus_meta=corpus_meta,
        transform_fingerprint=manifest["transform_fingerprint"],
        wall_seconds=time.monotonic() - started,
        device=dev,
    )


def attach_long_context_eval(
    output_dir: Path | str,
    long_eval_section: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Attach ``long_context_eval`` into ``<output>/qmla-validation-report.json`` atomically."""
    output_dir = Path(output_dir).resolve()
    report_path = output_dir / "qmla-validation-report.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
    else:
        if manifest is None:
            raise ValueError("manifest is required when creating a new qmla-validation-report.json")
        report = {
            "schema": "cuteafd-qmla-validation-report-v2",
            "artifact_status": manifest["artifact_status"],
            "protection_policy": manifest["protection_policy"],
            "protection_scope": manifest["protection_scope"],
            "source": manifest["source"],
            "supported_modes": list(manifest["supported_modes"]),
            "transform_fingerprint": manifest["transform_fingerprint"],
        }
    report["long_context_eval"] = dict(long_eval_section)
    _atomic_json(report_path, report)
    return report


def _resolve_long_context_source(corpus_dir: Path | None, *, standalone: bool) -> Path:
    if corpus_dir is not None and (standalone or has_prepared_long_context(corpus_dir)):
        return corpus_dir
    return DEFAULT_STAGED_LONG_CONTEXT_DIR


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.sidecar is not None and not args.long_eval:
        parser.error("--sidecar requires --long-eval")

    checkpoint = args.checkpoint.resolve()
    config_bytes = (checkpoint / "config.json").read_bytes()
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()

    teacher = args.teacher_checkpoint or checkpoint
    if args.checkpoint_kind == "mc":
        if args.teacher_checkpoint is None:
            from huggingface_hub import snapshot_download
            teacher = Path(snapshot_download(args.base_repo, revision=args.base_revision, local_files_only=True))
        manifest = json.loads((checkpoint / "cuteafd-mc.json").read_text())
        if (checkpoint.name != args.source_revision or teacher.name != args.base_revision
                or manifest["source"] != {"repo": args.base_repo, "revision": args.base_revision}):
            raise ValueError("BF16 teacher or MC source differs from the pinned calibration identities")
    reader = hy4_forward.Reader(teacher, device=args.device)
    model = hy4_forward.Model(reader)
    adapter = Hy4FlashCaptureAdapter(model, include_swa=False)
    eligible = tuple(g.layer_id for g in adapter.mla_geometries(include_swa=False))

    # Standalone --long-eval against an existing --sidecar (no refitting).
    if args.sidecar is not None:
        sidecar_manifest, _, _ = load_sidecar_transforms(args.sidecar, strict=True)
        expected_source = {
            "repo": args.source_repo,
            "revision": args.source_revision,
            "family": "hy4_flash",
            "config_sha256": config_sha256,
        }
        long_layers = parse_layers_arg(args.layers, eligible) if args.layers else None
        long_modes = parse_modes_arg(args.modes) if args.modes else None
        long_source = _resolve_long_context_source(args.corpus_dir, standalone=True)
        long_records, long_corpus_meta = load_long_context_records(long_source, checkpoint_dir=checkpoint)
        long_eval = run_hy4_flash_long_eval(
            adapter,
            sidecar_path=args.sidecar,
            records=long_records,
            corpus_meta=long_corpus_meta,
            device=args.device,
            layers=long_layers,
            modes=long_modes,
            expected_source=expected_source,
        )
        attach_long_context_eval(args.output, long_eval, manifest=sidecar_manifest)
        print(
            json.dumps({
                "status": "passed",
                "phase": "long_eval",
                "layers_evaluated": long_eval["layers_evaluated"],
                "combinations": sorted(long_eval["combinations"]),
                "transform_fingerprint": long_eval["transform_fingerprint"],
                "report": str(args.output.resolve() / "qmla-validation-report.json"),
            }),
            flush=True,
        )
        return 0

    resolved = resolve_run_settings(args, eligible)

    if args.corpus_dir is None and args.reuse_captures is None:
        parser.error("provide --corpus-dir PATH or --reuse-captures DIR")

    capture_dir = (args.reuse_captures or (args.output / "captures")).resolve()
    teacher_identity = {
        "target_revision": args.source_revision,
        "teacher_revision": args.base_revision if args.checkpoint_kind == "mc" else args.source_revision,
        "teacher_config_sha256": hashlib.sha256((teacher / "config.json").read_bytes()).hexdigest(),
    }
    all_records = None
    if args.corpus_dir is not None:
        all_records, corpus_provenance = load_pinned_qmla_corpus(
            args.corpus_dir,
            panel_revision=resolved["panel_revision"],
            components=resolved["components"],
        )
    split_counts = {f"{name}_count": resolved[f"{name}_sequences"] for name in SPLIT_NAMES}
    if args.reuse_captures is not None:
        banks = {lid: load_layer_capture(capture_dir, lid) for lid in resolved["layers"]}
        corpus_provenance = check_reused_captures(
            banks, resolved, teacher_identity, seed=args.seed,
            expected_splits=None if all_records is None else corpus_split_description(
                all_records, seed=args.seed, **split_counts
            ),
        )
    else:
        assert all_records is not None
        corpus_provenance.update(teacher_identity)
        banks = _stream_hy4_flash_captures(
            reader,
            adapter,
            resolved["layers"],
            all_records,
            seed=args.seed,
            capture_dir=capture_dir,
            provenance=corpus_provenance,
            **split_counts,
        )

    source = {
        "repo": args.source_repo,
        "revision": args.source_revision,
        "base_repo": args.base_repo,
        "base_revision": args.base_revision,
        "family": "hy4_flash",
        "model_type": "hy_v4_flash",
        "config_sha256": config_sha256,
        "checkpoint_kind": args.checkpoint_kind,
        "calibrated_view": args.view,
    }
    fit_cfg = FitConfig(steps=resolved["steps"], seed=args.seed, protection=resolved["protection"])
    template_path = checkpoint / "chat_template.jinja"
    template_sha = hashlib.sha256(template_path.read_bytes()).hexdigest() if template_path.is_file() else ""

    result = calibrate_hy4_flash_from_banks(
        banks,
        output_dir=args.output,
        source=source,
        modes=resolved["modes"],
        corpus_provenance=corpus_provenance,
        fit_config=fit_cfg,
        device=args.device,
        renderer_template_sha256=template_sha,
    )
    if args.long_eval:
        del banks
        long_source = _resolve_long_context_source(args.corpus_dir, standalone=False)
        long_records, long_corpus_meta = load_long_context_records(long_source, checkpoint_dir=checkpoint)
        long_eval = run_hy4_flash_long_eval(
            adapter,
            sidecar_path=result["sidecar_dir"],
            records=long_records,
            corpus_meta=long_corpus_meta,
            device=args.device,
            layers=resolved["layers"],
            modes=resolved["modes"],
            expected_source={"repo": source["repo"], "revision": source["revision"], "family": "hy4_flash"},
        )
        attach_long_context_eval(args.output, long_eval, manifest=result["manifest"])
    print(
        json.dumps({
            "status": result["validation"]["status"],
            "artifact_status": result["manifest"]["artifact_status"],
            "protection_policy": result["manifest"]["protection_policy"],
            "layers": list(resolved["layers"]),
            "supported_modes": result["manifest"]["supported_modes"],
            "transform_fingerprint": result["manifest"]["transform_fingerprint"],
            "sidecar_dir": result["sidecar_dir"],
            "long_eval": bool(args.long_eval),
        }),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
