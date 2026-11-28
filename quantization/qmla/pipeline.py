"""Shared MLA calibration, v3 candidate persistence, per-width selection, and sidecar reporting."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import resource
import subprocess
import time
from typing import Any, Callable, Mapping, Sequence

import torch

import panel_corpus
from qmla.capture import (
    SPLIT_NAMES,
    CapturedLayerBank,
    MlaLayerGeometry,
    _atomic_json,
    check_split_description,
    describe_splits,
    sequence_sha256,
    split_train_val_test,
)
from qmla.fit import (
    DEFAULT_FIT_BUDGET,
    DEFAULT_PROTECTION_POLICY,
    FIT_CHECKPOINT_SCHEMA,
    VALID_FIT_MODES,
    VALID_PROTECTION_POLICIES,
    ContentFitResult,
    FitConfig,
    RopeFitResult,
    _require_unprotected_tokens,
    check_content_identity,
    classify_content_kind,
    classify_rope_kind,
    fit_checkpoint_stem,
    fit_config_digest,
    fit_content_layer,
    fit_rope_layer,
    load_fit_checkpoint,
    save_fit_checkpoint,
)
from qmla.measurement import (
    HEADLINE_DEPLOYMENT_MODES,
    SELECTION_OBJECTIVE,
    LayerCandidates,
    _identity_content_result,
    _identity_rope_result,
    build_headline_table,
    format_headline_table,
    inverse_max_error,
    layer_candidates,
    layer_rows,
    measure_candidates,
    measure_shipped,
    parse_candidate_id,
    select_per_width,
    select_shared_v2,
    shipped_summary,
    summarize_modes,
    with_equivalence,
)
from qmla.quantizer import parse_qmla_mode
from qmla.sidecar import (
    CANONICAL_CONVENTIONS,
    resolve_qmla_panel,
    validate_sidecar,
    write_sidecar,
)

PILOT_SPLITS = (16, 4, 4)
FULL_SPLITS = (128, 16, 16)
VALIDATION_REPORT_SCHEMA = "cuteafd-qmla-validation-report-v3"
CANDIDATE_TABLE_SCHEMA = "cuteafd-qmla-candidate-table-v3"


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
    """Parse comma-separated component names into canonical panel order (full check is separate)."""
    raw = [part.strip() for part in spec.split(",") if part.strip()]
    _, _, ordered = resolve_qmla_panel(components=raw, full=False)
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
            raise ValueError(f"layer {lid} is not an eligible MLA layer (eligible: {sorted(avail)})")
        if lid not in layers:
            layers.append(lid)
    if not layers:
        raise ValueError("--layers selected no layers")
    return tuple(sorted(layers))


def load_pinned_qmla_corpus(
    corpus_dir: Path | str,
    *,
    panel_revision: str | None = None,
    components: Sequence[str] | None = None,
    full: bool = True,
) -> tuple[list[tuple[str, torch.Tensor]], dict[str, Any]]:
    """Load a prepared ``panel_corpus`` calibration split through ``load_component_records``."""
    dataset_repo, dataset_revision, component_names = resolve_qmla_panel(
        panel_revision=panel_revision, components=components, full=full
    )
    pin = panel_corpus.panel_pin(dataset_revision)
    rows, prov = panel_corpus.load_component_records(
        corpus_dir,
        "calibration",
        components=component_names,
        full=full,
        long_context_mode="windowed",
        dataset=dict(repo_id=pin.repo_id, revision=pin.revision, sha256sums_sha256=pin.sha256sums_sha256),
    )
    records = [(row["id"].replace("/", ":"), torch.tensor(row["input_ids"], dtype=torch.int64)) for row in rows]
    return records, {
        "dataset_repo": prov["dataset_repo"],
        "dataset_revision": prov["dataset_revision"],
        "components": prov["components"],
        "input_ids_sha256": prov["input_ids_sha256"],
    }


def split_identity(banks: Mapping[int, CapturedLayerBank]) -> dict[str, Any]:
    """The one split description every layer bank must share (checked, not assumed)."""
    descriptions = [bank.provenance.get("splits") for bank in banks.values()]
    first = check_split_description(descriptions[0])
    if any(d != first for d in descriptions[1:]):
        raise ValueError("layer captures disagree on the train/val/test split")
    return first


def _resolve_fit_budget(
    fit_budget: Mapping[str, Mapping[int, str | Sequence[str]]] | None,
) -> dict[str, dict[int, tuple[str, ...]]]:
    resolved: dict[str, dict[int, tuple[str, ...]]] = {
        path: {int(b): tuple(modes) for b, modes in widths.items()}
        for path, widths in DEFAULT_FIT_BUDGET.items()
    }
    if fit_budget is not None:
        for path, widths in fit_budget.items():
            if path not in resolved:
                raise ValueError(f"unknown fit_budget path: {path!r}")
            for bits, spec in widths.items():
                modes = (spec,) if isinstance(spec, str) else tuple(spec)
                if not modes or any(m not in VALID_FIT_MODES for m in modes):
                    raise ValueError(f"invalid fit_budget modes {spec!r} for {path} {bits}-bit")
                resolved[path][int(bits)] = tuple(dict.fromkeys(modes))
    return resolved


def _timing_path(ckpt_dir: Path, path: str, bits: int, lid: int, *, fit_mode: str) -> Path:
    tag = "scale" if fit_mode == "scale_only" else fit_mode
    return ckpt_dir / f"{path}-{tag}-{bits}-layer-{lid:04d}.time.json"


def _check_fit_result_numerics(
    result: ContentFitResult | RopeFitResult,
    geometry: MlaLayerGeometry,
    *,
    path: str,
    bits: int,
    lid: int,
) -> dict[str, float]:
    inverse_error = inverse_max_error(result, geometry)
    errors: dict[str, float] = {"inverse_max_abs_err": inverse_error, "equiv_max_rel_err": result.equiv_max_rel_err}
    if isinstance(result, ContentFitResult):
        errors["orth_max_err"] = result.orth_max_err
    for check, value in errors.items():
        if not math.isfinite(value):
            raise ValueError(f"nonfinite {path} {bits}-bit layer {lid} {check}")
        if check in ("equiv_max_rel_err", "orth_max_err") and value >= 1e-5:
            raise ValueError(f"{path} {bits}-bit layer {lid} {check} = {value} (requires <1e-5)")
    if not all(math.isfinite(value) for value in (result.best_val_loss, *result.baselines.values())):
        raise ValueError(f"nonfinite {path} {bits}-bit layer {lid} validation loss")
    return errors


def _fit_detail_entry(
    result: ContentFitResult | RopeFitResult,
    errors: Mapping[str, float],
    *,
    path: str,
    seconds: float | None,
    reused: bool,
) -> dict[str, Any]:
    kind = (
        classify_content_kind(result.r_c, result.s_c)
        if path == "content"
        else classify_rope_kind(result.theta, result.s_p)
    )
    return {
        "fit_mode": result.fit_mode,
        "classified_kind": kind,
        "checkpoint_stem": fit_checkpoint_stem(path, result.bits, result.layer_id, fit_mode=result.fit_mode),
        "fit_seconds": seconds,
        "reused_fit": reused,
        "best_step": result.best_step,
        "fitting_beat_init_on_val": result.best_step > 0,
        "val_objective": {
            "smoothquant_init": result.baselines["smoothquant_alpha_val"],
            "fitted": result.best_val_loss,
            **{k: v for k, v in result.baselines.items() if k != "smoothquant_alpha_val"},
        },
        **errors,
    }


def _candidate_catalog_entry(path: str, cid: str, result: ContentFitResult | RopeFitResult) -> dict[str, Any]:
    ckind, _, src_bits = parse_candidate_id(cid)
    kind = (
        classify_content_kind(result.r_c, result.s_c)
        if path == "content"
        else classify_rope_kind(result.theta, result.s_p)
    )
    stem = (
        None
        if ckind == "identity"
        else fit_checkpoint_stem(path, int(src_bits), result.layer_id, fit_mode=result.fit_mode)
    )
    entry: dict[str, Any] = {
        "candidate": cid,
        "fit_mode": "identity" if ckind == "identity" else result.fit_mode,
        "source_bits": src_bits,
        "classified_kind": kind,
        "checkpoint_stem": stem,
        "best_step": int(result.best_step),
        "init_alpha": float(result.init_alpha),
        "equiv_max_rel_err": float(result.equiv_max_rel_err),
    }
    if isinstance(result, ContentFitResult):
        entry["orth_max_err"] = float(result.orth_max_err)
    return entry


def compute_fit_identity_key(
    banks: Mapping[int, CapturedLayerBank],
    *,
    source: Mapping[str, Any],
    corpus_provenance: Mapping[str, Any],
    fit_config: FitConfig,
) -> str:
    """Canonical XXH3-128 v3 fit-checkpoint identity key."""
    ordered_layers = sorted(banks)
    splits = split_identity(banks)
    return fit_config_digest(
        fit_config,
        extra={
            "schema": FIT_CHECKPOINT_SCHEMA,
            "fit_semantics": "three-way-split-post-norm-fp64-validation-rotation-v3",
            "conventions": CANONICAL_CONVENTIONS,
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


def _persist_smoothquant_candidates(
    candidates: LayerCandidates,
    bank: CapturedLayerBank,
    *,
    ckpt_dir: Path,
    identity_key: str,
    device: torch.device | str,
) -> LayerCandidates:
    """Ensure every SmoothQuant init candidate has verified equivalence and is persisted as a v3 checkpoint."""
    new_content = dict(candidates.content)
    new_rope = dict(candidates.rope)
    for path, pool in (("content", new_content), ("rope", new_rope)):
        for cid, cand in list(pool.items()):
            ckind, _, src_bits = parse_candidate_id(cid)
            if ckind != "smoothquant":
                continue
            assert src_bits is not None
            loaded = load_fit_checkpoint(
                ckpt_dir,
                kind=path,
                bits=src_bits,
                layer_id=bank.geometry.layer_id,
                identity_key=identity_key,
                fit_mode="smoothquant",
            )
            if loaded is not None:
                pool[cid] = loaded
                continue
            checked = replace(with_equivalence(cand, bank, device), fit_mode="smoothquant")
            if not math.isfinite(checked.equiv_max_rel_err) or checked.equiv_max_rel_err >= 1e-5:
                raise ValueError(
                    f"smoothquant {path} {src_bits}-bit layer {bank.geometry.layer_id} "
                    f"equiv_max_rel_err={checked.equiv_max_rel_err}"
                )
            save_fit_checkpoint(ckpt_dir, checked, identity_key=identity_key)
            pool[cid] = checked
    return LayerCandidates(content=new_content, rope=new_rope)


def calibrate_from_banks(
    banks: Mapping[int, CapturedLayerBank],
    *,
    output_dir: Path | str,
    source: Mapping[str, Any],
    modes: Sequence[tuple[int, int]],
    corpus_provenance: Mapping[str, Any],
    fit_config: FitConfig = FitConfig(),
    fit_budget: Mapping[str, Mapping[int, str | Sequence[str]]] | None = None,
    applicable_views: Sequence[str] | None = None,
    device: torch.device | str = "cpu",
    code_commit: str | None = None,
    thinking_mode: str = "high",
    renderer_template_sha256: str = "",
) -> dict[str, Any]:
    """Fit every v3 candidate -> persist all candidates & scores -> select per width -> emit v3 sidecar.

    Default v3 fitting budget fits BOTH ``full`` and ``scale_only`` across every calibrated
    width (plus step-0 SmoothQuant inits and ``identity``), and persists EVERY candidate
    under ``<output>/fit-checkpoints/`` (``cuteafd-qmla-fit-checkpoint-v3``) along with the
    complete val/test score table in ``<output>/candidate-table.json`` so selection policies
    can be rerun via ``reselect_from_persisted_candidates`` without refitting.
    """
    started = time.monotonic()
    if not banks:
        raise ValueError("calibrate_from_banks requires at least one CapturedLayerBank")
    fit_config.validate()
    active_budget = _resolve_fit_budget(fit_budget)
    output_dir = Path(output_dir).resolve()
    ckpt_dir = output_dir / "fit-checkpoints"
    sidecar_dir = output_dir / "sidecar"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    sidecar_dir.mkdir(parents=True, exist_ok=True)

    c_bits_list, r_bits_list = required_profile_bits(modes)
    ordered_layers = sorted(banks)
    geometries: list[MlaLayerGeometry] = [banks[lid].geometry for lid in ordered_layers]
    splits = split_identity(banks)
    identity_key = compute_fit_identity_key(
        banks, source=source, corpus_provenance=corpus_provenance, fit_config=fit_config
    )

    content_selections: dict[int, dict[int, ContentFitResult]] = {b: {} for b in c_bits_list}
    rope_selections: dict[int, dict[int, RopeFitResult]] = {}
    has_any_rope = any(g.has_rope for g in geometries)
    if r_bits_list and has_any_rope:
        rope_selections = {b: {} for b in r_bits_list}

    selection_meta: dict[str, dict[str, dict[str, Any]] | None] = {
        "content": {str(b): {} for b in c_bits_list} if c_bits_list else None,
        "rope": {str(b): {} for b in r_bits_list} if (r_bits_list and has_any_rope) else None,
    }
    selection_by_layer: dict[str, Any] = {}
    v2_selection_by_layer: dict[str, Any] = {}
    layer_measurements = {}
    layer_fit_details = {}
    persisted_layers: dict[str, Any] = {}

    for lid in ordered_layers:
        bank = banks[lid]
        bank.validate()
        _require_unprotected_tokens(
            bank.test,
            protection=fit_config.protection,
            sink_tokens=fit_config.sink_tokens,
            recent_tokens=fit_config.recent_tokens,
            split_name="test",
        )
        layer_start = time.monotonic()
        if c_bits_list:
            check_content_identity((seq.to(device) for seq in bank.val), bank.params.to(device), bank.geometry)
        details: dict[str, Any] = {}
        full_fits: dict[str, dict[int, Any]] = {"content": {}, "rope": {}}
        scale_fits: dict[str, dict[int, Any]] = {"content": {}, "rope": {}}

        for path, widths, fit_layer in (
            ("content", c_bits_list, fit_content_layer),
            ("rope", r_bits_list if bank.geometry.has_rope else (), fit_rope_layer),
        ):
            for bits in widths:
                width_details: dict[str, Any] = {}
                for fit_mode in active_budget[path][bits]:
                    result = load_fit_checkpoint(
                        ckpt_dir,
                        kind=path,
                        bits=bits,
                        layer_id=lid,
                        identity_key=identity_key,
                        fit_mode=fit_mode,
                    )
                    reused = result is not None
                    timing_file = _timing_path(ckpt_dir, path, bits, lid, fit_mode=fit_mode)
                    if reused:
                        timing = json.loads(timing_file.read_text()) if timing_file.is_file() else {}
                        seconds = timing.get("seconds") if timing.get("identity_key") == identity_key else None
                    else:
                        fit_start = time.monotonic()
                        result = fit_layer(bank, bits, fit_config, device=device, fit_mode=fit_mode)
                        seconds = time.monotonic() - fit_start
                    assert result is not None
                    errors = _check_fit_result_numerics(result, bank.geometry, path=path, bits=bits, lid=lid)
                    if not reused:
                        save_fit_checkpoint(ckpt_dir, result, identity_key=identity_key)
                        _atomic_json(timing_file, {"identity_key": identity_key, "seconds": seconds})
                    if fit_mode == "full":
                        full_fits[path][bits] = result
                    else:
                        scale_fits[path][bits] = result
                    width_details[fit_mode] = _fit_detail_entry(
                        result, errors, path=path, seconds=seconds, reused=reused
                    )
                details.setdefault(path, {})[str(bits)] = width_details

        candidates = layer_candidates(
            bank,
            full_fits["content"],
            full_fits["rope"],
            fit_config,
            content_scale_fits=scale_fits["content"],
            rope_scale_fits=scale_fits["rope"],
        )
        candidates = _persist_smoothquant_candidates(
            candidates, bank, ckpt_dir=ckpt_dir, identity_key=identity_key, device=device
        )
        val_table = measure_candidates(bank, candidates, modes, fit_config, device, split="val")
        test_table = measure_candidates(bank, candidates, modes, fit_config, device, split="test")
        selection = select_per_width(val_table, candidates, bank.geometry)
        v2_selection = select_shared_v2(val_table, candidates, bank.geometry)
        selection_by_layer[str(lid)] = selection
        v2_selection_by_layer[str(lid)] = v2_selection

        persisted_layers[str(lid)] = {
            "geometry": bank.geometry.to_dict(),
            "candidates": {
                "content": {cid: _candidate_catalog_entry("content", cid, res) for cid, res in candidates.content.items()},
                "rope": {cid: _candidate_catalog_entry("rope", cid, res) for cid, res in candidates.rope.items()},
            },
            "val_table": val_table,
            "test_table": test_table,
            "selection_v3": selection,
            "selection_v2_shared": v2_selection,
        }

        if selection["content"] is not None:
            for bkey, rec in selection["content"].items():
                w = int(bkey)
                cand = with_equivalence(candidates.content[rec["candidate"]], bank, device)
                for check, value in (
                    ("equiv_max_rel_err", cand.equiv_max_rel_err),
                    ("inverse_max_abs_err", inverse_max_error(cand, bank.geometry)),
                ):
                    if not math.isfinite(value) or (check == "equiv_max_rel_err" and value >= 1e-5):
                        raise ValueError(f"shipped content width {w} layer {lid} {check} = {value}")
                content_selections[w][lid] = cand
                assert selection_meta["content"] is not None
                selection_meta["content"][bkey][str(lid)] = {
                    "candidate": rec["candidate"],
                    "source_bits": rec["source_bits"],
                    "val_score": rec["val_score"],
                }

        if selection["rope"] is not None:
            for bkey, rec in selection["rope"].items():
                w = int(bkey)
                cand = with_equivalence(candidates.rope[rec["candidate"]], bank, device)
                for check, value in (
                    ("equiv_max_rel_err", cand.equiv_max_rel_err),
                    ("inverse_max_abs_err", inverse_max_error(cand, bank.geometry)),
                ):
                    if not math.isfinite(value) or (check == "equiv_max_rel_err" and value >= 1e-5):
                        raise ValueError(f"shipped RoPE width {w} layer {lid} {check} = {value}")
                rope_selections[w][lid] = cand
                assert selection_meta["rope"] is not None
                selection_meta["rope"][bkey][str(lid)] = {
                    "candidate": rec["candidate"],
                    "source_bits": rec["source_bits"],
                    "val_score": rec["val_score"],
                }

        layer_measurements[lid] = layer_rows(
            test_table, val_table, selection, bank.geometry, v2_selection=v2_selection
        )
        details["fit_and_measure_wall_seconds"] = time.monotonic() - layer_start
        layer_fit_details[str(lid)] = details

    combinations = summarize_modes(layer_measurements)

    fit_budget_json = {
        path: {str(b): list(active_budget[path][b]) for b in widths}
        for path, widths in (("content", c_bits_list), ("rope", r_bits_list if has_any_rope else ()))
    }
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
        "measurement_reference": f"{source['calibrated_view']} 0:0",
        **{
            k: corpus_provenance[k]
            for k in (
                "forward_view_fingerprint",
                "forward_revision",
                "forward_config_sha256",
                "forward_attention_arithmetic",
                "capture_input_trajectory",
                "capture_forward",
            )
            if k in corpus_provenance
        },
        "measurement_split": "test",
        "combination_measurements": {key: shipped_summary(row) for key, row in combinations.items()},
        "fit_budget": fit_budget_json,
        "optimizer": "adamw",
        "steps_per_path": fit_config.steps,
        "seed": fit_config.seed,
        "code_commit": code_commit or _git_head_sha(Path(__file__).resolve().parents[2]),
    }

    resolved_views = (
        list(applicable_views)
        if applicable_views is not None
        else (["all"] if source.get("checkpoint_kind") == "mc" else [source["calibrated_view"]])
    )
    candidate_table_doc = {
        "schema": CANDIDATE_TABLE_SCHEMA,
        "fit_checkpoint_schema": FIT_CHECKPOINT_SCHEMA,
        "identity_key": identity_key,
        "source": dict(source),
        "protection_policy": fit_config.protection,
        "modes": [f"{c}:{r}" for c, r in modes],
        "fit_budget": fit_budget_json,
        "applicable_views": resolved_views,
        "provenance": provenance,
        "layers": persisted_layers,
    }
    _atomic_json(output_dir / "candidate-table.json", candidate_table_doc)

    manifest = write_sidecar(
        sidecar_dir,
        source=source,
        geometries=geometries,
        content_selections=content_selections if c_bits_list else None,
        rope_selections=rope_selections if rope_selections else None,
        selection_meta=selection_meta,
        provenance=provenance,
        applicable_views=resolved_views,
        artifact_status="calibrated_unqualified",
        protection_policy=fit_config.protection,
    )
    validation = validate_sidecar(
        sidecar_dir,
        mc_qmla=manifest,
        expected_source={"repo": source["repo"], "revision": source["revision"], "family": source["family"]},
        expected_geometries=geometries,
        banks_by_layer=banks,
        device=device,
        equiv_tol=1e-5,
        strict=True,
    )

    headline_table = build_headline_table(
        combinations,
        manifest["profiles"],
        unsupported_reasons=manifest["unsupported_modes_reason"],
    )
    report = {
        "schema": VALIDATION_REPORT_SCHEMA,
        "headline_modes": list(HEADLINE_DEPLOYMENT_MODES),
        "headline_table": headline_table,
        "headline_table_markdown": format_headline_table(headline_table),
        "candidate_table_file": "candidate-table.json",
        "artifact_status": manifest["artifact_status"],
        "protection_policy": manifest["protection_policy"],
        "protection_scope": manifest["protection_scope"],
        "source": manifest["source"],
        "measurement_reference": provenance["measurement_reference"],
        **{
            key: provenance[key]
            for key in ("forward_attention_arithmetic", "capture_input_trajectory")
            if key in provenance
        },
        "layers_calibrated": ordered_layers,
        "splits": {name: splits[name]["count"] for name in SPLIT_NAMES},
        "measurement_split": "test",
        "selection_split": "val",
        "selection_objective": SELECTION_OBJECTIVE,
        "fit_budget": fit_budget_json,
        "combinations": combinations,
        "selection_by_layer": selection_by_layer,
        "v2_shared_selection_by_layer": v2_selection_by_layer,
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
            path: (
                None
                if prof is None
                else {
                    "bits": prof["bits"],
                    "layers": prof["layers"],
                    "widths": {
                        bkey: {
                            "width_fingerprint": w["width_fingerprint"],
                            "kind_counts": w["kind_counts"],
                            "chosen": {
                                lid: {"kind": row["kind"], "source_candidate": row["source_candidate"]}
                                for lid, row in w["per_layer"].items()
                            },
                        }
                        for bkey, w in prof["widths"].items()
                    },
                    "equiv_max_rel_err": prof["validation"]["equiv_max_rel_err"],
                }
            )
            for path, prof in manifest["profiles"].items()
        },
    }
    _atomic_json(output_dir / "qmla-validation-report.json", report)
    return {
        "manifest": manifest,
        "validation": validation,
        "report": report,
        "candidate_table": candidate_table_doc,
        "sidecar_dir": str(sidecar_dir),
    }


def reselect_from_persisted_candidates(
    run_dir: Path | str,
    *,
    output_dir: Path | str | None = None,
    selector: Callable[[Mapping[str, Mapping[str, Any]], LayerCandidates, MlaLayerGeometry], dict[str, Any]] = select_per_width,
) -> dict[str, Any]:
    """Re-run selection and emit a validated v3 sidecar from persisted v3 candidates with zero refitting.

    Reads ``<run_dir>/candidate-table.json`` and verified ``cuteafd-qmla-fit-checkpoint-v3``
    files under ``<run_dir>/fit-checkpoints/``. No capture banks or GPU are required.
    """
    run_dir = Path(run_dir).resolve()
    table_path = run_dir / "candidate-table.json"
    if not table_path.is_file():
        raise ValueError(f"persisted candidate table missing: {table_path}")
    table = json.loads(table_path.read_text(encoding="utf-8"))
    if table.get("schema") != CANDIDATE_TABLE_SCHEMA:
        raise ValueError(
            f"unsupported candidate table schema: {table.get('schema')!r} (expected {CANDIDATE_TABLE_SCHEMA!r})"
        )
    identity_key = str(table["identity_key"])
    ckpt_dir = run_dir / "fit-checkpoints"
    out_dir = Path(output_dir).resolve() if output_dir is not None else run_dir
    sidecar_dir = out_dir / "sidecar"
    sidecar_dir.mkdir(parents=True, exist_ok=True)

    modes = tuple(parse_qmla_mode(m) for m in table["modes"])
    c_bits_list, r_bits_list = required_profile_bits(modes)
    ordered_layers = sorted(int(lid) for lid in table["layers"])
    geometries = [MlaLayerGeometry.from_dict(table["layers"][str(lid)]["geometry"]) for lid in ordered_layers]
    has_any_rope = any(g.has_rope for g in geometries)
    policy = str(table["protection_policy"])

    content_selections: dict[int, dict[int, ContentFitResult]] = {b: {} for b in c_bits_list}
    rope_selections: dict[int, dict[int, RopeFitResult]] = {b: {} for b in r_bits_list} if (r_bits_list and has_any_rope) else {}
    selection_meta: dict[str, dict[str, dict[str, Any]] | None] = {
        "content": {str(b): {} for b in c_bits_list} if c_bits_list else None,
        "rope": {str(b): {} for b in r_bits_list} if (r_bits_list and has_any_rope) else None,
    }
    selection_by_layer: dict[str, Any] = {}
    v2_selection_by_layer: dict[str, Any] = {}
    layer_measurements = {}

    for lid, geom in zip(ordered_layers, geometries, strict=True):
        lentry = table["layers"][str(lid)]
        c_pool: dict[str, ContentFitResult] = {}
        r_pool: dict[str, RopeFitResult] = {}
        for path, pool in (("content", c_pool), ("rope", r_pool)):
            for cid, meta in lentry["candidates"][path].items():
                ckind, _, src_bits = parse_candidate_id(cid)
                if ckind == "identity":
                    if path == "content":
                        pool[cid] = _identity_content_result(lid, geom.kv_lora_rank, policy)
                    else:
                        pool[cid] = _identity_rope_result(lid, geom.qk_rope_dim // 2, policy)
                else:
                    assert src_bits is not None
                    loaded = load_fit_checkpoint(
                        ckpt_dir,
                        kind=path,
                        bits=src_bits,
                        layer_id=lid,
                        identity_key=identity_key,
                        fit_mode=str(meta["fit_mode"]),
                    )
                    if loaded is None:
                        raise ValueError(f"missing persisted v3 candidate checkpoint for layer {lid} {cid}")
                    pool[cid] = loaded

        candidates = LayerCandidates(content=c_pool, rope=r_pool)
        val_table = lentry["val_table"]
        test_table = lentry["test_table"]
        selection = selector(val_table, candidates, geom)
        v2_selection = select_shared_v2(val_table, candidates, geom)
        selection_by_layer[str(lid)] = selection
        v2_selection_by_layer[str(lid)] = v2_selection

        if selection["content"] is not None:
            for bkey, rec in selection["content"].items():
                w = int(bkey)
                content_selections[w][lid] = candidates.content[rec["candidate"]]
                assert selection_meta["content"] is not None
                selection_meta["content"][bkey][str(lid)] = {
                    "candidate": rec["candidate"],
                    "source_bits": rec["source_bits"],
                    "val_score": rec["val_score"],
                }
        if selection["rope"] is not None:
            for bkey, rec in selection["rope"].items():
                w = int(bkey)
                rope_selections[w][lid] = candidates.rope[rec["candidate"]]
                assert selection_meta["rope"] is not None
                selection_meta["rope"][bkey][str(lid)] = {
                    "candidate": rec["candidate"],
                    "source_bits": rec["source_bits"],
                    "val_score": rec["val_score"],
                }
        layer_measurements[lid] = layer_rows(test_table, val_table, selection, geom, v2_selection=v2_selection)

    combinations = summarize_modes(layer_measurements)
    provenance = dict(table["provenance"])
    provenance["combination_measurements"] = {key: shipped_summary(row) for key, row in combinations.items()}
    source = dict(table["source"])

    manifest = write_sidecar(
        sidecar_dir,
        source=source,
        geometries=geometries,
        content_selections=content_selections if c_bits_list else None,
        rope_selections=rope_selections if rope_selections else None,
        selection_meta=selection_meta,
        provenance=provenance,
        applicable_views=table.get("applicable_views"),
        artifact_status="calibrated_unqualified",
        protection_policy=policy,
    )
    validation = validate_sidecar(
        sidecar_dir,
        mc_qmla=manifest,
        expected_source={"repo": source["repo"], "revision": source["revision"], "family": source["family"]},
        expected_geometries=geometries,
        strict=True,
    )
    headline_table = build_headline_table(
        combinations,
        manifest["profiles"],
        unsupported_reasons=manifest["unsupported_modes_reason"],
    )
    return {
        "headline_table": headline_table,
        "headline_table_markdown": format_headline_table(headline_table),
        "manifest": manifest,
        "validation": validation,
        "combinations": combinations,
        "selection_by_layer": selection_by_layer,
        "v2_shared_selection_by_layer": v2_selection_by_layer,
        "sidecar_dir": str(sidecar_dir),
    }


def build_run_parser(
    description: str, *, target_repo: str, base_repo: str, base_revision: str
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Local HF snapshot directory")
    parser.add_argument("--source-repo", default=target_repo, help="Target/source HF repo id")
    parser.add_argument("--source-revision", required=True, help="Pinned 40-char immutable source revision")
    parser.add_argument("--base-repo", default=base_repo)
    parser.add_argument("--base-revision", default=base_revision)
    parser.add_argument("--checkpoint-kind", choices=("mc", "plain"), default="mc")
    parser.add_argument("--view", choices=("original",), default="original", help="Core view used for forward capture")
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=None,
        help="Prepared panel_corpus dir (prepare_panel_corpus output, windowed long_context)",
    )
    parser.add_argument(
        "--panel-revision",
        default=None,
        help="Pinned panel revision from panel_corpus (default: agentic-quant-panel-v3)",
    )
    parser.add_argument(
        "--components",
        default=None,
        help="Panel components (default: all of the panel's; a full run requires all of them)",
    )
    parser.add_argument("--output", type=Path, required=True, help="Task output/work directory on root NVMe")
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Family pilot layers and modes; 16/4/4 train/val/test sequences; 100 steps",
    )
    parser.add_argument("--layers", default=None, help="Layer selection: 'all-dsa' or comma-separated IDs")
    parser.add_argument("--modes", default=None, help="Comma-separated QUANT_MLA modes (e.g. 4:4 or 8:8,4:8,4:4,2:4)")
    parser.add_argument(
        "--protection",
        choices=sorted(VALID_PROTECTION_POLICIES),
        default=DEFAULT_PROTECTION_POLICY,
        help="Recent-token protection policy on full-history MLA layers (default: sink4_local128; SWA untouched)",
    )
    parser.add_argument("--train-sequences", type=int, default=None)
    parser.add_argument("--val-sequences", type=int, default=None, help="Step and per-width selection split")
    parser.add_argument("--test-sequences", type=int, default=None, help="Held-out split for every reported error")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--reuse-captures",
        type=Path,
        default=None,
        help="Reuse verified XXH3-128 layer captures only when identical checkpoint/corpus/splits/forward match",
    )
    parser.add_argument(
        "--long-eval",
        action="store_true",
        help="Run 16K/32K long-context evaluation after fitting, or standalone with --sidecar",
    )
    parser.add_argument(
        "--sidecar",
        type=Path,
        default=None,
        help="Existing QMLA v3 sidecar (mla_calibration.safetensors or directory) for standalone --long-eval",
    )
    return parser


def resolve_settings(
    args: argparse.Namespace,
    eligible_layers: Sequence[int],
    *,
    pilot_layers: Sequence[int],
    pilot_modes: Sequence[str],
    full_modes: Sequence[str],
) -> dict[str, Any]:
    """Resolve explicit CLI flags on top of pilot or full defaults."""
    if args.pilot:
        default_layers = ",".join(str(x) for x in pilot_layers if x in eligible_layers) or str(eligible_layers[0])
        default_modes = ",".join(pilot_modes)
        (default_train, default_val, default_test), default_steps = PILOT_SPLITS, 100
    else:
        default_layers = "all-dsa"
        default_modes = ",".join(full_modes)
        (default_train, default_val, default_test), default_steps = FULL_SPLITS, 300

    layers = parse_layers_arg(args.layers or default_layers, eligible_layers)
    modes = parse_modes_arg(args.modes or default_modes)
    full = not args.pilot
    _, panel_revision, components = resolve_qmla_panel(
        panel_revision=args.panel_revision,
        components=None if args.components is None else parse_components_arg(args.components),
        full=full,
    )
    train_n = default_train if args.train_sequences is None else int(args.train_sequences)
    val_n = default_val if args.val_sequences is None else int(args.val_sequences)
    test_n = default_test if args.test_sequences is None else int(args.test_sequences)
    steps = default_steps if args.steps is None else int(args.steps)
    return {
        "layers": layers,
        "modes": modes,
        "panel_revision": panel_revision,
        "components": components,
        "full": full,
        "protection": args.protection,
        "train_sequences": train_n,
        "val_sequences": val_n,
        "test_sequences": test_n,
        "steps": steps,
    }


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
        token_records,
        train_count=train_count,
        val_count=val_count,
        test_count=test_count,
        key_fn=lambda r: r[0],
        seed=seed,
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
    if (
        corpus_provenance["dataset_revision"] != resolved["panel_revision"]
        or set(corpus_provenance["components"]) != set(resolved["components"])
    ):
        raise ValueError("capture panel revision or components differ from the requested run")
    splits = split_identity(banks)
    if splits["seed"] != seed:
        raise ValueError(f"capture split seed {splits['seed']} differs from the requested seed {seed}")
    for bank in banks.values():
        if (
            "forward_attention_arithmetic" in teacher_identity
            and bank.params.attention_arithmetic != teacher_identity["forward_attention_arithmetic"]
        ):
            raise ValueError(f"capture content cache arithmetic differs at layer {bank.geometry.layer_id}")
        if bank.provenance != corpus_provenance:
            raise ValueError(f"capture provenance differs at layer {bank.geometry.layer_id}")
        for name in SPLIT_NAMES:
            if len(bank.split(name)) != resolved[f"{name}_sequences"]:
                raise ValueError(f"capture {name} split count differs at layer {bank.geometry.layer_id}")
    if expected_splits is not None and dict(expected_splits) != splits:
        raise ValueError("capture split membership differs from the split of the requested corpus")
    return corpus_provenance


def attach_long_context_eval(
    output_dir: Path | str,
    long_eval_section: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Attach ``long_context_eval`` into ``<output>/qmla-validation-report.json`` and rebuild the combined short+long headline table."""
    output_dir = Path(output_dir).resolve()
    report_path = output_dir / "qmla-validation-report.json"
    if report_path.is_file():
        existing = json.loads(report_path.read_text(encoding="utf-8"))
        if "combinations" in existing:
            profiles = manifest["profiles"] if manifest is not None else existing.get("shipped_profiles")
            reasons = manifest["unsupported_modes_reason"] if manifest is not None else None
            combined_headline = build_headline_table(
                existing["combinations"],
                profiles,
                long_combinations=long_eval_section.get("combinations"),
                unsupported_reasons=reasons,
            )
            combined_md = format_headline_table(combined_headline)
        else:
            combined_headline = long_eval_section.get("headline_table")
            combined_md = long_eval_section.get("headline_table_markdown")
        report: dict[str, Any] = {
            "schema": existing.get("schema", VALIDATION_REPORT_SCHEMA),
            "headline_modes": list(HEADLINE_DEPLOYMENT_MODES),
            "headline_table": combined_headline,
            "headline_table_markdown": combined_md,
        }
        for k, v in existing.items():
            if k not in report:
                report[k] = v
    else:
        if manifest is None:
            raise ValueError("manifest is required when creating a new qmla-validation-report.json")
        report = {
            "schema": VALIDATION_REPORT_SCHEMA,
            "headline_modes": list(HEADLINE_DEPLOYMENT_MODES),
            "headline_table": long_eval_section.get("headline_table"),
            "headline_table_markdown": long_eval_section.get("headline_table_markdown"),
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
