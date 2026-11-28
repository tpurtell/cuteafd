"""Shared MLA calibration, split verification, fit checkpoints and sidecar reporting."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import resource
import subprocess
import time
from typing import Any, Mapping, Sequence

import torch

import panel_corpus
from qmla.capture import (SPLIT_NAMES, CapturedLayerBank, MlaLayerGeometry, _atomic_json,
                          check_split_description, describe_splits, sequence_sha256, split_train_val_test)
from qmla.fit import (_require_unprotected_tokens, DEFAULT_PROTECTION_POLICY, VALID_PROTECTION_POLICIES,
                      ContentFitResult, FitConfig, RopeFitResult, check_content_identity, fit_config_digest, fit_content_layer,
                      fit_rope_layer, load_fit_checkpoint, save_fit_checkpoint)
from qmla.measurement import (SELECTION_OBJECTIVE, inverse_max_error, layer_candidates, layer_rows,
                              measure_candidates, measure_shipped, select_shipped, shipped_summary,
                              summarize_modes, with_equivalence)
from qmla.quantizer import parse_qmla_mode
from qmla.sidecar import CANONICAL_CONVENTIONS, resolve_qmla_panel, validate_sidecar, write_sidecar


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
    """Load a prepared ``panel_corpus`` calibration split through ``load_component_records``.

    The panel resolves through ``panel_corpus.resolve_panel_policy`` (default v3, all three
    components; ``full`` requires every component). The prepared corpus must carry exactly
    the pinned dataset identity; ``long_context`` arrives as 2,048-token windows.
    """
    dataset_repo, dataset_revision, component_names = resolve_qmla_panel(
        panel_revision=panel_revision, components=components, full=full)
    pin = panel_corpus.panel_pin(dataset_revision)
    rows, prov = panel_corpus.load_component_records(
        corpus_dir, "calibration", components=component_names, full=full, long_context_mode="windowed",
        dataset=dict(repo_id=pin.repo_id, revision=pin.revision, sha256sums_sha256=pin.sha256sums_sha256),
    )
    # Capture seq_ids may not contain "/" (exl3_standard/000, lc-16k-cal-01/w00).
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


def _shipped_result(candidates, record, bank, device):
    if record is None:
        return None
    pool = candidates.content if record["candidate"].split(":")[1][0] == "c" else candidates.rope
    return with_equivalence(pool[record["candidate"]], bank, device)


def calibrate_from_banks(
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
        raise ValueError("calibrate_from_banks requires at least one CapturedLayerBank")
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
            "fit_semantics": "three-way-split-post-norm-fp64-validation-rotation",
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
        if c_bits_list:
            check_content_identity((seq.to(device) for seq in bank.val), bank.params.to(device), bank.geometry)
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
        "measurement_reference": f"{source['calibrated_view']} 0:0",
        **{k: corpus_provenance[k] for k in ("forward_view_fingerprint", "forward_revision",
           "forward_config_sha256", "forward_attention_arithmetic", "capture_input_trajectory", "capture_forward") if k in corpus_provenance},
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
        device=device,
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
        "measurement_reference": provenance["measurement_reference"],
        **{key: provenance[key] for key in ("forward_attention_arithmetic", "capture_input_trajectory") if key in provenance},
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


def build_run_parser(description: str, *, target_repo: str, base_repo: str, base_revision: str) -> argparse.ArgumentParser:
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
    parser.add_argument("--pilot", action="store_true",
                        help="Family pilot layers and modes; 16/4/4 train/val/test sequences; 100 steps")
    parser.add_argument("--layers", default=None, help="Layer selection: 'all-dsa' or comma-separated IDs ")
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


def resolve_settings(args: argparse.Namespace, eligible_layers: Sequence[int], *, pilot_layers: Sequence[int],
                     pilot_modes: Sequence[str], full_modes: Sequence[str]) -> dict[str, Any]:
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
        if ("forward_attention_arithmetic" in teacher_identity
                and bank.params.attention_arithmetic != teacher_identity["forward_attention_arithmetic"]):
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
