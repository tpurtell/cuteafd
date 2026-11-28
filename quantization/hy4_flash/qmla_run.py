"""End-to-end Hy4-flash QMLA calibration driver (reference MLA pipeline to copy).

Supports both short pilot runs (``--pilot`` or explicit ``--layers``, ``--modes``,
``--train-sequences``, ``--val-sequences``, ``--steps``) and full 12-DSA-layer
calibration over all principal modes.

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
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qmla.capture import (
    CapturedLayerBank,
    MlaLayerGeometry,
    _atomic_json,
    load_layer_capture,
    save_layer_capture,
)
from qmla.fit import (
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
from qmla.quantizer import parse_qmla_mode
from qmla.sidecar import (
    DEFAULT_QMLA_PANEL_REVISION,
    QMLA_BASE_COMPONENTS,
    QMLA_KNOWN_COMPONENTS,
    resolve_qmla_panel,
    validate_sidecar,
    write_sidecar,
)

from . import forward as hy4_forward
from . import recipe as hy4_recipe
from .qmla_capture import Hy4FlashCaptureAdapter

DEFAULT_PILOT_LAYERS = (0, 21, 41)
DEFAULT_PILOT_MODES = ("4:4",)
DEFAULT_FULL_MODES = ("8:8", "4:8", "4:4", "2:4", "4:0", "0:4", "8:0", "0:8", "2:0")


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


def calibrate_hy4_flash_from_banks(
    banks: Mapping[int, CapturedLayerBank],
    *,
    output_dir: Path | str,
    source: Mapping[str, Any],
    modes: Sequence[tuple[int, int]],
    corpus_provenance: Mapping[str, Any],
    fit_config: FitConfig = FitConfig(),
    applicable_views: Sequence[str] = ("original", "k8-compact", "k6-compact", "fp8-compact"),
    view_transfer_validation: Mapping[str, Any] | None = None,
    device: torch.device | str = "cpu",
    code_commit: str | None = None,
    thinking_mode: str = "high",
    renderer_template_sha256: str = "",
) -> dict[str, Any]:
    """Run fit -> checkpoint/resume -> single-file sidecar -> validation from captured layer banks.

    Shared by the production CLI driver and CPU numeric oracle tests.
    """
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

    identity_key = fit_config_digest(
        fit_config,
        extra={
            "source_revision": source["revision"],
            "layers": ordered_layers,
            "corpus": dict(corpus_provenance),
        },
    )

    content_profiles: dict[int, list[ContentFitResult]] = {b: [] for b in c_bits_list}
    rope_profiles: dict[int, list[RopeFitResult]] = {b: [] for b in r_bits_list}

    for lid in ordered_layers:
        bank = banks[lid]
        bank.validate()
        for c_bits in c_bits_list:
            cached = load_fit_checkpoint(
                ckpt_dir, kind="content", bits=c_bits, layer_id=lid, identity_key=identity_key
            )
            if cached is not None:
                assert isinstance(cached, ContentFitResult)
                res_c = cached
            else:
                res_c = fit_content_layer(bank, c_bits, fit_config, device=device)
                save_fit_checkpoint(ckpt_dir, res_c, identity_key=identity_key)
            content_profiles[c_bits].append(res_c)

        if bank.geometry.has_rope:
            for r_bits in r_bits_list:
                cached = load_fit_checkpoint(
                    ckpt_dir, kind="rope", bits=r_bits, layer_id=lid, identity_key=identity_key
                )
                if cached is not None:
                    assert isinstance(cached, RopeFitResult)
                    res_r = cached
                else:
                    res_r = fit_rope_layer(bank, r_bits, fit_config, device=device)
                    save_fit_checkpoint(ckpt_dir, res_r, identity_key=identity_key)
                rope_profiles[r_bits].append(res_r)

    rope_profiles = {b: rows for b, rows in rope_profiles.items() if rows}

    sample_bank = banks[ordered_layers[0]]
    provenance: dict[str, Any] = {
        "dataset_repo": corpus_provenance["dataset_repo"],
        "dataset_revision": corpus_provenance["dataset_revision"],
        "components": corpus_provenance["components"],
        "train_sequences": len(sample_bank.train),
        "val_sequences": len(sample_bank.val),
        "input_ids_sha256": corpus_provenance["input_ids_sha256"],
        "renderer_template_sha256": renderer_template_sha256,
        "thinking_mode": thinking_mode,
        "optimizer": "adamw",
        "steps_per_path": fit_config.steps,
        "seed": fit_config.seed,
        "code_commit": code_commit or _git_head_sha(Path(__file__).resolve().parents[2]),
    }

    manifest = write_sidecar(
        sidecar_dir,
        source=source,
        geometries=geometries,
        content_profiles=content_profiles,
        rope_profiles=rope_profiles,
        provenance=provenance,
        applicable_views=applicable_views,
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
        strict=True,
    )
    report = {
        "schema": "cuteafd-qmla-validation-report-v1",
        "artifact_status": manifest["artifact_status"],
        "protection_policy": manifest["protection_policy"],
        "protection_scope": manifest["protection_scope"],
        "source": manifest["source"],
        "layers_calibrated": ordered_layers,
        "supported_modes": manifest["supported_modes"],
        "validation": validation,
        "content_summary": {
            k: {
                "val_loss_mean": v["validation"]["val_loss_mean"],
                "equiv_max_rel_err": v["validation"]["equiv_max_rel_err"],
                "orth_max_err": v["validation"]["orth_max_err"],
            }
            for k, v in manifest["profiles"]["content"].items()
        },
        "rope_summary": {
            k: {
                "val_loss_mean": v["validation"]["val_loss_mean"],
                "equiv_max_rel_err": v["validation"]["equiv_max_rel_err"],
            }
            for k, v in manifest["profiles"]["rope"].items()
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
    parser.add_argument("--pilot", action="store_true", help="Short pilot preset (layers 0,21,41; mode 4:4; 16/4 seqs; 100 steps)")
    parser.add_argument("--layers", default=None, help="Layer selection: 'all-dsa' or comma-separated IDs (e.g. 0,21,41)")
    parser.add_argument("--modes", default=None, help="Comma-separated QUANT_MLA modes (e.g. 4:4 or 8:8,4:8,4:4,2:4)")
    parser.add_argument(
        "--protection",
        choices=sorted(VALID_PROTECTION_POLICIES),
        default=DEFAULT_PROTECTION_POLICY,
        help="Recent-token protection policy on full-history MLA layers (default: sink4_local128; SWA untouched)",
    )
    parser.add_argument("--train-sequences", type=int, default=None)
    parser.add_argument("--val-sequences", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--reuse-captures", type=Path, default=None, help="Reuse verified XXH3-128 layer captures")
    return parser


def resolve_run_settings(args: argparse.Namespace, eligible_layers: Sequence[int]) -> dict[str, Any]:
    """Resolve explicit CLI flags on top of pilot or full defaults."""
    if args.pilot:
        default_layers = ",".join(str(x) for x in DEFAULT_PILOT_LAYERS if x in eligible_layers) or str(eligible_layers[0])
        default_modes = ",".join(DEFAULT_PILOT_MODES)
        default_train, default_val, default_steps = 16, 4, 100
    else:
        default_layers = "all-dsa"
        default_modes = ",".join(DEFAULT_FULL_MODES)
        default_train, default_val, default_steps = 128, 32, 300

    layers = parse_layers_arg(args.layers or default_layers, eligible_layers)
    modes = parse_modes_arg(args.modes or default_modes)
    components = parse_components_arg(args.components)
    train_n = default_train if args.train_sequences is None else int(args.train_sequences)
    val_n = default_val if args.val_sequences is None else int(args.val_sequences)
    steps = default_steps if args.steps is None else int(args.steps)
    return {
        "layers": layers,
        "modes": modes,
        "panel_revision": args.panel_revision,
        "components": components,
        "protection": args.protection,
        "train_sequences": train_n,
        "val_sequences": val_n,
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


@torch.no_grad()
def _stream_hy4_flash_captures(
    reader: hy4_forward.Reader,
    adapter: Hy4FlashCaptureAdapter,
    target_layers: Sequence[int],
    token_records: Sequence[tuple[str, torch.Tensor]],
    *,
    train_count: int,
    val_count: int,
    seed: int,
    capture_dir: Path,
    provenance: dict[str, Any],
) -> dict[int, CapturedLayerBank]:
    """Run one frozen-weight layer-major forward up to max(target_layers) and capture target MLA layers."""
    model = adapter.model
    max_layer = max(target_layers)
    target_set = set(target_layers)
    embedding = reader.raw(hy4_forward.reference.OUTER_NAMES[0])

    from qmla.capture import split_train_val

    train_items, val_items = split_train_val(
        token_records, train_count=train_count, val_count=val_count, key_fn=lambda r: r[0], seed=seed
    )
    active = [*train_items, *val_items]
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
                                     tuple(captures[:train_count]), tuple(captures[train_count:]), provenance)
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    checkpoint = args.checkpoint.resolve()
    config_bytes = (checkpoint / "config.json").read_bytes()
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()

    teacher = args.teacher_checkpoint or checkpoint
    if args.teacher_checkpoint is not None:
        manifest = json.loads((checkpoint / "cuteafd-mc.json").read_text())
        if (checkpoint.name != args.source_revision or teacher.name != args.base_revision
                or manifest["source"] != {"repo": args.base_repo, "revision": args.base_revision}):
            raise ValueError("BF16 teacher or MC source differs from the pinned calibration identities")
    reader = hy4_forward.Reader(teacher, device=args.device)
    model = hy4_forward.Model(reader)
    adapter = Hy4FlashCaptureAdapter(model, include_swa=False)
    eligible = tuple(g.layer_id for g in adapter.mla_geometries(include_swa=False))
    resolved = resolve_run_settings(args, eligible)

    if args.corpus_dir is None and args.reuse_captures is None:
        parser.error("provide --corpus-dir PATH or --reuse-captures DIR")

    capture_dir = (args.reuse_captures or (args.output / "captures")).resolve()
    if args.reuse_captures is not None:
        banks = {lid: load_layer_capture(capture_dir, lid) for lid in resolved["layers"]}
        corpus_provenance = dict(banks[resolved["layers"][0]].provenance)
    else:
        assert args.corpus_dir is not None
        all_records, corpus_provenance = load_pinned_qmla_corpus(
            args.corpus_dir,
            panel_revision=resolved["panel_revision"],
            components=resolved["components"],
        )
        banks = _stream_hy4_flash_captures(
            reader,
            adapter,
            resolved["layers"],
            all_records,
            train_count=resolved["train_sequences"],
            val_count=resolved["val_sequences"],
            seed=args.seed,
            capture_dir=capture_dir,
            provenance=corpus_provenance,
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
    print(
        json.dumps({
            "status": result["validation"]["status"],
            "artifact_status": result["manifest"]["artifact_status"],
            "protection_policy": result["manifest"]["protection_policy"],
            "layers": list(resolved["layers"]),
            "supported_modes": result["manifest"]["supported_modes"],
            "transform_fingerprint": result["manifest"]["transform_fingerprint"],
            "sidecar_dir": result["sidecar_dir"],
        }),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
