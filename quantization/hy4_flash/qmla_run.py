"""End-to-end Hy4-flash QMLA calibration driver (reference MLA pipeline to copy).

Supports both short pilot runs (``--pilot`` or explicit ``--layers``, ``--modes``,
``--train-sequences``, ``--val-sequences``, ``--steps``) and full 12-DSA-layer
calibration over all principal modes.

Corpus components are data-driven via repeatable ``--corpus-component`` flags
(defaulting to the pinned ``agentic`` and ``exl3_standard`` streams from
``wrldsuksgo2mars/agentic-quant-panel-v2``, and accepting optional long-context
components such as 16K/32K traces without code changes).

Outputs:
- ``<output>/sidecar/mla_calibration.safetensors``
- ``<output>/sidecar/mla_calibration.json`` (``artifact_status="calibrated_unqualified"``)
- ``<output>/qmla-validation-report.json``
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
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
from qmla.sidecar import CorpusComponentSpec, validate_sidecar, write_sidecar

from . import forward as hy4_forward
from . import prepare_corpus as hy4_corpus
from . import recipe as hy4_recipe
from .qmla_capture import Hy4FlashCaptureAdapter, hy4_flash_mla_geometries

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


def parse_corpus_component_arg(spec: str) -> dict[str, str]:
    """Parse ``NAME=REPO@REVISION:PATH`` data-driven corpus component descriptor."""
    if "=" not in spec or "@" not in spec or ":" not in spec:
        raise ValueError(
            f"invalid --corpus-component {spec!r}: expected NAME=REPO@REVISION:PATH"
        )
    name, rest = spec.split("=", 1)
    repo_rev, path_str = rest.rsplit(":", 1)
    if "@" not in repo_rev:
        raise ValueError(f"invalid --corpus-component {spec!r}: missing @REVISION")
    repo, revision = repo_rev.split("@", 1)
    if not name or not repo or not revision or not path_str:
        raise ValueError(f"incomplete --corpus-component {spec!r}")
    return {"name": name.strip(), "repo": repo.strip(), "revision": revision.strip(), "path": path_str.strip()}


def load_token_stream_component(
    name: str,
    repo: str,
    revision: str,
    path: Path | str,
    *,
    max_sequences: int | None = None,
) -> tuple[list[tuple[str, torch.Tensor]], CorpusComponentSpec]:
    """Load one prepared token-stream component (safetensors ``input_ids`` + ``offsets``).

    Works identically for ``agentic``, ``exl3_standard``, and any optional
    long-context bank (e.g. 16K/32K traces) prepared in the same canonical format.
    """
    from safetensors.torch import load_file

    path = Path(path)
    if path.is_dir():
        st_path = path / "calibration.safetensors"
    else:
        st_path = path
    if not st_path.is_file():
        raise ValueError(f"corpus component {name!r} token file missing: {st_path}")

    data = load_file(str(st_path))
    if "input_ids" not in data or "offsets" not in data:
        raise ValueError(f"corpus component {name!r} must contain input_ids and offsets tensors")
    ids = data["input_ids"].to(torch.int64).contiguous()
    offsets = data["offsets"].to(torch.int64).contiguous().tolist()
    if len(offsets) < 2 or offsets[0] != 0 or offsets[-1] != int(ids.numel()):
        raise ValueError(f"corpus component {name!r} has malformed offsets")

    records: list[tuple[str, torch.Tensor]] = []
    hasher = hashlib.sha256()
    limit = len(offsets) - 1 if max_sequences is None else min(len(offsets) - 1, max_sequences)
    total_tokens = 0
    for idx in range(limit):
        seq = ids[offsets[idx] : offsets[idx + 1]]
        if seq.numel() == 0:
            continue
        seq_id = f"{name}:{idx:05d}"
        records.append((seq_id, seq))
        hasher.update(seq.numpy().tobytes())
        total_tokens += int(seq.numel())

    spec = CorpusComponentSpec(
        name=name,
        repo=repo,
        revision=revision,
        sequences=len(records),
        tokens=total_tokens,
        input_ids_sha256=hasher.hexdigest(),
    )
    spec.validate()
    return records, spec


def calibrate_hy4_flash_from_banks(
    banks: Mapping[int, CapturedLayerBank],
    *,
    output_dir: Path | str,
    source: Mapping[str, Any],
    modes: Sequence[tuple[int, int]],
    corpus_components: Sequence[CorpusComponentSpec],
    fit_config: FitConfig = FitConfig(),
    applicable_views: Sequence[str] = ("original", "k8-compact", "k6-compact", "fp8-compact"),
    view_transfer_validation: Mapping[str, Any] | None = None,
    device: torch.device | str = "cpu",
    code_commit: str | None = None,
    thinking_mode: str = "high",
    renderer_template_sha256: str = "",
) -> dict[str, Any]:
    """Run fit -> checkpoint/resume -> sidecar -> validation from captured layer banks.

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
            "components": [c.to_dict() for c in corpus_components],
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

    # Drop empty RoPE profiles if every selected layer is NoPE.
    rope_profiles = {b: rows for b, rows in rope_profiles.items() if rows}

    sample_bank = banks[ordered_layers[0]]
    combined_input_sha = hashlib.sha256(
        "\n".join(f"{c.name}:{c.input_ids_sha256}" for c in corpus_components).encode("utf-8")
    ).hexdigest()
    provenance: dict[str, Any] = {
        "components": [c.to_dict() for c in corpus_components],
        "train_sequences": len(sample_bank.train),
        "val_sequences": len(sample_bank.val),
        "input_ids_sha256": combined_input_sha,
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
    )
    validation = validate_sidecar(
        sidecar_dir,
        expected_source={"repo": source["repo"], "revision": source["revision"], "family": source["family"]},
        expected_geometries=geometries,
        banks_by_layer=banks,
        strict=True,
    )
    report = {
        "schema": "cuteafd-qmla-validation-report-v1",
        "artifact_status": manifest["artifact_status"],
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
    parser.add_argument("--source-repo", default=hy4_recipe.TARGET_REPO, help="Target/source HF repo id")
    parser.add_argument("--source-revision", required=True, help="Pinned 40-char immutable source revision")
    parser.add_argument("--base-repo", default=hy4_recipe.BASE_REPO)
    parser.add_argument("--base-revision", default=hy4_recipe.BASE_REVISION)
    parser.add_argument("--checkpoint-kind", choices=("mc", "plain"), default="mc")
    parser.add_argument("--view", default="original", help="Core view used for forward capture")
    parser.add_argument(
        "--corpus-component",
        action="append",
        default=[],
        metavar="NAME=REPO@REV:PATH",
        help="Repeatable data-driven corpus component (agentic, exl3_standard, optional long-context bank)",
    )
    parser.add_argument("--output", type=Path, required=True, help="Task output/work directory on root NVMe")
    parser.add_argument("--pilot", action="store_true", help="Short pilot preset (layers 0,21,41; mode 4:4; 16/4 seqs; 100 steps)")
    parser.add_argument("--layers", default=None, help="Layer selection: 'all-dsa' or comma-separated IDs (e.g. 0,21,41)")
    parser.add_argument("--modes", default=None, help="Comma-separated QUANT_MLA modes (e.g. 4:4 or 8:8,4:8,4:4,2:4)")
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
    train_n = default_train if args.train_sequences is None else int(args.train_sequences)
    val_n = default_val if args.val_sequences is None else int(args.val_sequences)
    steps = default_steps if args.steps is None else int(args.steps)
    return {
        "layers": layers,
        "modes": modes,
        "train_sequences": train_n,
        "val_sequences": val_n,
        "steps": steps,
    }


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

    # Keep only the selected subset (train_count + val_count) in the layer-major wavefront.
    from qmla.capture import split_train_val

    train_items, val_items = split_train_val(
        token_records, train_count=train_count, val_count=val_count, key_fn=lambda r: r[0], seed=seed
    )
    active = [*train_items, *val_items]
    states = [model.initial(tokens, embedding=embedding) for _, tokens in active]
    banks: dict[int, CapturedLayerBank] = {}

    for layer_idx in range(max_layer + 1):
        layer = model.load_layer(layer_idx)
        next_states: list[torch.Tensor] = []
        captured_inputs: list[tuple[str, torch.Tensor, torch.Tensor | None]] = []

        for (seq_id, tokens), state in zip(active, states, strict=True):
            residual, attention_post, attn_in = model.attention_start(state, tokens, layer)
            if layer_idx in target_set:
                captured_inputs.append((seq_id, attn_in.detach().cpu(), None))
            attn_out, _ = model.attention(attn_in, layer)
            if layer_idx < max_layer:
                st, post, mlp_in = model.mlp_start(residual, attention_post, attn_out, layer)
                routed = None
                if model.config.mlp_layer_types[layer_idx] == "sparse":
                    latent, ids, gamma = model.moe_route(mlp_in, layer)
                    routed = torch.zeros_like(latent)
                    for expert in ids.unique().tolist():
                        rows, weights = model.expert_rows(ids, gamma, expert)
                        routed[rows] += model.expert(expert, latent[rows], layer) * weights[:, None]
                next_states.append(model.mlp_finish(st, post, mlp_in, layer, routed))

        if layer_idx in target_set:
            bank = adapter.capture_layer(
                layer,
                captured_inputs,
                train_count=train_count,
                val_count=val_count,
                seed=seed,
                provenance=provenance,
            )
            save_layer_capture(capture_dir, bank)
            banks[layer_idx] = bank
        states = next_states
        del layer
    return banks


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    checkpoint = args.checkpoint.resolve()
    config_bytes = (checkpoint / "config.json").read_bytes()
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()

    reader = hy4_forward.Reader(checkpoint, device=args.device)
    model = hy4_forward.Model(reader)
    adapter = Hy4FlashCaptureAdapter(model, include_swa=False)
    eligible = tuple(g.layer_id for g in adapter.mla_geometries(include_swa=False))
    resolved = resolve_run_settings(args, eligible)

    if not args.corpus_component and args.reuse_captures is None:
        parser.error("provide at least one --corpus-component NAME=REPO@REV:PATH or --reuse-captures DIR")

    component_specs: list[CorpusComponentSpec] = []
    all_records: list[tuple[str, torch.Tensor]] = []
    for raw in args.corpus_component:
        desc = parse_corpus_component_arg(raw)
        recs, spec = load_token_stream_component(
            desc["name"], desc["repo"], desc["revision"], desc["path"]
        )
        all_records.extend(recs)
        component_specs.append(spec)

    capture_dir = (args.reuse_captures or (args.output / "captures")).resolve()
    if args.reuse_captures is not None:
        banks = {lid: load_layer_capture(capture_dir, lid) for lid in resolved["layers"]}
        if not component_specs:
            # Recover recorded corpus components from capture provenance when reusing captures.
            first_prov = banks[resolved["layers"][0]].provenance
            for c in first_prov.get("components", ()):
                component_specs.append(CorpusComponentSpec(**c))
    else:
        banks = _stream_hy4_flash_captures(
            reader,
            adapter,
            resolved["layers"],
            all_records,
            train_count=resolved["train_sequences"],
            val_count=resolved["val_sequences"],
            seed=args.seed,
            capture_dir=capture_dir,
            provenance={"components": [c.to_dict() for c in component_specs]},
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
    fit_cfg = FitConfig(steps=resolved["steps"], seed=args.seed)
    template_path = checkpoint / "chat_template.jinja"
    template_sha = hashlib.sha256(template_path.read_bytes()).hexdigest() if template_path.is_file() else ""

    result = calibrate_hy4_flash_from_banks(
        banks,
        output_dir=args.output,
        source=source,
        modes=resolved["modes"],
        corpus_components=component_specs,
        fit_config=fit_cfg,
        device=args.device,
        renderer_template_sha256=template_sha,
    )
    print(
        json.dumps({
            "status": result["validation"]["status"],
            "artifact_status": result["manifest"]["artifact_status"],
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
