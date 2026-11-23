"""Single-capture core and expert encodes for GLM Flash K4 MUL1."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from inkling2 import numerical as fork

ERROR_BARS = {8: 1e-4, 6: 1e-3, 4: 1e-2}
MIN_EXPERT_ROWS = 1024


def covariance_accumulator(width, device):
    """FP32 128-row products, FP64 outer sum and finite FP32 snapshot."""
    return fork.fork_module("utils.exl3_hessian").BoundedHessianAccumulator(
        width, device=device, fail_stop=True)


@torch.inference_mode()
def raw_hessian(acc, identity_rows=0):
    if type(identity_rows) is not int or identity_rows < 0 or acc.count + identity_rows == 0:
        raise ValueError("expected a positive effective Hessian row count")
    raw = acc.snapshot()
    if identity_rows:
        raw.diagonal().add_(identity_rows)
    return dict(H=raw, count=acc.count + identity_rows, identity_rows=identity_rows, finalized=False)


def reconstruct(packed, shape, device):
    restored = fork.reconstruct_projection(packed, list(shape), device=str(device),
                                          dtype=torch.bfloat16, codebook="mul1")
    # A large vocabulary head can exceed 600M entries. Bound the finite-test buffer.
    valid = torch.ones((), dtype=torch.bool, device=restored.device)
    for chunk in restored.split(max(1, (8 << 20) // restored.shape[1])):
        valid.logical_and_(torch.isfinite(chunk).all())
    torch._assert_async(valid, "nonfinite serialized MUL1 reconstruction")
    return restored


@dataclass
class ProjectionEncoding:
    packed: dict[int, dict[str, torch.Tensor]]
    metrics: dict[int, dict]
    reconstruction: torch.Tensor


@torch.inference_mode()
def _encode(name, weight, raw, device, widths, evaluate):
    packed_by_width, measurements = {}, {}
    causal = None
    for width in widths:
        packed, metrics, evidence = fork.quantize_projection(
            weight, raw, bits=width, seed=787, sigma=.025, device=str(device), codebook="mul1")
        error = metrics["hessian_weighted_relative_error"]
        if not math.isfinite(error) or error < 0 or (width == 6 and error > ERROR_BARS[6]):
            raise ValueError(f"{name} K{width} training error outside contract: {error}")

        size = sum(value.numel() * value.element_size() for value in packed.values())
        record = dict(evidence | metrics, module=name, bits=width, side=1,
                      codebook="mul1", multiplier=0x83DCD12D,
                      effective_bits_per_weight=8 * size / weight.numel())
        if width == widths[0] or evaluate is not None:
            restored = reconstruct(packed, weight.shape, device)
            if width == widths[0]:
                causal = restored
            if evaluate is not None:
                heldout = float(evaluate(width, restored))
                if not math.isfinite(heldout) or not 0 <= heldout <= ERROR_BARS[width]:
                    raise ValueError(f"{name} K{width} heldout NMSE {heldout} exceeds {ERROR_BARS[width]}")
                record.update(heldout_nmse=heldout, heldout_bar=ERROR_BARS[width])
            del restored
        packed_by_width[width], measurements[width] = packed, record

    return ProjectionEncoding(packed_by_width, measurements, causal)


def encode_core(name, weight, raw, device, *, evaluate=None):
    if raw.get("identity_rows", 0) != 0:
        raise ValueError("core projections cannot use expert recovery rows")
    return _encode(name, weight, raw, device, (8, 6), evaluate)


def encode_expert(name, weight, raw, device, *, evaluate=None):
    if raw["count"] < MIN_EXPERT_ROWS:
        raise ValueError(f"routed expert Hessian needs at least {MIN_EXPERT_ROWS} effective rows")
    return _encode(name, weight, raw, device, (4,), evaluate)
