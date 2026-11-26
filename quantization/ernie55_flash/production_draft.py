"""Calibrate dSpark's feature projection on the three causal target taps."""
import math

import torch
import torch.nn.functional as F

from kimi_k3.uniform_k2.storage import read_header, verify_receipt
from .forward import reference


FEATURE = "feature_projection"
INITIAL = "mtp-input-000"


def field_tail(stages, stage, index, key, shape, rows, device):
    """Read only the final needed rows of an immutable segmented FP32 activation record."""
    path, _, offset, length = stages.reference(stage, index, key)
    verify_receipt(path, stages._manifests[stage]["entries"][index])
    header, base = read_header(path, offset=offset, length=length)
    spec = header[key]
    if spec["dtype"] != "F32" or spec["shape"] != list(shape) or not 0 < rows <= shape[0]:
        raise ValueError(f"{stage} prompt {index}: unexpected {key} activation shape or dtype")
    count = rows * math.prod(shape[1:]) * 4
    payload = bytearray(count)
    with path.open("rb") as stream:
        stream.seek(base + spec["data_offsets"][1] - count)
        if stream.readinto(payload) != count:
            raise ValueError(f"{stage} prompt {index}: truncated {key} tail")
    return torch.frombuffer(payload, dtype=torch.float32).reshape(rows, *shape[1:]).to(device)


@torch.inference_mode()
def prepare_draft(model, stages, counts, tokens_per_prompt, tap_stages, last_frontier, event):
    from .production_core import LayerCore

    c = model.config
    total = counts["calibration"] + counts["heldout"]
    if (len(tokens_per_prompt) != total or min(tokens_per_prompt) < 1
            or len(tap_stages) != 3 or len(set(tap_stages)) != 3
            or len(c.dspark_target_layer_ids) != len(tap_stages)):
        raise ValueError("incomplete causal target frontiers for ERNIE dSpark")
    device = model.checkpoint.device
    weights = {name: model.checkpoint.load("model.mtp." + name)
               for name in ("feature_projection.weight", "feature_norm.weight")}
    worker = LayerCore(model, weights, stages, "mtp-feature", "model.mtp.", counts, event)

    def taps(i, limit=None):
        length = tokens_per_prompt[i]
        rows = min(length, limit) if limit else length
        return torch.cat([field_tail(stages, stage, i, "tap", (length, c.hidden_size), rows, device)
                          for stage in tap_stages], dim=1)

    def feature_inputs(start):
        for i in range(start, total):
            yield i, (taps(i),)

    worker.fit("projection", [(FEATURE,)], feature_inputs)
    if not all(row["heldout_passed"] for row in worker.metrics):
        raise ValueError("dSpark feature projection missed its held-out NMSE bound")

    def initial(start):
        outer = {name: model.checkpoint.load(name) for name in reference.OUTER_NAMES}
        embed = outer[reference.OUTER_NAMES[0]]
        # Head collapse, multimax and greedy choice are positionwise. Match the
        # reference's fixed GEMM row tile while amortizing the large native head.
        for batch in range(start, total, reference.ROWS):
            indices = range(batch, min(batch + reference.ROWS, total))
            last = torch.cat([field_tail(stages, last_frontier, i, "state",
                                         (tokens_per_prompt[i], c.hc_mult, c.hidden_size), 1, device)
                              for i in indices])
            logits = model.logits(last, outer)
            valid = torch.isfinite(logits).all()
            anchors = logits.argmax(dim=-1)
            del logits
            records = []
            for row, i in enumerate(indices):
                length = tokens_per_prompt[i]
                ctx = model.draft_features(taps(i, c.sliding_window), weights)
                valid = valid & torch.isfinite(ctx).all()
                ids = torch.full((c.dspark_block_size,), c.dspark_noise_token_id,
                                 dtype=torch.int64, device=device)
                ids[0] = anchors[row]
                state = F.embedding(ids, embed)[:, None].expand(-1, c.hc_mult, -1).contiguous()
                records.append((i, dict(state=state, context=ctx, anchor=anchors[row],
                                        context_positions=torch.arange(length - len(ctx), length, device=device),
                                        positions=torch.arange(length, length + len(ids), device=device))))
            if not valid.item():
                raise ValueError(f"nonfinite target logits or dSpark context in batch starting at {batch}")
            yield from records

    event("draft_frontier", stage=INITIAL)
    stages.write_activations(INITIAL, initial, total)
    return worker.packed[8], worker.packed[6], worker.metrics, INITIAL
