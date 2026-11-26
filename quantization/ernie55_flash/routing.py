"""ERNIE native top-10 dispatch, rank 11–20 augmentation and cold-expert evidence."""
from __future__ import annotations

import torch

from hy4_flash.storage import Routes


TOP_K = 10
CANDIDATE_GAMMA_CONTRACT = "ernie55-last-top10-substitution-unbiased-sum-sigmoid-learned-scale-v1"


def rank_routes(scores, bias, scales):
    """Match the reference's unsorted native slots, then rank the remaining choices."""
    if (scores.ndim != 2 or scores.shape[1] < 2 * TOP_K or bias.shape != scores.shape[1:]
            or scales.shape != bias.shape or any(t.dtype != torch.float32 for t in (scores, bias, scales))):
        raise ValueError("invalid ERNIE two-router score, correction bias or scale geometry")
    choice = scores + bias
    ids = choice.topk(TOP_K, dim=-1, sorted=False).indices
    probs = scores.gather(1, ids)
    gamma = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-20) * scales[ids]

    # Keep the native set even at a tie across rank 10. The reference uses
    # topk(sorted=False), so its final *slot* need not be its lowest rank.
    order = choice.gather(1, ids).argsort(dim=1, descending=True, stable=True)
    keep = probs.gather(1, order[:, :-1])
    remaining = choice.scatter_(1, ids, -torch.inf)
    near = remaining.topk(TOP_K, dim=-1, sorted=True).indices
    alternatives = scores.gather(1, near)
    hypothetical = torch.cat((keep[:, None].expand(-1, TOP_K, -1), alternatives[:, :, None]), dim=-1)
    near_gamma = alternatives / hypothetical.sum(dim=-1).clamp_min(1e-20) * scales[near]
    return dict(ids=ids, gamma=gamma, candidate_ids=near.to(torch.int32), candidate_gamma=near_gamma)


class ErnieRoutes(Routes):
    """Host input bank and corpus-ordered selections for target or dSpark latent experts."""

    candidate_gamma_contract = CANDIDATE_GAMMA_CONTRACT
    signed_scaling_factors = True

    def __init__(self, batches, experts, tokens):
        def checked():
            expected_index = 0
            used = 0
            for index, batch in batches:
                latent = batch["latent"]
                if (index != expected_index or latent.ndim != 2 or len(latent) == 0
                        or any(t.device.type != "cpu" for t in batch.values())
                        or batch["ids"].shape != (len(latent), TOP_K)
                        or batch["candidate_ids"].shape != (len(latent), TOP_K)
                        or used + len(latent) > tokens or not torch.isfinite(latent).all()):
                    raise ValueError("invalid ERNIE host route record or corpus order")
                yield index, batch
                expected_index += 1
                used += len(latent)

        super().__init__(checked(), experts, tokens, device="cpu", recover=True)


def recovered_covariance(accumulator, evidence):
    """Apply the identity deficit to the raw outer sum before GPTQ normalizes."""
    rows = evidence["natural_rows"] + evidence["augmented_rows"]
    missing = evidence["identity_rows"]
    effective = evidence["effective_rows"]
    if (accumulator.count != rows or missing != max(0, 1024 - rows) or effective != rows + missing):
        raise ValueError("ERNIE expert observation count differs from route selection")
    matrix = accumulator.snapshot()
    if missing:
        matrix.diagonal().add_(missing)
    return dict(H=matrix, count=effective, finalized=False, identity_rows=missing)
