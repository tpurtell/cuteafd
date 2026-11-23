"""GB10 expert inputs, calibration selections and ordered FP32 reduction."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import xxhash

from hy4_flash.storage import (
    CANDIDATE_GAMMA_CONTRACT, IDENTITY_CONTRACT, MINIMUM_EXPERT_ROWS, candidate_routes,
)


@torch.inference_mode()
def route_batch(moe, hidden):
    """Take the native unsorted top eight, then rank candidates with the same bias."""
    config = moe.config
    if (config.n_group, config.topk_group, config.num_experts_per_tok,
            config.norm_topk_prob, config.routed_scaling_factor) != (1, 1, 8, True, 2.5):
        raise ValueError("unexpected GLM Flash router contract")
    logits, gamma, ids = moe.gate(hidden)
    scores = logits.sigmoid()
    choice = scores + moe.gate.e_score_correction_bias
    near, near_gamma = candidate_routes(scores, choice, ids, config)
    return ids, gamma, near, near_gamma


def memory_plan(tokens, width, *, frontier_bytes, weight_bytes, hessian_bytes,
                workspace_bytes, mem_available=None, container_remaining=None):
    """Admit against unified RAM headroom before allocating the routed layer.

    Frontier, weights, Hessians and workspace are peak additional resident bytes
    supplied by the caller from its active phase. Workspace includes sort scratch,
    GEMMs, gather chunks and the trellis workspace measured by the pilot.
    """
    if mem_available is None:
        value = next(s for s in Path("/proc/meminfo").read_text().splitlines()
                     if s.startswith("MemAvailable:"))
        mem_available = int(value.split()[1]) * 1024
    if min(tokens, width) <= 0 or min(frontier_bytes, weight_bytes, hessian_bytes, workspace_bytes) < 0:
        raise ValueError("invalid expert working set")
    bank = tokens * width * 4
    # For two width-8 route sets, a conservative peak including input ids,
    # sort order, row ids, ranks, and both sorted and original coefficients.
    route_bytes = tokens * 16 * (4 + 8 + 8 + 8 + 4 + 4)
    owners = dict(input_bank=bank, output_sum=bank, routes=route_bytes, frontier=frontier_bytes,
                  weights=weight_bytes, hessians=hessian_bytes, workspace=workspace_bytes)
    available = mem_available - (10 << 30)
    if container_remaining is not None:
        available = min(available, container_remaining)
    return dict(owners=owners, required=sum(owners.values()), available=available,
                mem_available=mem_available, admitted=sum(owners.values()) <= available)


@dataclass
class _Grouped:
    rows: torch.Tensor
    gamma: torch.Tensor
    bounds: list[int]
    ranks: torch.Tensor | None = None

    def part(self, i):
        return slice(self.bounds[i], self.bounds[i + 1])


def _group(ids, weights, experts, *, deficits=None):
    # Stable grouping preserves natural corpus order. Transposing candidates
    # first gives rank-major, corpus-order ties within each expert.
    candidate = deficits is not None
    width = ids.size(1)
    flat = (ids.T if candidate else ids).reshape(-1)
    order = flat.argsort(stable=True)
    counts = torch.zeros(experts, dtype=torch.int64, device=ids.device)
    counts.scatter_add_(0, flat.long(), torch.ones_like(flat, dtype=torch.int64))
    if candidate:
        starts = counts.cumsum(0) - counts
        owners = flat[order].long()
        offsets = torch.arange(len(order), device=ids.device) - starts[owners]
        order = order[offsets < deficits[owners]]
        counts = torch.minimum(counts, deficits)
    rows = order % len(ids) if candidate else order // width
    ranks = order // len(ids) + width + 1 if candidate else None
    gamma = (weights.T if candidate else weights).reshape(-1)[order]
    return _Grouped(rows, gamma, [0, *counts.cumsum(0).tolist()], ranks), counts


class ExpertBank:
    """One device input bank. The same selections drive calibration and replay."""

    @torch.inference_mode()
    def __init__(self, batches, *, tokens, width=4096, experts=288, device="cuda:0"):
        if tokens <= 0 or width <= 0 or experts < 16:
            raise ValueError("invalid GLM expert bank dimensions")
        self.inputs = torch.empty((tokens, width), dtype=torch.float32, device=device)
        ids = torch.empty((tokens, 8), dtype=torch.int32, device=device)
        near = torch.empty_like(ids)
        gamma = torch.empty_like(ids, dtype=torch.float32)
        near_gamma = torch.empty_like(gamma)
        offset = 0
        for hidden, native, scales, candidates, recovery_scales in batches:
            n = len(hidden)
            if (not n or offset + n > tokens or hidden.shape != (n, width)
                    or hidden.dtype not in (torch.bfloat16, torch.float32)
                    or any(t.shape != (n, 8) for t in (native, scales, candidates, recovery_scales))
                    or any(t.dtype not in (torch.int64, torch.int32) for t in (native, candidates))
                    or any(t.dtype != torch.float32 for t in (scales, recovery_scales))
                    or any(t.device != self.inputs.device for t in
                           (hidden, native, scales, candidates, recovery_scales))):
                raise ValueError("incorrect expert batch geometry, precision or device")
            sorted_ids = torch.cat((native, candidates), dim=1).sort(dim=1).values
            valid = ((sorted_ids[:, 0] >= 0).all() & (sorted_ids[:, -1] < experts).all()
                     & (sorted_ids[:, :-1] != sorted_ids[:, 1:]).all()
                     & torch.isfinite(hidden).all() & torch.isfinite(scales).all()
                     & torch.isfinite(recovery_scales).all() & (scales >= 0).all()
                     & (recovery_scales >= 0).all())
            if not bool(valid):
                raise ValueError("invalid expert input, route IDs or coefficients")
            part = slice(offset, offset + n)
            self.inputs[part].copy_(hidden)
            ids[part].copy_(native)
            gamma[part].copy_(scales)
            near[part].copy_(candidates)
            near_gamma[part].copy_(recovery_scales)
            offset += n
        if offset != tokens:
            raise ValueError("incomplete routed layer")
        self.native, counts = _group(ids, gamma, experts)
        del ids, gamma
        self.recovery, _ = _group(near, near_gamma, experts,
                                  deficits=(MINIMUM_EXPERT_ROWS - counts).clamp(min=0))
        self.experts = experts

    def selection(self, expert):
        if not 0 <= expert < self.experts:
            raise IndexError("expert outside layer")
        n, r = self.native.part(expert), self.recovery.part(expert)
        rows, gamma = self.native.rows[n], self.native.gamma[n]
        natural = len(rows)
        if r.stop != r.start:
            rows = torch.cat((rows, self.recovery.rows[r]))
            gamma = torch.cat((gamma, self.recovery.gamma[r]))
        return rows, gamma, natural, max(0, MINIMUM_EXPERT_ROWS - len(rows))

    def chunks(self, rows, gather_buffer):
        """Gather into the caller's reusable device workspace, in selection order."""
        if (gather_buffer.ndim != 2 or not len(gather_buffer)
                or gather_buffer.shape[1] != self.inputs.shape[1]
                or gather_buffer.dtype != self.inputs.dtype or gather_buffer.device != self.inputs.device):
            raise ValueError("invalid expert gather workspace")
        for indices in rows.split(len(gather_buffer)):
            if len(indices):
                chunk = gather_buffer[:len(indices)]
                torch.index_select(self.inputs, 0, indices, out=chunk)
                yield chunk

    def receipts(self):
        """Transfer selection metadata once per layer; scratch and hashes stay small."""
        n, r = self.native, self.recovery
        natural, scales, near, near_scales, ranks = [
            tensor.cpu().numpy() for tensor in (n.rows, n.gamma, r.rows, r.gamma, r.ranks)]
        for expert in range(self.experts):
            first, extra = n.part(expert), r.part(expert)
            fields = natural[first], scales[first], near[extra], near_scales[extra], ranks[extra]
            digest = xxhash.xxh3_128()
            for field in fields:
                digest.update(len(field).to_bytes(8, "little"))
                digest.update(memoryview(field).cast("B"))
            count = len(fields[0]) + len(fields[2])
            identity = max(0, MINIMUM_EXPERT_ROWS - count)
            yield dict(expert=expert, natural_rows=len(fields[0]), augmented_rows=len(fields[2]),
                       identity_rows=identity, effective_rows=count + identity,
                       candidate_rank_first=9, augmented_rank_counts=[
                           int((fields[4] == rank).sum()) for rank in range(9, 17)],
                       selection_xxh3_128=digest.hexdigest(), selection_policy="rank-major-then-corpus-row",
                       candidate_gamma_contract=CANDIDATE_GAMMA_CONTRACT, identity_contract=IDENTITY_CONTRACT)

    @torch.inference_mode()
    def combine(self, expert_outputs):
        """Call expert_outputs(expert, natural_rows) for ordered unweighted chunks."""
        result = torch.zeros_like(self.inputs)
        for expert in range(self.experts):
            part = self.native.part(expert)
            rows, gamma = self.native.rows[part], self.native.gamma[part]
            offset = 0
            for output in expert_outputs(expert, rows):
                end = offset + len(output)
                if (not len(output) or end > len(rows) or output.ndim != 2
                        or output.shape[1] != result.shape[1] or output.device != result.device
                        or output.dtype not in (torch.bfloat16, torch.float32)):
                    raise ValueError("invalid routed expert output chunk")
                weighted = output.float() * gamma[offset:end, None]
                result.index_add_(0, rows[offset:end], weighted)
                offset = end
            if offset != len(rows):
                raise ValueError("incomplete routed expert output")
        return result


def raw_covariance(accumulator, identity_rows):
    """Apply the low-row fallback to raw XTX before any count normalization."""
    if identity_rows < 0 or accumulator.count + identity_rows <= 0:
        raise ValueError("invalid expert Hessian count")
    matrix = accumulator.snapshot()
    matrix.diagonal().add_(identity_rows)
    return dict(H=matrix, count=accumulator.count + identity_rows,
                finalized=False, identity_rows=identity_rows)
