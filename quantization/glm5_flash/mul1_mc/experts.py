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
                workspace_bytes, mem_available=None, container_remaining=None, input_dtype=torch.float32):
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
    if input_dtype not in (torch.float32, torch.bfloat16):
        raise ValueError("expert input bank must use FP32 or native BF16")
    bank = tokens * width * (2 if input_dtype == torch.bfloat16 else 4)
    # For two width-8 route sets, a conservative peak including input ids,
    # sort order, row ids, ranks, and both sorted and original coefficients.
    route_bytes = tokens * 16 * (4 + 8 + 8 + 8 + 4 + 4)
    owners = dict(input_bank=bank, output_sum=tokens * width * 4, routes=route_bytes, frontier=frontier_bytes,
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
    def __init__(self, batches, *, tokens, width=4096, experts=288, device="cuda:0", input_dtype=torch.float32):
        if tokens <= 0 or width <= 0 or experts < 16:
            raise ValueError("invalid GLM expert bank dimensions")
        if input_dtype not in (torch.bfloat16, torch.float32):
            raise ValueError("expert input bank must use FP32 or native BF16")
        self.inputs = torch.empty((tokens, width), dtype=input_dtype, device=device)
        ids = torch.empty((tokens, 8), dtype=torch.int32, device=device)
        near = torch.empty_like(ids)
        gamma = torch.empty_like(ids, dtype=torch.float32)
        near_gamma = torch.empty_like(gamma)
        offset = 0
        for hidden, native, scales, candidates, recovery_scales in batches:
            if input_dtype == torch.bfloat16 and hidden.dtype != torch.bfloat16:
                raise ValueError("native BF16 expert bank cannot narrow higher precision activations")
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

    def chunks(self, rows, gather_buffer, *, native_buffer=None):
        """Gather into the caller's reusable device workspace, in selection order."""
        if (gather_buffer.ndim != 2 or not len(gather_buffer)
                or gather_buffer.shape[1] != self.inputs.shape[1] or gather_buffer.dtype != torch.float32
                or gather_buffer.device != self.inputs.device):
            raise ValueError("invalid expert gather workspace")
        gather = gather_buffer
        if self.inputs.dtype == torch.bfloat16:
            if (native_buffer is None or native_buffer.shape != gather_buffer.shape
                    or native_buffer.dtype != torch.bfloat16 or native_buffer.device != self.inputs.device):
                raise ValueError("native BF16 bank requires a reusable BF16 device gather workspace")
            gather = native_buffer
        for indices in rows.split(len(gather_buffer)):
            if len(indices):
                source = gather[:len(indices)]
                torch.index_select(self.inputs, 0, indices, out=source)
                chunk = gather_buffer[:len(indices)]
                if gather is not gather_buffer:
                    chunk.copy_(source)
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
    def combine(self, expert_outputs, *, weighted_buffer=None):
        """Call expert_outputs(expert, natural_rows) for ordered unweighted chunks."""
        result = torch.zeros_like(self.inputs, dtype=torch.float32)
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
                if weighted_buffer is None or len(weighted_buffer) < len(output):
                    weighted_buffer = torch.empty(output.shape, device=result.device, dtype=torch.float32)
                weighted = weighted_buffer[:len(output)]
                if weighted.shape != output.shape or weighted.dtype != result.dtype or weighted.device != result.device:
                    raise ValueError("incorrect weighted reduction workspace")
                torch.mul(output, gamma[offset:end, None], out=weighted)
                result.index_add_(0, rows[offset:end], weighted)
                offset = end
            if offset != len(rows):
                raise ValueError("incomplete routed expert output")
        for part in result.split(2048):
            if not bool(torch.isfinite(part).all()):
                raise ValueError("nonfinite routed expert sum")
        return result


@dataclass
class ExpertWorkspace:
    """The same native GEMM buffers serve fitting, replay and held-out batches."""
    gather: torch.Tensor
    inputs: torch.Tensor
    fused: torch.Tensor
    activated: torch.Tensor
    scaled: torch.Tensor
    outputs: torch.Tensor

    @classmethod
    def allocate(cls, width, intermediate, *, device, chunk_rows=2048):
        if min(width, intermediate, chunk_rows) <= 0:
            raise ValueError("invalid expert workspace geometry")
        def bank(features, dtype):
            return torch.empty((chunk_rows, features), dtype=dtype, device=device)
        return cls(bank(width, torch.float32), bank(width, torch.bfloat16),
                   bank(intermediate * 2, torch.bfloat16), bank(intermediate, torch.bfloat16),
                   bank(intermediate, torch.float32), bank(width, torch.bfloat16))

    def activate(self, gathered, gate_up, limit):
        import torch.nn.functional as F
        count = len(gathered)
        inputs, fused = self.inputs[:count], self.fused[:count]
        inputs.copy_(gathered)
        torch.mm(inputs, gate_up.T, out=fused)
        gate, up = fused.chunk(2, dim=-1)
        # Native order: clamp before SiLU, then multiply in BF16.
        F.silu(gate.clamp_(max=limit), inplace=True)
        torch.mul(gate, up.clamp_(min=-limit, max=limit), out=self.activated[:count])
        return self.activated[:count]


_PARTS = ("gate_proj", "up_proj", "down_proj")


def _reconstruct(packed, part, shape, device):
    from .numerics import reconstruct
    return reconstruct(packed[part], list(shape), device)


def _gate_up(packed, workspace):
    shape = workspace.activated.shape[-1], workspace.inputs.shape[-1]
    return torch.cat([_reconstruct(packed, part, shape, workspace.inputs.device)
                      for part in _PARTS[:2]], dim=0)


@torch.inference_mode()
def fit_expert(bank, expert, weights, *, prefix, workspace, swiglu_limit):
    """Encode one K4 expert from its selected observations and return CPU packed weights.

    Weights and return keys are gate_proj/up_proj/down_proj. The caller seals
    packed buffers and evidence in groups. Recovery replay only needs packed.
    """
    from .numerics import covariance_accumulator, encode_expert, raw_hessian
    intermediate, width = workspace.activated.size(1), workspace.inputs.size(1)
    shapes = ((intermediate, width), (intermediate, width), (width, intermediate))
    if set(weights) != set(_PARTS) or any(
            weights[key].shape != shape or weights[key].dtype != torch.bfloat16
            for key, shape in zip(_PARTS, shapes, strict=True)):
        raise ValueError("GLM expert source weight contract mismatch")
    rows, gamma, _, identity = bank.selection(expert)
    device = bank.inputs.device
    packed, metrics = {}, {}

    def encode(part, covariance):
        weight = weights[part].to(device)
        encoded = encode_expert(f"{prefix}.{part}", weight, covariance, device)
        packed[part] = encoded.packed[4]
        metrics[part] = encoded.metrics[4]
        return encoded.reconstruction

    acc = covariance_accumulator(width, device=device)
    for chunk in bank.chunks(rows, workspace.gather, native_buffer=workspace.inputs):
        acc.add(chunk)
    covariance = raw_hessian(acc, identity_rows=identity)
    del acc
    fused = torch.cat([encode(part, covariance) for part in _PARTS[:2]], dim=0)
    del covariance
    acc = covariance_accumulator(intermediate, device=device)
    offset = 0
    for chunk in bank.chunks(rows, workspace.gather, native_buffer=workspace.inputs):
        count = len(chunk)
        activation = workspace.activate(chunk, fused, swiglu_limit)
        # The linear down output is route weighted. Apply the identical weight
        # to its Hessian observations, including hypothetical candidate routes.
        torch.mul(activation, gamma[offset:offset + count, None], out=workspace.scaled[:count])
        acc.add(workspace.scaled[:count])
        offset += count
    encode("down_proj", raw_hessian(acc, identity_rows=identity))
    return packed, metrics


def replay_expert(bank, expert, packed, *, workspace, swiglu_limit):
    """Reconstruct the serialized expert for the ordered natural-route sum."""
    part = bank.native.part(expert)
    rows = bank.native.rows[part]
    if not len(rows):
        return
    fused = _gate_up(packed, workspace)
    shape = (workspace.inputs.size(1), workspace.activated.size(1))
    down = _reconstruct(packed, "down_proj", shape, bank.inputs.device)
    for chunk in bank.chunks(rows, workspace.gather, native_buffer=workspace.inputs):
        activation = workspace.activate(chunk, fused, swiglu_limit)
        out = workspace.outputs[:len(chunk)]
        torch.mm(activation, down.T, out=out)
        yield out
