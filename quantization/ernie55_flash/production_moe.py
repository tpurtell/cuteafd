"""Production K4 routed experts for target and dSpark layers on maxq."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import sys
import time

import torch

from hy4_flash.storage import Routes
from inkling2 import numerical
from .forward import reference
from .routing import ErnieRoutes, recovered_covariance
from .storage import RoutedWorkspace


SUFFIXES = ("trellis", "suh", "svh", "mul1")
EXPERT_GROUP = 32


@dataclass
class RoutedResult:
    sums: dict
    packed: dict = field(default_factory=dict)
    metrics: list = field(default_factory=list)
    stages: list = field(default_factory=list)


def route_records(model, weights, inputs, calibration):
    for expected, (index, x) in enumerate(inputs):
        if index != expected:
            raise ValueError("unordered ERNIE expert input stream")
        batch = model.route_batch(x.to(model.checkpoint.device), weights)
        if calibration:
            yield index, {key: value.cpu() for key, value in batch.items()}
        else:
            # A small disjoint held-out bank fits on the device. Only its route
            # selections go to the host for the stable expert-major ordering.
            yield index, dict(latent=batch["latent"], ids=batch["ids"].cpu(), gamma=batch["gamma"].cpu())


def decode(values, name, shape, device):
    packed = {key: values[name + "." + key] for key in SUFFIXES}
    return numerical.reconstruct_projection(packed, list(shape), device=device,
                                            dtype=torch.bfloat16, codebook="mul1").float()


def fit(name, weight, covariance, evidence, values, metrics):
    packed, measurements, spec = numerical.quantize_projection(weight, covariance, bits=4,
        seed=787, sigma=.025, device=weight.device, codebook="mul1")
    if set(packed) != set(SUFFIXES):
        raise ValueError(f"incomplete ERNIE K4 projection: {name}")
    values.update({name + "." + key: value for key, value in packed.items()})
    nbytes = sum(t.numel() * t.element_size() for t in packed.values())
    metrics.append(dict(spec | measurements | evidence, module=name, bits=4,
                        effective_bits_per_weight=8 * nbytes / weight.numel()))
    return decode(values, name, weight.shape, weight.device)


class ExpertPass:
    """One reusable GPU row bank, ordered FP32 sums, and bounded witness storage per layer."""
    def __init__(self, model, weights, calibration, heldout, chunk_rows, stage_bytes):
        self.model = model
        self.store = weights["experts"]
        self.device = model.checkpoint.device
        self.calibration = calibration
        self.heldout = heldout
        self.ws = RoutedWorkspace(calibration, self.device, chunk_rows=chunk_rows, stage_bytes=stage_bytes)
        self.ws.sum.zero_()
        self.held_sum = torch.zeros_like(heldout.latent)
        self.finite = torch.ones((), dtype=torch.bool, device=self.device)
        self.energy = torch.zeros((3, 2), dtype=torch.float64, device=self.device)
        self.chunk_rows = min(chunk_rows, len(self.ws.staging.staged))
        self.probes = torch.arange(max(0, len(heldout.latent) - 128), len(heldout.latent))
        self.shapes = [(model.config.moe_intermediate_size, model.config.moe_latent_size)] * 2 + [
            (model.config.moe_latent_size, model.config.moe_intermediate_size)]

    def activated(self, x, restored):
        return self.model.activated(x, restored[0], restored[1])

    def add_sum(self, dest, rows, gamma, output):
        n = len(rows)
        indices = self.ws.rows[:n].copy_(rows)
        gains = self.ws.gamma[:n].copy_(gamma[:, None])
        scratch = self.ws.weighted[:n]
        torch.mul(output, gains, out=scratch)
        self.finite.logical_and_(torch.isfinite(scratch).all())
        dest.index_add_(0, indices, scratch)

    def add_calibration(self, rows, gamma, chunks, restored):
        offset = 0
        if not len(rows):
            return
        for x in chunks(len(rows)):
            stop = offset + len(x)
            out = reference.linear(self.activated(x, restored), restored[2])
            self.add_sum(self.ws.sum, rows[offset:stop], gamma[offset:stop], out)
            offset = stop
        if offset != len(rows):
            raise ValueError("incomplete ERNIE calibration replay")

    def observe(self, index, actual, expect):
        expected = expect.double()
        self.energy[index, 0].add_((actual.double() - expected).square().sum())
        self.energy[index, 1].add_(expected.square().sum())

    def add_heldout(self, expert, restored, native=None, metrics=None):
        rows, gamma = self.heldout.expert(expert)
        natural = len(rows)
        if not natural and native is None:
            return
        if not natural:
            rows = self.probes
        self.energy.zero_()
        for start in range(0, len(rows), self.chunk_rows):
            chunk = rows[start:start + self.chunk_rows]
            indices = self.ws.rows[:len(chunk)].copy_(chunk)
            x = self.ws.staging.staged[:len(chunk)]
            torch.index_select(self.heldout.latent, 0, indices, out=x)
            gate, up = (reference.linear(x, w) for w in restored[:2])
            c = self.model.config
            active = reference.situ(gate, up, c.activation_situ_beta, c.activation_situ_linear_beta)
            if natural:
                g = gamma[start:start + len(chunk)]
                self.add_sum(self.held_sum, chunk, g, reference.linear(active, restored[2]))
                if native is not None:
                    active.mul_(self.ws.gamma[:len(chunk)])
            if native is not None:
                for i, y in enumerate((gate, up)):
                    self.observe(i, y, reference.linear(x, native[i]))
                self.observe(2, reference.linear(active, restored[2]), reference.linear(active, native[2]))
        if native is not None:
            for metric, (numerator, denom) in zip(metrics, self.energy.tolist(), strict=True):
                if not (math.isfinite(numerator) and math.isfinite(denom)):
                    raise ValueError(f"nonfinite ERNIE held-out measurement for {metric['module']}")
                # A vanishing learned scale can leave a down projection with no
                # reference energy on actual routes. Keep that observation explicit.
                nmse = numerator / denom if denom > 0 else None
                metric.update(heldout_nmse=nmse, heldout_rows=len(rows), heldout_natural_rows=natural,
                              heldout_bar=.01, heldout_passed=nmse <= .01 if nmse is not None else None,
                              heldout_selection="natural" if natural else "heldout-tail-no-natural-route",
                              heldout_reference_energy=denom)

    def encode(self, expert, names):
        rows, gamma, evidence = self.calibration.calibration(expert)
        chunks = self.ws.staging.select(rows)
        weights = self.store.get(expert, self.device)
        acc = numerical.covariance_accumulator(weights[0].shape[1], self.device)
        if len(rows):
            for x in chunks(len(rows)):
                acc.add(x)
        covariance = recovered_covariance(acc, evidence)
        del acc
        values, metrics = {}, []
        restored = [fit(name, w, covariance, evidence, values, metrics) for name, w in zip(names[:2], weights[:2])]
        del covariance
        acc = numerical.covariance_accumulator(weights[2].shape[1], self.device)
        offset = 0
        if len(rows):
            for x in chunks(len(rows)):
                n = len(x)
                gain = self.ws.gamma[:n].copy_(gamma[offset:offset+n, None])
                acc.add(self.activated(x, restored).mul_(gain))
                offset += n
        covariance = recovered_covariance(acc, evidence)
        restored.append(fit(names[2], weights[2], covariance, evidence, values, metrics))
        del covariance, acc
        natural = evidence["natural_rows"]
        self.add_calibration(rows[:natural], gamma[:natural], chunks, restored)
        self.add_heldout(expert, restored, weights, metrics)
        return values, metrics

    def replay(self, expert, names, values):
        restored = [decode(values, name, shape, self.device) for name, shape in zip(names, self.shapes, strict=True)]
        rows, gamma = self.calibration.expert(expert)
        self.add_calibration(rows, gamma, self.ws.staging.select(rows), restored)
        self.add_heldout(expert, restored)


@torch.inference_mode()
def encode_routed(model, weights, prefix, stages, bank_inputs, tokens, *, event=None,
                  chunk_rows=2048, stage_bytes=8 << 30):
    """Fit every expert, then return per-split sums for latent_norm -> fc2 and direct MC references.

    Each split provides an ordered iterator of (batch number, normalized MLP x).
    fc1_latent_proj in weights must already contain the serialized K8 reconstruction.
    Sealed 32-expert groups use bounded XXH3 segments; a restart rebuilds sums by
    replaying packed K4, without loading source expert matrices or recalibrating.
    """
    count = model.config.n_routed_experts
    device = model.checkpoint.device
    train = ErnieRoutes(route_records(model, weights, bank_inputs["calibration"], True), count, tokens["calibration"])
    held = Routes(route_records(model, weights, bank_inputs["heldout"], False), count, tokens["heldout"], device=device)
    worker = ExpertPass(model, weights, train, held, chunk_rows, stage_bytes)
    result = RoutedResult(dict(calibration=worker.ws.sum, heldout=worker.held_sum))
    def names(expert):
        return [f"{prefix}block_sparse_moe.experts.{expert}.{op}" for op in ("w1", "w3", "w2")]

    for start in range(0, count, EXPERT_GROUP):
        stop = min(start + EXPERT_GROUP, count)
        stage = prefix.rstrip(".").replace(".", "-") + f"-experts-{start:03d}"
        tick = time.monotonic()
        manifest = stages.load(stage)
        replayed = manifest is not None
        if replayed:
            if len(manifest["entries"]) != stop - start:
                raise ValueError(f"incomplete ERNIE expert group: {stage}")
            for expert in range(start, stop):
                worker.replay(expert, names(expert), stages.read_batch(stage, expert - start))
        else:
            metrics = []
            def records():
                for expert in range(start, stop):
                    values, rows = worker.encode(expert, names(expert))
                    metrics.extend(rows)
                    yield expert - start, values
                if not worker.finite.item():
                    raise ValueError(f"nonfinite ERNIE weighted expert output in {stage}")
            manifest = stages.write_batches(stage, records(), {"metrics": metrics},
                                            segment_batches=EXPERT_GROUP, segment_bytes=256 << 20)
        if replayed and not worker.finite.item():
            raise ValueError(f"nonfinite ERNIE replay in {stage}")
        for expert in range(start, stop):
            for module in names(expert):
                for suffix in SUFFIXES:
                    key = module + "." + suffix
                    result.packed[key] = stages.reference(stage, expert - start, key)
        result.metrics.extend(manifest["metadata"]["metrics"])
        result.stages.append(stage)
        if event:
            stats = {}
            if torch.device(device).type == "cuda":
                mod = sys.modules.get("gptqmodel.exllamav3.modules.quant.exl3_lib.quantize")
                stats = dict(cuda_allocated_bytes=torch.cuda.memory_allocated(device),
                             cuda_reserved_bytes=torch.cuda.memory_reserved(device),
                             trellis_workspace_count=len(mod._workspace_buffers) if mod else 0)
            event("expert_group_replayed" if replayed else "expert_group_complete", group=stage,
                  expert_projections=3*stop, group_seconds=time.monotonic()-tick, **stats)
    for summed in result.sums.values():
        for chunk in summed.split(chunk_rows):
            worker.finite.logical_and_(torch.isfinite(chunk).all())
    if not worker.finite.item():
        raise ValueError(f"nonfinite ERNIE routed sum for {prefix}")
    return result
