"""Segmented activation checkpoints and ordered replay of sealed ERNIE experts."""
from __future__ import annotations

import torch

from hy4_flash.storage import HostExpertStaging, Stages


class RoutedWorkspace:
    """Keep latent inputs once on the host and the complete routed sum on the device.

    The caller's expert loader decodes one sealed K4 expert at a time. Its
    callable consumes staged FP32 rows and returns unweighted FP32 outputs.
    Replaying this method reconstructs the sum without an expert-output spool.
    The returned sum is borrowed until the next replay on this workspace.
    """

    def __init__(self, routes, device, *, chunk_rows=2048, stage_bytes=8 << 30):
        latent = routes.latent
        if (latent.device.type != "cpu" or latent.dtype != torch.float32
                or latent.ndim != 2 or not latent.numel() or chunk_rows <= 0):
            raise ValueError("expected a nonempty host FP32 latent bank and positive chunk size")
        self.routes = routes
        self.staging = HostExpertStaging(latent, device, chunk_rows, stage_bytes)
        self.sum = torch.empty_like(latent, device=device)
        self.rows = torch.empty(chunk_rows, dtype=torch.long, device=device)
        self.gamma = torch.empty((chunk_rows, 1), dtype=torch.float32, device=device)
        self.weighted = torch.empty((chunk_rows, latent.shape[1]), dtype=torch.float32, device=device)

    def calibration_inputs(self, expert):
        rows, gamma, evidence = self.routes.calibration(expert)
        return self.staging.select(rows), gamma, evidence

    @torch.no_grad()
    def replay(self, load_expert):
        self.sum.zero_()
        for expert in range(len(self.routes.bounds) - 1):
            rows, gamma = self.routes.expert(expert)
            if not len(rows):
                continue
            project = load_expert(expert)
            offset = 0
            for inputs in self.staging.select(rows)(len(rows)):
                n = len(inputs)
                idx = self.rows[:n].copy_(rows[offset:offset+n])
                weight = self.gamma[:n].copy_(gamma[offset:offset+n, None])
                output = project(inputs)
                if output.shape != inputs.shape or output.dtype != inputs.dtype or output.device != inputs.device:
                    raise ValueError("expert output differs from the staged latent geometry")
                torch.mul(output, weight, out=self.weighted[:n])
                # Each token occurs at most once within an expert. Separate calls
                # on one stream preserve ascending expert order for every token.
                self.sum.index_add_(0, idx, self.weighted[:n])
                offset += n
            del project, output
        return self.sum


def commit_wavefront(stages: Stages, name, batches, count):
    """Checkpoint only complete causal states, resuming at a sealed segment boundary."""
    def records(start):
        for index, state in batches(start):
            if not state.is_floating_point() or not torch.isfinite(state).all().item():
                raise ValueError("non-finite ERNIE wavefront")
            yield index, {"state": state}
    return stages.write_activations(name, records, count)
