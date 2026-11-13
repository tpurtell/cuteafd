"""Bounded, receipt-backed Inkling route capture and rolling AttnRes state."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from glm5.core_k6.storage import ProjectionStore, load_tensors
from kimi_k3.core_k6.storage import StageStore
from kimi_k3.uniform_k2.storage import (atomic_json, file_identity,
    fsync_directory, sha256_file, verify_receipt)

ROUTE_SCHEMA = "inkling2-route-spool-v1"
STATE_SCHEMA = "inkling2-attnres-state-v1"
CANDIDATE_GAMMA_CONTRACT = "native-last-top8-substitution-renormalized-v1"
ACCUMULATION_CONTRACT = "fp32-products-k128-fp64-sum-fp32-storage-v1"
_FILES = {"inputs": ("inputs.bf16", np.uint16),
          "routes": ("routes.i16", np.int16),
          "weights": ("gamma.f32", np.float32),
          "candidates": ("candidates.i16", np.int16),
          "candidate_weights": ("candidate-gamma.f32", np.float32)}


def _geometry(tokens, width, experts, minimum):
    if (any(type(n) is not int for n in (tokens, width, experts, minimum))
            or tokens <= 0 or width <= 0 or not 16 <= experts <= 32767 or minimum < 0):
        raise ValueError("invalid Inkling route spool geometry")


def _routes_valid(ids, gamma, near, near_gamma, experts):
    if (ids.ndim != 2 or ids.shape[1] != 8 or near.shape != ids.shape
            or gamma.shape != ids.shape or near_gamma.shape != ids.shape
            or ids.dtype.kind not in "iu" or near.dtype.kind not in "iu"
            or np.any(ids < 0) or np.any(ids >= experts)
            or np.any(near < 0) or np.any(near >= experts)
            or np.any(np.diff(np.sort(np.concatenate((ids, near), axis=1), axis=1), axis=1) == 0)
            or not np.isfinite(gamma).all() or not np.isfinite(near_gamma).all()
            or np.any(gamma < 0) or np.any(near_gamma < 0)):
        raise ValueError("invalid native top8/rank9..16 route IDs or gamma")


class RouteSpool:
    """One layer of BF16 MLP inputs and native/hypothetical routing weights.

    Natural routes retain all their rows in corpus order. A cold expert adds
    rank-9..16 rows in rank-major/corpus order until ``minimum`` observations,
    then declares an identity contribution for any remaining shortfall.
    """

    def __init__(self, root, identity, tokens, width=4608, experts=384, minimum=1024,
                 *, layer=None, candidate_gamma_contract=CANDIDATE_GAMMA_CONTRACT):
        _geometry(tokens, width, experts, minimum)
        self.root, self.identity = Path(root), identity
        self.tokens, self.width, self.experts, self.minimum = tokens, width, experts, minimum
        if self.root.is_symlink():
            raise ValueError("refuse symlink route spool")
        manifest = json.loads((self.root / "manifest.json").read_text())
        expected = dict(schema=ROUTE_SCHEMA, identity=identity, tokens=tokens, width=width,
                        experts=experts, minimum=minimum, candidate_gamma_contract=candidate_gamma_contract)
        if any(manifest.get(k) != v for k, v in expected.items()) or (layer is not None and manifest.get("layer") != layer):
            raise ValueError("route spool identity/geometry/contract differs")
        if set(manifest["files"]) != {name for name, _ in _FILES.values()}:
            raise ValueError("route spool file inventory differs")
        self.manifest, self.counts, self.route_rows, self.recovery_rows = manifest, [], [], []
        try:
            for key, (name, dtype) in _FILES.items():
                path = self.root / name
                if path.is_symlink():
                    raise ValueError("refuse symlink route spool payload")
                verify_receipt(path, manifest["files"][name], checksum=True)
                shape = (tokens, width) if key == "inputs" else ((8, tokens) if key.startswith("candidate") else (tokens, 8))
                if path.stat().st_size != int(np.prod(shape)) * np.dtype(dtype).itemsize:
                    raise ValueError("route spool file extent differs")
                setattr(self, key, np.memmap(path, mode="r", dtype=dtype, shape=shape))
            # Bound temporary masks even on the full million-token panel.
            natural, recovery = [[] for _ in range(experts)], [[] for _ in range(experts)]
            counts = [0] * experts
            for start in range(0, tokens, 8192):
                stop = min(start + 8192, tokens)
                ids, near = self.routes[start:stop], self.candidates[:, start:stop].T
                _routes_valid(ids, self.weights[start:stop], near, self.candidate_weights[:, start:stop].T, experts)
                values = torch.from_numpy(np.array(self.inputs[start:stop], copy=True)).view(torch.bfloat16)
                if not torch.isfinite(values).all():
                    raise ValueError("nonfinite route spool BF16 inputs")
                for expert in np.unique(ids):
                    rows = np.flatnonzero(np.any(ids == expert, axis=1)) + start
                    natural[expert].append(rows)
                    counts[expert] += len(rows)
            self.counts = counts
            self.route_rows = [np.concatenate(rows) if rows else np.empty(0, dtype=np.int64) for rows in natural]
            missing = [max(0, minimum - count) for count in counts]
            # Unique top16 per token makes recovery disjoint from natural and
            # prevents a row appearing twice at different recovery ranks.
            for rank in range(8):
                for start in range(0, tokens, 8192):
                    values = self.candidates[rank, start:start + 8192]
                    for expert in np.unique(values):
                        if missing[expert]:
                            rows = (np.flatnonzero(values == expert) + start)[:missing[expert]]
                            recovery[expert].append(rows)
                            missing[expert] -= len(rows)
            self.recovery_rows = [np.concatenate(rows) if rows else np.empty(0, dtype=np.int64) for rows in recovery]
            if sum(self.counts) != tokens * 8:
                raise ValueError("route spool loses natural expert slots")
        except BaseException:
            self.close()
            raise

    @classmethod
    @torch.inference_mode()
    def create(cls, root, identity, tokens, batches, router, progress, layer,
               width=4608, experts=384, minimum=1024,
               *, candidate_gamma_contract=CANDIDATE_GAMMA_CONTRACT):
        _geometry(tokens, width, experts, minimum)
        root = Path(root)
        if root.is_symlink():
            raise ValueError("refuse symlink route spool")
        kwargs = dict(layer=layer, candidate_gamma_contract=candidate_gamma_contract)
        if (root / "manifest.json").exists():
            return cls(root, identity, tokens, width, experts, minimum, **kwargs)
        root.mkdir(parents=True, exist_ok=True)
        maps, offset = {}, 0
        try:
            for key, (name, dtype) in _FILES.items():
                if (root / name).is_symlink():
                    raise ValueError("refuse symlink route spool payload")
                shape = (tokens, width) if key == "inputs" else ((8, tokens) if key.startswith("candidate") else (tokens, 8))
                maps[key] = np.memmap(root / name, mode="w+", dtype=dtype, shape=shape)
            for index, row, state in batches:
                hidden = state["mlp_input"].reshape(-1, width)
                if hidden.dtype != torch.bfloat16 or not torch.isfinite(hidden).all():
                    raise ValueError("route capture requires finite native BF16 MLP inputs")
                if not len(hidden) or offset + len(hidden) > tokens:
                    raise ValueError("route capture exceeds declared token count")
                routed = router(hidden)
                if len(routed) != 4:
                    raise ValueError("router must return top8 and candidate IDs with native gamma")
                ids, gamma, near, near_gamma = [x.detach().cpu().numpy() for x in routed]
                _routes_valid(ids, gamma, near, near_gamma, experts)
                if len(ids) != len(hidden):
                    raise ValueError("route capture token geometry differs")
                if gamma.dtype != np.float32 or near_gamma.dtype != np.float32:
                    raise ValueError("route gamma must preserve native FP32 precision")
                end = offset + len(hidden)
                maps["inputs"][offset:end] = hidden.detach().cpu().contiguous().view(torch.uint16).numpy()
                maps["routes"][offset:end], maps["weights"][offset:end] = ids, gamma
                maps["candidates"][:, offset:end], maps["candidate_weights"][:, offset:end] = near.T, near_gamma.T
                offset = end
                if progress is not None and index % 64 == 0:
                    progress.emit("route_capture_progress", layer=layer, batches=index + 1)
            if offset != tokens:
                raise ValueError("route capture token count differs")
            for mapping in maps.values():
                mapping.flush()
                mapping._mmap.close()
            maps.clear()
            files = {}
            for name, _ in _FILES.values():
                with (root / name).open("rb") as stream:
                    os.fsync(stream.fileno())
                files[name] = dict(sha256=sha256_file(root / name), file_identity=file_identity(root / name))
            fsync_directory(root)
            atomic_json(root / "manifest.json", dict(schema=ROUTE_SCHEMA, identity=identity, layer=layer,
                tokens=tokens, width=width, experts=experts, minimum=minimum,
                candidate_gamma_contract=candidate_gamma_contract, files=files))
        finally:
            for mapping in maps.values():
                mapping._mmap.close()
        return cls(root, identity, tokens, width, experts, minimum, **kwargs)

    def _expert(self, expert):
        if type(expert) is not int or not 0 <= expert < self.experts:
            raise ValueError("expert outside route spool geometry")

    def rows(self, expert):
        self._expert(expert)
        return self.route_rows[expert], self.recovery_rows[expert]

    def gather(self, rows, device="cuda:0"):
        rows = self._rows(rows)
        return torch.from_numpy(np.array(self.inputs[rows], copy=True)).view(torch.bfloat16).to(device)

    def _rows(self, rows):
        rows = np.asarray(rows)
        if rows.ndim != 1 or rows.dtype.kind not in "iu" or np.any(rows < 0) or np.any(rows >= self.tokens):
            raise ValueError("route spool row outside geometry")
        return rows

    def gamma(self, expert, rows, device="cuda:0"):
        self._expert(expert)
        rows = self._rows(rows)
        natural, candidates = self.routes[rows] == expert, self.candidates[:, rows].T == expert
        if np.any(natural.sum(1) + candidates.sum(1) != 1):
            raise ValueError("expert has no unique native/candidate gamma for requested row")
        weights = np.sum(np.asarray(self.weights[rows]) * natural, axis=1)
        weights += np.sum(np.asarray(self.candidate_weights[:, rows].T) * candidates, axis=1)
        return torch.from_numpy(weights.copy()).to(device=device, dtype=torch.float32)

    @torch.inference_mode()
    def hessian(self, expert, down=None, device="cuda:0", chunk=2048, *, numerical=None,
                input_qdq=None, down_width=None):
        if type(chunk) is not int or chunk <= 0:
            raise ValueError("invalid Hessian activation chunk")
        if numerical is None:
            from . import codec as numerical
        if getattr(numerical, "ACCUMULATION_CONTRACT", None) != ACCUMULATION_CONTRACT:
            raise ValueError("bounded FP32-product/FP64-sum covariance contract differs")
        input_qdq = input_qdq or numerical.mxfp8_qdq
        width = self.width if down is None else (down_width or self.width // 2)
        state = numerical.covariance_accumulator(width, device)
        natural, recovery = self.rows(expert)
        for selection in (natural, recovery):
            for start in range(0, len(selection), chunk):
                rows = selection[start:start + chunk]
                x = input_qdq(self.gather(rows, device))
                if down is not None:
                    x = down(x, self.gamma(expert, rows, device))
                if x.shape != (len(rows), width) or not torch.isfinite(x).all():
                    raise ValueError("native expert Hessian observations differ")
                state.add(x)
        count = len(natural) + len(recovery)
        if state.count != count:
            raise ValueError("bounded covariance row count differs")
        H = state.snapshot()
        residual = max(0, self.minimum - count)
        H.diagonal().add_(float(residual))
        if H.dtype != torch.float32 or not torch.isfinite(H).all() or count + residual <= 0:
            raise ValueError("invalid raw routed covariance")
        evidence = dict(natural_rows=len(natural), augmented_rows=len(recovery), identity_rows=residual,
            effective_rows=count + residual, ranks=[9, 16], selection_policy="rank-major-then-original-corpus-row",
            recovery_selection_sha256=hashlib.sha256(recovery.astype("<i8", copy=False).tobytes()).hexdigest(),
            candidate_gamma_contract=self.manifest["candidate_gamma_contract"],
            input_quantization="native-mxfp8-e4m3-ue8m0-row-block32-qdq",
            down_inputs=None if down is None else "serialized-K4-gate-up-native-gamma-capped-swiglu-bf16-then-mxfp8-qdq",
            identity_contract="raw-XTX-missing-times-I-equivalent-to-normalized-2I",
            accumulation_contract=ACCUMULATION_CONTRACT)
        return dict(H=H, count=count + residual, finalized=False, identity_rows=residual,
                    accumulation_contract=ACCUMULATION_CONTRACT), evidence

    def close(self):
        for key in _FILES:
            mapping = getattr(self, key, None)
            if mapping is not None:
                mapping._mmap.close()
                setattr(self, key, None)


def serialize_attnres(state, *, metadata=None):
    """Expose the current boundary tensors without casting or stacking copies."""
    partial = state["partial"] if isinstance(state, dict) else state.partial
    blocks = state["blocks"] if isinstance(state, dict) else state.blocks
    if not isinstance(blocks, (list, tuple)) or not isinstance(partial, torch.Tensor):
        raise ValueError("invalid rolling AttnRes state")
    tensors = dict(partial=partial.detach())
    for index, block in enumerate(blocks):
        tensors[f"block_{index:05d}"] = block.detach()
    for value in tensors.values():
        if value.ndim != 2 or value.shape != partial.shape or value.dtype not in (torch.bfloat16, torch.float32):
            raise ValueError("invalid AttnRes tensor geometry/dtype")
    extra = {} if metadata is None else metadata
    json.dumps(extra, allow_nan=False)
    return tensors, dict(schema=STATE_SCHEMA, blocks=len(blocks),
        tensors={name: dict(shape=list(value.shape), dtype=str(value.dtype)) for name, value in tensors.items()},
        metadata=extra)


def unserialize_attnres(tensors, metadata, *, state_factory=None):
    if (metadata.get("schema") != STATE_SCHEMA or type(metadata.get("blocks")) is not int
            or metadata["blocks"] < 0):
        raise ValueError("invalid AttnRes state receipt")
    keys = {"partial"} | {f"block_{index:05d}" for index in range(metadata["blocks"])}
    if set(tensors) != keys or set(metadata["tensors"]) != keys:
        raise ValueError("AttnRes state tensor inventory differs")
    for name, value in tensors.items():
        if metadata["tensors"][name] != dict(shape=list(value.shape), dtype=str(value.dtype)):
            raise ValueError("AttnRes state tensor receipt differs")
    state = dict(partial=tensors["partial"], blocks=[tensors[f"block_{n:05d}"] for n in range(metadata["blocks"])])
    serialize_attnres(state)
    return state if state_factory is None else state_factory(state["partial"], state["blocks"])
