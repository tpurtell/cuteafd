"""Sequential Hy4 scratch stages and ordered expert dispatch."""
from __future__ import annotations

import fcntl
import json
import os
import re

import numpy as np
import torch
import xxhash

from kimi_k3.core_k6.storage import StageStore
from kimi_k3.uniform_k2.storage import atomic_json, canonical, checkpoint_io, fsync_directory, read_header, verify_receipt


_FIELDS = {"BF16": torch.bfloat16, "F32": torch.float32, "F16": torch.float16, "I64": torch.int64,
           "I32": torch.int32, "I16": torch.int16}


def record_bytes(values):
    """Size of the pinned checkpoint_io safetensors document, without touching tensor data."""
    header, offset = {}, 0
    for key, value in sorted(values.items()):
        end = offset + value.numel() * value.element_size()
        header[key] = dict(dtype=checkpoint_io()._DTYPES[str(value.dtype)], shape=list(value.shape),
                           data_offsets=[offset, end])
        offset = end
    size = len(json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode())
    return 8 + size + (-size % 8) + offset


class ActivationJournal:
    """A durable contiguous segment prefix, bound to its exact stage and numerical identity."""
    def __init__(self, identity, stage, count):
        self.count = count
        self.key = xxhash.xxh3_128_hexdigest(canonical(dict(
            schema="hy4-activation-segments-v1", identity=identity, stage=stage, count=count)).encode())

    def recover(self, directory):
        entries, committed = [], set()
        receipts = sorted(directory.glob("segment-*.json"))
        for ordinal, receipt in enumerate(receipts):
            stem = f"segment-{ordinal:05d}"
            payload = directory / (stem + ".bin")
            if receipt.name != stem + ".json" or receipt.is_symlink() or payload.is_symlink():
                raise ValueError("invalid activation segment sequence")
            saved = json.loads(receipt.read_text())
            group = saved["entries"]
            if (saved.get("key") != self.key or not group or
                    any(e["filename"] != payload.name for e in group)):
                raise ValueError("activation segment provenance differs")
            verify_receipt(payload, group[0])
            entries.extend(group)
            committed.update((receipt, payload))
        if entries:
            StageStore._validate_entries(entries)
        if len(entries) > self.count:
            raise ValueError("activation segment prefix exceeds the frozen prompt count")
        # A failure can leave only the next unreceipted payload, its partial,
        # or an unfinished atomic JSON rename. Never remove an unknown leaf.
        ordinal = len(receipts)
        allowed = {f"segment-{ordinal:05d}.bin", f"segment-{ordinal:05d}.bin.partial"}
        orphans = set(directory.iterdir()) - committed
        if any(p.is_symlink() or not p.is_file() or (p.name not in allowed and
               not re.fullmatch(r"\.json-[a-z0-9_]{8}", p.name)) for p in orphans):
            raise ValueError("unexpected uncommitted activation segment")
        for path in orphans:
            path.unlink()
        return entries, ordinal

    def seal(self, directory, ordinal, entries):
        atomic_json(directory / f"segment-{ordinal:05d}.json", dict(key=self.key, entries=entries))


class Stages(StageStore):
    """Expose committed record extents directly to the MC writer."""

    def write_batches(self, name, batches, metadata=None, *, segment_bytes=256 << 20, segment_batches=32):
        # A packed head can exceed the limit as one indivisible safetensors record.
        return super().write_batches(name, batches, metadata, segment_bytes=segment_bytes,
                                     segment_batches=segment_batches, record_bytes=record_bytes)

    def write_activations(self, name, batches, count, *, segment_bytes=256 << 20, segment_batches=32):
        directory = self.directory(name)
        if not directory.exists():
            directory.mkdir(exist_ok=True)
            fsync_directory(self.root)
        fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            # Reclaiming an incomplete tail requires exclusive ownership of this
            # stage. The directory lock is local to this filesystem transaction.
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            journal = ActivationJournal(self.identity, name, count)
            result = super().write_batches(name, batches, segment_bytes=segment_bytes,
                                           segment_batches=segment_batches, record_bytes=record_bytes, resume=journal)
        finally:
            os.close(fd)
        if len(result["entries"]) != count:
            raise ValueError("committed activation count differs from the frozen prompt count")
        return result

    def records(self, name, *, device="cpu", keys=None, start=0):
        manifest = self.load(name)
        if manifest is None:
            raise ValueError(f"Hy4 stage is missing: {name}")
        for index in range(start, len(manifest["entries"])):
            if keys is None:
                values = self.read_batch(name, index, device=device)
            else:
                values = self.fields(name, index, keys, device=device)
            yield index, values

    def fields(self, stage, index, keys, *, device="cpu"):
        """Read just the used tensors of a sealed record without rereading large peers."""
        path, _, offset, length = self.reference(stage, index, "")
        verify_receipt(path, self._manifests[stage]["entries"][index])
        header, base = read_header(path, offset=offset, length=length)
        values = {}
        with path.open("rb") as stream:
            for key in keys:
                spec = header[key]
                start, stop = spec["data_offsets"]
                dtype = _FIELDS[spec["dtype"]]
                if start == stop:
                    value = torch.empty(spec["shape"], dtype=dtype)
                else:
                    raw = bytearray(stop - start)
                    stream.seek(base + start)
                    if stream.readinto(raw) != len(raw):
                        raise ValueError("truncated Hy4 stage field")
                    value = torch.frombuffer(raw, dtype=dtype).reshape(spec["shape"])
                values[key] = value.to(device)
        return values

    def reference(self, stage, index, tensor):
        manifest = self._manifests.get(stage) or self.load(stage)
        if manifest is None:
            raise ValueError(f"Hy4 stage is missing: {stage}")
        record = manifest["entries"][index]
        return (self.directory(stage) / record["filename"], tensor,
                record["offset"], record["bytes"])


MINIMUM_EXPERT_ROWS = 1024
CANDIDATE_GAMMA_CONTRACT = "native-last-ranked-topk-substitution-renormalized-v1"
IDENTITY_CONTRACT = "raw-XTX-missing-times-I-equivalent-to-normalized-2I"


def candidate_routes(scores, choice, ids, config):
    """Select the next k admitted experts and substitute each at native rank k."""
    k = ids.shape[1]
    exposed = scores.shape[1] // config.n_group * config.topk_group
    width = min(k, exposed - k)
    remaining = choice.scatter(1, ids, -torch.inf)
    near = remaining.topk(width, dim=1, sorted=True).indices
    candidate_scores = scores.gather(1, near)
    if config.norm_topk_prob:
        # Hy4 returns unsorted native slots. Identify rank k using corrected
        # ranking scores; only the uncorrected sigmoid enters normalization.
        native_order = choice.gather(1, ids).argsort(dim=1, descending=True, stable=True)
        keep = scores.gather(1, ids.gather(1, native_order[:, :-1]))
        hypothetical = torch.cat((keep[:, None, :].expand(-1, width, -1),
                                  candidate_scores[:, :, None]), dim=2)
        candidate_scores = candidate_scores / (hypothetical.sum(dim=2) + 1e-20)
    return near.to(torch.int32), candidate_scores * config.routed_scaling_factor


def select_recovery(neighbors, gammas, natural_counts, first_rank, minimum):
    """Keep at most the deficit for each expert, by rank then global corpus row."""
    needed = np.maximum(minimum - np.asarray(natural_counts), 0)
    pieces = {}
    for column in range(neighbors.shape[1]):
        if not needed.any():
            break
        ids = neighbors[:, column].numpy()
        eligible_rows = np.flatnonzero(needed[ids])
        eligible_ids = ids[eligible_rows]
        for expert in np.unique(eligible_ids):
            rows = eligible_rows[eligible_ids == expert][:needed[expert]]
            selected = torch.from_numpy(rows)
            rank = torch.full_like(selected, first_rank + column)
            pieces.setdefault(int(expert), []).append((selected, gammas[selected, column], rank))
            needed[expert] -= len(rows)
    return {expert: tuple(torch.cat(field) for field in zip(*parts, strict=True))
            for expert, parts in pieces.items()}


class HostExpertStaging:
    """One reusable GPU row bank and one bounded host gather buffer for a layer."""

    def __init__(self, latent, device, chunk_rows, max_bytes):
        if latent.device.type != "cpu":
            raise ValueError("expert staging requires host-resident calibration inputs")
        width = latent.shape[1]
        capacity = min(len(latent), max_bytes // (width * latent.element_size()))
        if capacity < min(len(latent), chunk_rows):
            raise ValueError("expert staging must fit one numerical chunk")
        self.latent = latent
        self.chunk_rows = chunk_rows
        self.staged = torch.empty((capacity, width), dtype=latent.dtype, device=device)
        self.host_chunk = torch.empty((min(chunk_rows, capacity), width), dtype=latent.dtype,
                                      device="cpu", pin_memory=self.staged.is_cuda)

    def copy_rows(self, selected, target):
        host = self.host_chunk[:len(selected)]
        torch.index_select(self.latent, 0, selected, out=host)
        # Blocking H2D makes the host buffer immediately safe to refill. The
        # ordinary current stream orders reuse of the GPU slice with its readers.
        target.copy_(host)

    def select(self, rows):
        """Return a repeatable reader accepting a natural or augmented row count."""
        if len(rows) <= len(self.staged):
            for offset in range(0, len(rows), self.chunk_rows):
                part = rows[offset:offset + self.chunk_rows]
                self.copy_rows(part, self.staged[offset:offset + len(part)])
            return lambda count: self.staged[:count].split(self.chunk_rows)

        # An exceptionally hot expert streams the same ordered chunks on each
        # pass. Its row count never increases the GPU or pinned-memory ceiling.
        def stream(count):
            for part in rows[:count].split(self.chunk_rows):
                chunk = self.staged[:len(part)]
                self.copy_rows(part, chunk)
                yield chunk
        return stream


class Routes:
    """Stable expert-major view of immutable token-major route records."""

    candidate_gamma_contract = CANDIDATE_GAMMA_CONTRACT
    signed_scaling_factors = False

    def __init__(self, batches, experts, tokens, *, device="cpu", recover=False):
        self.latent = None
        self.recovery = None
        count = 0
        for _, batch in batches:
            latent, ids, gamma = (batch[k] for k in ("latent", "ids", "gamma"))
            if (latent.dtype != torch.float32 or ids.ndim != 2 or ids.shape != gamma.shape
                    or len(latent) != len(ids) or ids.dtype != torch.int64
                    or gamma.dtype != torch.float32
                    or bool((ids < 0).any()) or bool((ids >= experts).any())
                    or not bool(torch.isfinite(gamma).all())):
                raise ValueError("invalid Hy4 natural-route record")
            if recover:
                near, near_gamma = batch["candidate_ids"], batch["candidate_gamma"]
                if (near.dtype != torch.int32 or near_gamma.dtype != torch.float32 or near.ndim != 2
                        or near.shape != near_gamma.shape or len(near) != len(ids)
                        or near.shape[1] > ids.shape[1] or bool((near < 0).any())
                        or bool((near >= experts).any())
                        or (not self.signed_scaling_factors and bool((near_gamma < 0).any()))
                        or not bool(torch.isfinite(near_gamma).all())):
                    raise ValueError("invalid Hy4 candidate route record")
                all_ids = torch.cat((ids, near), dim=1).sort(dim=1).values
                if bool((all_ids[:, :-1] == all_ids[:, 1:]).any()):
                    raise ValueError("duplicate expert in Hy4 native and candidate ranks")
            if self.latent is None:
                self.latent = torch.empty((tokens, latent.shape[-1]), dtype=latent.dtype, device=device)
                choices = torch.empty((tokens, ids.shape[-1]), dtype=ids.dtype)
                weights = torch.empty_like(choices, dtype=gamma.dtype)
                if recover:
                    neighbors = torch.empty((tokens, near.shape[1]), dtype=near.dtype)
                    candidate_weights = torch.empty_like(neighbors, dtype=near_gamma.dtype)
            stop = count + len(latent)
            self.latent[count:stop].copy_(latent)
            choices[count:stop].copy_(ids)
            weights[count:stop].copy_(gamma)
            if recover:
                neighbors[count:stop].copy_(near)
                candidate_weights[count:stop].copy_(near_gamma)
            count = stop
        if count != tokens or self.latent is None:
            raise ValueError("Hy4 route records disagree with the corpus token count")
        width = choices.shape[-1]
        ids = choices.flatten()
        counts = torch.bincount(ids, minlength=experts)
        self.bounds = [0] + counts.cumsum(0).tolist()
        if recover:
            self.recovery = select_recovery(neighbors, candidate_weights, counts, width + 1, MINIMUM_EXPERT_ROWS)
        self.top_k = width
        order = torch.argsort(ids, stable=True)
        self.rows = order.div(width, rounding_mode="floor")
        self.gamma = weights.flatten()[order]
        sorted_ids = ids[order]
        same_expert = sorted_ids[1:] == sorted_ids[:-1]
        if bool(((self.rows[1:] == self.rows[:-1]) & same_expert).any()):
            raise ValueError("Hy4 router assigned the same expert twice for a token")

    def expert(self, index):
        start, stop = self.bounds[index:index + 2]
        return self.rows[start:stop], self.gamma[start:stop]

    def input_chunks(self, selected, chunk_rows, staging):
        if staging is not None:
            return staging.select(selected)
        rows = selected.to(self.latent.device)

        def gather(count):
            for part in rows[:count].split(chunk_rows):
                yield self.latent.index_select(0, part)
        return gather

    def calibration(self, index):
        if self.recovery is None:
            raise ValueError("missing candidate route evidence for expert calibration")
        natural, native_gamma = self.expert(index)
        addition = self.recovery.get(index)
        if addition is None:
            near = ranks = torch.empty(0, dtype=torch.int64)
            near_gamma = torch.empty(0, dtype=torch.float32)
        else:
            near, near_gamma, ranks = addition
        count = len(natural) + len(near)
        identity = max(0, MINIMUM_EXPERT_ROWS - count)
        digest = xxhash.xxh3_128()
        for field in (natural, native_gamma, near, near_gamma, ranks):
            digest.update(len(field).to_bytes(8, "little"))
            digest.update(field.contiguous().numpy())
        rank_counts = torch.bincount(ranks, minlength=2 * self.top_k + 1)
        evidence = dict(natural_rows=len(natural), augmented_rows=len(near), identity_rows=identity,
                        effective_rows=count + identity, candidate_rank_first=self.top_k + 1,
                        augmented_rank_counts=rank_counts[self.top_k + 1:].tolist(),
                        augmented_corpus_rows=near.tolist(),
                        selection_xxh3_128=digest.hexdigest(), selection_policy="rank-major-then-corpus-row",
                        candidate_gamma_contract=self.candidate_gamma_contract,
                        identity_contract=IDENTITY_CONTRACT)
        if addition is None:
            return natural, native_gamma, evidence
        return torch.cat((natural, near)), torch.cat((native_gamma, near_gamma)), evidence

    def combined(self, outputs):
        """Accumulate weighted outputs in the reference's ascending expert order."""
        combined = torch.zeros_like(self.latent)
        expected = len(self.bounds) - 1
        count = 0
        for expert, weighted in outputs:
            if count >= expected or expert != count:
                raise ValueError("Hy4 expert outputs are out of order")
            rows, _ = self.expert(expert)
            if weighted.shape != (len(rows), self.latent.shape[-1]) or weighted.dtype != combined.dtype:
                raise ValueError("Hy4 weighted output differs from its natural route geometry")
            combined.index_add_(0, rows, weighted)
            count += 1
        if count != expected:
            raise ValueError("Hy4 expert output inventory is incomplete")
        return combined
