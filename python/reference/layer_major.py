"""Layer-major F32 fidelity golden runner shared by families built on model.py adapters.

One decoder layer is resident at a time; every window's residual stream (and any
cross-layer family state) is a durable NVMe file. Only an atomic, fsynced
``latest.json`` advances resume, so an interruption loses at most one layer.
Prefix qualification runs this same loop on two prefixes of one window first
(shared ``qualify_prefix``), plus an independent untiled official-geometry arm on a
short window whose logits must agree with the fixed-geometry arm.

A family supplies a ``Family`` object:
  name, config, checkpoint, layers (count),
  layer_tensors(index) -> checkpoint tensor names the layer reads,
  load_layer(index) -> module, release_layer(module),
  embed(tokens) -> [1, T, H] F32, forward(module, index, hidden, carry) -> (hidden, carry),
  carry_spec(index) -> names of tensors forwarded to later layers (persisted),
  final(hidden) -> F32 logits rows, geometry() context, official_geometry() context,
  provenance() -> dict, observe(module, index) -> context yielding per-window extras.
"""
from __future__ import annotations

from contextlib import nullcontext
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

from fidelity_windows import (admit_reference_memory, canonical, finish_golden, load_set,
                              log_reference_cuda_memory, verify_snapshot, write_scored_logits)

HEADROOM = 100 * 2 ** 30
# Reference device; CPU only for the synthetic contract tests.
DEVICE = "cuda"


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 2 ** 20), b""):
            value.update(block)
    return value.hexdigest()


def sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_json(path, value):
    temporary = path.with_suffix(".pending")
    with temporary.open("wb") as stream:
        stream.write(canonical(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    sync_directory(path.parent)


class LayerState:
    """Durable per-window tensors after each committed layer (hidden + family carry)."""

    def __init__(self, root, binding):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        latest = self.root / "latest.json"
        self.latest = json.loads(latest.read_bytes()) if latest.exists() else None
        if self.latest and self.latest["binding"] != binding:
            raise ValueError("resume source, set, arithmetic or runtime differs")
        self.binding = binding
        if self.latest:
            for entry in self.latest["windows"].values():
                for blob in entry.values():
                    self.verify(blob)

    def contained(self, name):
        path = self.root / name
        if not path.resolve().is_relative_to(self.root.resolve()) or path.is_symlink():
            raise ValueError("unsafe reference state path")
        return path

    def verify(self, blob):
        path = self.contained(blob["path"])
        if path.stat().st_size != blob["bytes"] or digest(path) != blob["sha256"]:
            raise ValueError("reference state blob differs")
        return path

    def load(self, wid, device=None):
        import numpy as np
        import torch
        values = {}
        for name, blob in self.latest["windows"][wid].items():
            raw = np.fromfile(self.contained(blob["path"]), dtype=np.dtype(blob["dtype"]))
            values[name] = torch.from_numpy(raw.reshape(blob["shape"])).to(device or DEVICE)
        return values

    def begin(self, layer):
        if shutil.disk_usage(self.root).free < self.binding["state_bytes_per_layer"] + HEADROOM:
            raise ValueError("reference state write would violate 100 GiB NVMe headroom")
        folder = self.root / ("layer%03d" % (layer + 1))
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir()
        previous = self.latest["checks"] if self.latest else []
        return {"schema": "cuteafd.layer-major.state/1", "binding": self.binding, "layer": layer,
                "folder": folder.name, "windows": {}, "checks": list(previous),
                "seconds_per_layer": list(self.latest.get("seconds_per_layer", [])) if self.latest else []}

    def save(self, pending, wid, values, reuse=()):
        """Write this window's tensors; names in ``reuse`` keep the previous layer's blob."""
        import torch
        entry = {}
        for name, value in values.items():
            if name in reuse:
                entry[name] = self.latest["windows"][wid][name]
                continue
            if value.dtype == torch.float32 and not torch.isfinite(value).all():
                raise ValueError(f"reference state {name} must remain finite")
            array = value.detach().cpu().contiguous().numpy()
            path = self.root / pending["folder"] / f"{wid}.{name}.bin"
            with path.open("wb") as stream:
                array.tofile(stream)
                stream.flush()
                os.fsync(stream.fileno())
            entry[name] = {"path": str(path.relative_to(self.root)), "bytes": path.stat().st_size,
                           "sha256": digest(path), "dtype": array.dtype.str, "shape": list(array.shape)}
        pending["windows"][wid] = entry

    def commit(self, pending):
        sync_directory(self.root / pending["folder"])
        atomic_json(self.root / "latest.json", pending)
        previous, self.latest = self.latest, pending
        if previous:
            referenced = {Path(b["path"]).parts[0] for w in pending["windows"].values() for b in w.values()}
            if previous["folder"] not in referenced:
                shutil.rmtree(self.root / previous["folder"])
            # Older folders that nothing references any more (carry blobs moved on).
            for folder in self.root.glob("layer*"):
                if folder.is_dir() and folder.name not in referenced and folder.name != pending["folder"]:
                    shutil.rmtree(folder)


def window_tokens(window):
    return len(window["tokens"])


def run(a, family, *, extras_root=None):
    """Full layer-major panel (or a nested prefix probe) for one family."""
    import numpy as np
    import torch
    import transformers
    import tokenizers
    from shape_invariant import qualify

    checking = getattr(a, "_prefix_probe", False)
    started = time.monotonic()
    a.out.mkdir(parents=True, exist_ok=True)
    arithmetic = family.configure()
    torch.manual_seed(0)
    manifest = load_set(a.windows)
    if manifest["family"] != family.name or any(w.get("media") for w in manifest["windows"]):
        raise ValueError(f"{family.name} golden supports only its own text panel")
    snapshot = verify_snapshot(manifest, a.snapshot)
    # Header-only preflight for every layer already downloaded; the rest are
    # checked right after their shards complete, before any of their bytes load.
    present = [i for i in range(family.layers) if not pending_shards(family, i)]
    for index in present:
        family.check_layer(index)
    print(f"native shape preflight passed for {len(present)}/{family.layers} downloaded layers", flush=True)
    pins = family.provenance()
    marker = a.out / "qualified-gate.json"
    if not checking and marker.exists():
        # Resume: the prefix proof and official arm already passed for this exact binding.
        gate = json.loads(marker.read_bytes())
        proof = json.loads((a.out / "prefix-qualification.json").read_bytes())
        if (gate.get("set_sha256") != manifest["set_sha256"] or gate.get("source_pin") != pins
                or hashlib.sha256(canonical(proof)).hexdigest() != gate.get("proof_sha256")):
            raise ValueError("stale prefix qualification; remove the output directory to requalify")
        print(f"reusing prefix qualification {gate['gate']}", flush=True)
    else:
        proof = qualify(a, manifest, lambda probe: run(probe, family, extras_root=None))
        if not checking:
            gates = [g for g in a.out.glob("prefix-gate-*") if g.is_dir()
                     and (g / "qualification.json").exists()
                     and json.loads((g / "qualification.json").read_bytes()) == proof]
            if len(gates) != 1 or not (gates[0] / "official-forward.json").exists():
                raise ValueError("cannot identify the passing prefix gate")
            atomic_json(marker, {"gate": gates[0].name, "set_sha256": manifest["set_sha256"], "source_pin": pins,
                                 "proof_sha256": hashlib.sha256(canonical(proof)).hexdigest()})
    if getattr(a, "prefix_only", False) and not checking:
        print("prefix qualification only; no full panel", flush=True)
        return proof
    windows = copy.deepcopy(manifest["windows"])
    if checking:
        # Independent unmodified-geometry arm on a short official window.
        source = copy.deepcopy(windows[0])
        length = min(len(source["tokens"]), getattr(a, "official_tokens", 512))
        for name in ("official_short", "adapted_short"):
            short = copy.deepcopy(source)
            short.update(id=name, tokens=source["tokens"][:length], score_from=1, roles=["ctx"] * length)
            windows.append(short)
    state_root = a.out.parent / (a.out.name + "-layer-state")
    binding = {"set_sha256": manifest["set_sha256"], "source_pin": pins, "snapshot_identity": snapshot,
               "torch": torch.__version__, "transformers": transformers.__version__,
               "compute_capability": list(torch.cuda.get_device_capability()) if DEVICE == "cuda" else [],
               "image_sha256": os.environ.get("CUTEAFD_REFERENCE_IMAGE_SHA256", ""),
               "arithmetic": arithmetic, "state_dtype": "float32",
               "state_bytes_per_layer": sum(family.state_bytes(window_tokens(w)) for w in windows),
               "windows": {w["id"]: hashlib.sha256(canonical(w["tokens"])).hexdigest() for w in windows}}
    state = LayerState(state_root, binding)
    with torch.inference_mode():
        if state.latest is None:
            wait_for_tensors(family, family.outer_tensors("embed"), getattr(a, "wait_shards", 0))
            pending = state.begin(-1)
            for window in windows:
                state.save(pending, window["id"], {"hidden": family.embed(window["tokens"])})
            family.release_outer()
            state.commit(pending)
        print(f"resume after layer {state.latest['layer']}; {len(windows)} windows", flush=True)
        for index in range(state.latest["layer"] + 1, family.layers):
            tick = time.monotonic()
            read_before, read_seconds_before = family.checkpoint.read_bytes, family.checkpoint.read_seconds
            wait_for_shards(family, index, getattr(a, "wait_shards", 0))
            family.check_layer(index)
            if DEVICE == "cuda":
                admit_reference_memory(torch.cuda, family.layer_bytes(index) + family.work_bytes(
                    max(window_tokens(w) for w in windows)))
            module = family.load_layer(index)
            load_seconds = time.monotonic() - tick
            pending = state.begin(index)
            short = {}
            extras = {}
            for window in windows:
                wid = window["id"]
                values = state.load(wid)
                hidden = values.pop("hidden")
                official = wid == "official_short"
                context = family.official_geometry() if official else family.geometry()
                with context, family.observe(module, index) as observed:
                    hidden, carry, reuse = family.forward(module, index, hidden, values)
                state.save(pending, wid, {"hidden": hidden, **carry}, reuse=reuse)
                if observed is not None and extras_root is not None and not checking:
                    extras[wid] = observed
                if checking and wid in ("official_short", "adapted_short"):
                    short[wid] = hidden.float().cpu()
                del hidden, carry, values
                if DEVICE == "cuda":
                    # GB10 shares memory with the host: return cached blocks between windows.
                    torch.cuda.empty_cache()
            if short:
                difference = short["official_short"] - short["adapted_short"]
                pending["checks"].append({"layer": index, "max_abs_hidden_difference": difference.abs().max().item(),
                                          "relative_hidden_l2": (difference.norm() /
                                              short["official_short"].norm().clamp_min(1e-30)).item()})
            if extras:
                family.write_extras(extras_root, index, extras, windows)
            seconds = time.monotonic() - tick
            pending["seconds_per_layer"].append({
                "layer": index, "seconds": seconds, "load_seconds": load_seconds,
                "read_bytes": family.checkpoint.read_bytes - read_before,
                "read_seconds": family.checkpoint.read_seconds - read_seconds_before})
            state.commit(pending)
            family.release_layer(module)
            del module
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
                log_reference_cuda_memory(torch.cuda, f"after layer {index}")
            print(f"layer {index + 1}/{family.layers} committed; {seconds:.1f}s (load {load_seconds:.1f}s); "
                  f"read {(family.checkpoint.read_bytes - read_before) / 1e9:.2f} GB", flush=True)
        wait_for_tensors(family, family.outer_tensors("final"), getattr(a, "wait_shards", 0))
        family.load_final()
        entries, short_logits = [], {}
        total_nll, total_rows = 0.0, 0
        for window in windows:
            wid = window["id"]
            hidden = state.load(wid)["hidden"]
            start = window["score_from"]
            selected = hidden[0, start - 1:len(window["tokens"]) - 1]
            logits = family.final(selected).float().cpu().numpy()
            if not np.isfinite(logits).all():
                raise ValueError("non-finite reference logits")
            if wid in ("official_short", "adapted_short"):
                short_logits[wid] = logits
            else:
                entry = write_scored_logits(a.out, window, logits)
                atomic_json(a.out / entry["path"] / "meta.json", entry)
                entries.append(entry)
                total_nll += entry["nll"] * entry["scored_tokens"]
                total_rows += entry["scored_tokens"]
            del hidden, selected, logits
        family.release_final()
    if checking:
        comparison = official_check(a.out, short_logits, state.latest["checks"])
    else:
        evidence = a.out / json.loads(marker.read_bytes())["gate"] / "official-forward.json"
        if not evidence.exists():
            raise ValueError("missing independent official-geometry evidence")
        comparison = json.loads(evidence.read_bytes())
        shutil.copyfile(evidence, a.out / "official-forward.json")
    finish_golden(a.out, manifest, entries, snapshot_identity=snapshot, source_pin=pins,
                  nll=total_nll / total_rows, scored_tokens=total_rows, seconds=time.monotonic() - started,
                  seconds_per_layer=state.latest["seconds_per_layer"],
                  torch=torch.__version__, transformers=transformers.__version__, tokenizers=tokenizers.__version__,
                  attention="eager", mtp=False, use_cache=False, prefix_qualification=proof, fixed_rows=128,
                  official_forward=comparison, arithmetic=arithmetic, **family.golden_meta(),
                  harness_commit=os.environ.get("CUTEAFD_REFERENCE_HARNESS_COMMIT", ""),
                  platform={"capability": list(torch.cuda.get_device_capability()),
                            "gpu": torch.cuda.get_device_name(), "cuda": torch.version.cuda}
                  if DEVICE == "cuda" else {"device": DEVICE})
    print(f"reference NLL {total_nll / total_rows:.12f}; {total_rows} scored rows", flush=True)
    return proof


def shard_complete(path):
    """A shard is readable once its size reaches the header's declared data extent."""
    import struct
    try:
        size = path.stat().st_size
        with path.open("rb") as stream:
            count = struct.unpack("<Q", stream.read(8))[0]
            header = json.loads(stream.read(count))
    except (OSError, ValueError, struct.error):
        return False
    end = max((v["data_offsets"][1] for k, v in header.items() if k != "__metadata__"), default=0)
    return size == 8 + count + end


def pending_shards(family, index):
    return missing_shards(family, family.tensors(index))


def missing_shards(family, names):
    shards = sorted({family.checkpoint.index[n] for n in names})
    return [s for s in shards if not shard_complete(family.checkpoint.snapshot / s)]


def wait_for_tensors(family, names, timeout):
    deadline = time.monotonic() + timeout
    while missing_shards(family, names):
        if time.monotonic() >= deadline:
            raise ValueError(f"outer tensors not downloaded: {missing_shards(family, names)[:4]}")
        time.sleep(120)


def wait_for_shards(family, index, timeout):
    """Block (sleeping, not spinning) until every shard the layer reads is complete."""
    deadline = time.monotonic() + timeout
    announced = False
    while True:
        pending = pending_shards(family, index)
        if not pending:
            return
        if time.monotonic() >= deadline:
            raise ValueError(f"layer {index} shards not complete: {pending[:4]}")
        if not announced:
            print(f"layer {index} waiting for {len(pending)} shard(s): {pending[:4]}", flush=True)
            announced = True
        time.sleep(120)


def official_check(out, logits, per_layer):
    """Untiled official geometry vs fixed geometry on the same short window."""
    import numpy as np
    native, adapted = logits["official_short"], logits["adapted_short"]
    logp = native.astype(np.float64)
    candidate = adapted.astype(np.float64)
    logp -= np.logaddexp.reduce(logp, axis=-1, keepdims=True)
    candidate -= np.logaddexp.reduce(candidate, axis=-1, keepdims=True)
    kl = float((np.exp(logp) * (logp - candidate)).sum(-1).mean())
    top1 = float(np.mean(native.argmax(-1) == adapted.argmax(-1)))
    relative = float(np.linalg.norm(native - adapted) / max(np.linalg.norm(native), 1e-30))
    result = {"schema": "cuteafd.layer-major.official-forward/1", "tokens": len(native),
              "top1": top1, "kl": kl, "relative_logit_l2": relative,
              "max_abs_logit_difference": float(np.abs(native - adapted).max()),
              "comparison": "independent streams: official eager attention, official linears, official indexer "
                            "versus fixed-M128 linears and 128-row causal tiles; same exact checkpoint weights, F32",
              "passed": top1 >= .99 and kl <= 1e-4 and relative <= .01,
              "limits": {"top1_min": .99, "kl_max": 1e-4, "relative_logit_l2_max": .01},
              "layers": per_layer}
    atomic_json(out / "official-forward.json", result)
    print(f"official-geometry check: top1 {top1:.4f} kl {kl:.3e} rel {relative:.3e}", flush=True)
    if not result["passed"]:
        raise ValueError("official-geometry numerical check failed")
    return result


def no_observation(module, index):
    return nullcontext(None)
