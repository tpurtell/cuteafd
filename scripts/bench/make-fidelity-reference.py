#!/usr/bin/env python3
"""Compact top-k fidelity reference from a family golden run.

The golden harnesses (python/reference/families/<id>/golden.py) write the
reference model's full logits for a prompt (tokens.bin i32 [T], logits.bin
f32 [T, vocab], meta.json). Explicit local diagnostic runs can score the
served model on the same tokens and compare against a compact summary of
those rows: per position the reference's top-k ids and log-probabilities,
the log of the remaining tail mass and the next token's log-probability.

    make-fidelity-reference.py --golden runs/qwen4-golden --model 'Qwen/Qwen3.8-Flash-Next*' \
        --out runs/qwen4-reference.json
"""
import argparse
import hashlib
import json
import pathlib

import numpy as np
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "python/reference"))
from fidelity_windows import canonical, load_set, validate_qualification


def tail_log_mass(logp, order):
    # Sum the omitted probabilities directly: 1 - top_mass cancels for peaked rows.
    omitted = np.ones(logp.size, dtype=bool)
    omitted[order] = False
    return float(np.log(max(np.exp(logp[omitted]).sum(dtype=np.float64), 1e-30)))


def convert_windows(args) -> None:
    manifest = load_set(args.windows)
    meta = json.loads((args.golden / "meta.json").read_text())
    if (meta.get("set_sha256") != manifest["set_sha256"] or meta.get("checkpoint") != manifest["checkpoint"]
            or meta.get("family") != manifest["family"]):
        raise ValueError("golden/set checkpoint or set hash mismatch")
    validate_qualification(meta.get("prefix_qualification"), manifest, meta.get("snapshot_identity"))
    entries = {entry["id"]: entry for entry in meta["windows"]}
    if len(entries) != len(meta["windows"]) or set(entries) != {w["id"] for w in manifest["windows"]}:
        raise ValueError("golden must contain exactly the set's windows")
    if not args.rows_dir:
        raise ValueError("--rows-dir is required with --windows (full-vocabulary f16 rows)")
    args.rows_dir.mkdir(parents=True, exist_ok=True)
    windows, row_manifest, vocab = [], [], None
    k = args.top_k if args.top_k is not None else 32
    for w in manifest["windows"]:
        entry = entries[w["id"]]
        positions = list(range(w["score_from"], len(w["tokens"])))
        if entry["positions"] != positions:
            raise ValueError("golden positions differ from the pinned set")
        folder = (args.golden / entry["path"]).resolve()
        if not folder.is_relative_to(args.golden.resolve()):
            raise ValueError("golden path escapes its directory")
        pinned = np.fromfile(folder / "tokens.bin", dtype="<i4").tolist()
        if pinned != w["tokens"]:
            raise ValueError("golden tokens differ from the pinned set")
        current_vocab = entry["vocab"]
        if current_vocab != meta["prefix_qualification"]["vocab"]:
            raise ValueError("prefix qualification vocabulary differs from golden")
        if vocab is not None and current_vocab != vocab:
            raise ValueError("mixed vocabularies in golden")
        vocab = current_vocab
        if not 1 <= k <= vocab:
            raise ValueError("top-k must be in [1,vocab]")
        logits = np.memmap(folder / "logits.bin", dtype="<f4", mode="r")
        if logits.size != len(positions) * vocab:
            raise ValueError("scored logits shape does not match metadata")
        logits = logits.reshape(len(positions), vocab)
        row_path = args.rows_dir / (w["id"] + ".bin")
        compact = []
        with row_path.open("wb") as full:
            for row, pos in zip(logits, positions):
                row = row.astype(np.float64)
                if not np.isfinite(row).all():
                    raise ValueError("non-finite golden logits")
                logp = row - (row.max() + np.log(np.exp(row - row.max()).sum()))
                logp.astype("<f2").tofile(full)
                # Include the exact legacy ordering/rounding when logits are not tied.
                # Boundary ties use the lower token id, independent of argpartition.
                threshold = np.partition(row, vocab - k)[vocab - k]
                better = np.flatnonzero(row > threshold)
                ties = np.flatnonzero(row == threshold)[:k - len(better)]
                order = np.concatenate((better, ties))
                order = order[np.lexsort((order, -row[order]))]
                nxt = w["tokens"][pos]
                if nxt >= vocab:
                    raise ValueError("next token outside golden vocabulary")
                compact.append({"pos": pos, "next": nxt, "next_lp": round(float(logp[nxt]), 5),
                    "top": [{"id": int(i), "lp": round(float(logp[i]), 5)} for i in order],
                    "tail_lp": round(tail_log_mass(logp, order), 5)})
        if entry.get("media", []) != w.get("media", []):
            raise ValueError("golden media identity differs from pinned set")
        windows.append({**{key: w[key] for key in ("id", "block", "bucket", "tokens", "roles", "score_from")},
                        **({"media": w["media"]} if w.get("media") else {}),
                        "positions": compact, "top_k": k,
                        "nll": round(-float(np.mean([r["next_lp"] for r in compact])), 6)})
        row_manifest.append({"id": w["id"], "path": row_path.name,
            "sha256": file_hash(row_path), "positions": positions, "shape": [len(positions), vocab]})
    reference = {"schema": "cuteafd.fidelity.reference/2", "name": args.name or args.model[0],
        "models": args.model, "model": manifest["model"], "family": manifest["family"],
        "checkpoint": manifest["checkpoint"], "set_sha256": manifest["set_sha256"], "vocab": vocab,
        "tokenizer_sha256": manifest.get("tokenizer_sha256"), "generation_arm": manifest["generation_arm"],
        "generation_server": manifest.get("generation_server", {}), "set_version": manifest.get("version"),
        "source": {"golden": args.golden.name, "snapshot": meta.get("snapshot"),
                   "reference": meta.get("reference"), "experts_snapshot": meta.get("experts_snapshot"),
                   "snapshot_identity": meta.get("snapshot_identity"),
                   "prefix_qualification": meta["prefix_qualification"]},
        "expect": {"kl_max": args.kl_max if args.kl_max is not None else 0.06,
                   "top1_min": args.top1_min if args.top1_min is not None else 0.90},
        "quick_windows": manifest["quick_windows"], "windows": windows}
    rows = {"schema": "cuteafd.fidelity.rows/1", "family": manifest["family"],
            "checkpoint": manifest["checkpoint"], "set_sha256": manifest["set_sha256"],
            "vocab": vocab, "dtype": "<f2", "kind": "log_softmax", "windows": row_manifest}
    (args.rows_dir / "rows.json").write_bytes(canonical(rows) + b"\n")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(canonical(reference) + b"\n")
    if args.quick_out:
        quick = {**reference, "windows": [w for w in windows if w["id"] in manifest["quick_windows"]]}
        args.quick_out.parent.mkdir(parents=True, exist_ok=True)
        args.quick_out.write_bytes(canonical(quick) + b"\n")
    print(f"{args.out}: {len(windows)} windows, {sum(len(w['positions']) for w in windows)} positions, top-{k}")


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for data in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--golden", required=True, type=pathlib.Path)
    parser.add_argument("--out", required=True, type=pathlib.Path)
    parser.add_argument("--model", action="append", required=True,
                        help="checkpoint id pattern this reference applies to (fnmatch, repeatable)")
    parser.add_argument("--name", help="display name (default: the first pattern)")
    parser.add_argument("--positions", type=int, default=512, help="scored positions from --from")
    parser.add_argument("--from", dest="start", type=int, default=1)
    parser.add_argument("--top-k", type=int, help="default: 12 legacy, 32 schema 2")
    parser.add_argument("--kl-max", type=float)
    parser.add_argument("--top1-min", type=float)
    parser.add_argument("--windows", type=pathlib.Path, help="pinned fidelity set manifest (schema 2 output)")
    parser.add_argument("--legacy-window", help="emit schema 1 from one qualified scored-row window, e.g. legacy")
    parser.add_argument("--rows-dir", type=pathlib.Path, help="full f16 log-softmax row directory, e.g. on sparknest")
    parser.add_argument("--quick-out", type=pathlib.Path, help="optional compact quick-subset reference")
    args = parser.parse_args()
    if args.windows and args.legacy_window:
        parser.error("--windows and --legacy-window select different output schemas")
    if args.windows:
        convert_windows(args)
        return
    args.top_k = args.top_k if args.top_k is not None else 12
    args.kl_max = args.kl_max if args.kl_max is not None else 0.15
    args.top1_min = args.top1_min if args.top1_min is not None else 0.80

    meta = json.loads((args.golden / "meta.json").read_text())
    if args.legacy_window:
        manifest = load_set(args.golden / "windows.json")
        if (meta.get("set_sha256") != manifest["set_sha256"] or meta.get("checkpoint") != manifest["checkpoint"]
                or meta.get("family") != manifest["family"]):
            raise ValueError("golden/set checkpoint or set hash mismatch")
        validate_qualification(meta.get("prefix_qualification"), manifest, meta.get("snapshot_identity"))
        windows = [w for w in manifest["windows"] if w["id"] == args.legacy_window]
        entries = [w for w in meta["windows"] if w["id"] == args.legacy_window]
        if len(windows) != 1 or len(entries) != 1:
            raise ValueError("expected exactly one selected legacy window")
        window, entry = windows[0], entries[0]
        args.start = window["score_from"]
        positions = list(range(args.start, len(window["tokens"])))
        if entry["positions"] != positions:
            raise ValueError("golden positions differ from the pinned set")
        folder = (args.golden / entry["path"]).resolve()
        if not folder.is_relative_to(args.golden.resolve()):
            raise ValueError("golden path escapes its directory")
        tokens = np.fromfile(folder / "tokens.bin", dtype="<i4")
        if tokens.tolist() != window["tokens"]:
            raise ValueError("golden tokens differ from the pinned set")
        vocab = entry["vocab"]
        if vocab != meta["prefix_qualification"]["vocab"]:
            raise ValueError("prefix qualification vocabulary differs from golden")
        logits = np.memmap(folder / "logits.bin", dtype="<f4", mode="r")
        if logits.size != len(positions) * vocab:
            raise ValueError("scored logits shape does not match metadata")
        logits = logits.reshape(len(positions), vocab)
        row_offset = args.start
    else:
        tokens = np.fromfile(args.golden / "tokens.bin", dtype=np.int32)
        t = len(tokens)
        logits = np.memmap(args.golden / "logits.bin", dtype=np.float32, mode="r")
        vocab = logits.size // t
        assert vocab * t == logits.size, "logits.bin is not [T, vocab]"
        logits = logits.reshape(t, vocab)
        row_offset = 1
    if not 1 <= args.top_k < vocab or args.positions < 1:
        raise ValueError("invalid top-k or number of positions")
    end = min(len(tokens), args.start + args.positions)
    snapshot = pathlib.Path(meta.get("snapshot", ""))
    # Goldens run in containers that mount the hub at /root/.cache/huggingface/hub.
    container_hub = pathlib.Path("/root/.cache/huggingface/hub")
    local = pathlib.Path.home() / ".cache/huggingface/hub" / snapshot.relative_to(container_hub) \
        if snapshot.is_relative_to(container_hub) else snapshot
    tokenizer = local / "tokenizer.json"
    ids, lps, tail, nxt = [], [], [], []
    for p in range(args.start, end):
        # Panel goldens store only selected prediction rows, unlike old full-T goldens.
        row = logits[p - row_offset].astype(np.float64)
        if not np.isfinite(row).all() or not 0 <= tokens[p] < vocab:
            raise ValueError("non-finite logits or next token outside golden vocabulary")
        top = row.max()
        lse = top + np.log(np.exp(row - top).sum())
        logp = row - lse
        if args.legacy_window:
            threshold = np.partition(row, vocab - args.top_k)[vocab - args.top_k]
            better = np.flatnonzero(row > threshold)
            ties = np.flatnonzero(row == threshold)[:args.top_k - len(better)]
            order = np.concatenate((better, ties))
        else:
            order = np.argpartition(-row, args.top_k)[: args.top_k]
        order = order[np.lexsort((order, -row[order]))]
        mass = np.exp(logp[order]).sum()
        ids.append([int(i) for i in order])
        lps.append([round(float(v), 5) for v in logp[order]])
        tail.append(round(tail_log_mass(logp, order) if args.legacy_window
                          else float(np.log(max(1.0 - mass, 1e-30))), 5))
        nxt.append(round(float(logp[tokens[p]]), 5))
    reference = {
        "schema": "cuteafd.bench.reference/1",
        "name": args.name or args.model[0],
        "models": args.model,
        "source": {
            "golden": args.golden.name,
            "snapshot": str(snapshot),
            "reference": meta.get("reference", "family golden.py"),
            "experts_snapshot": meta.get("experts_snapshot"),
        },
        "tokenizer_sha256": hashlib.sha256(tokenizer.read_bytes()).hexdigest() if tokenizer.exists() else None,
        "vocab": vocab,
        "tokens": [int(x) for x in tokens[:end]],
        "score_from": args.start,
        "top_k": args.top_k,
        "ids": ids,
        "lps": lps,
        "tail_lp": tail,
        "next_lp": nxt,
        "nll": round(-float(np.mean(nxt)), 6),
        "expect": {"kl_max": args.kl_max, "top1_min": args.top1_min},
    }
    if args.legacy_window:
        reference["source"].update(window=args.legacy_window, set_sha256=manifest["set_sha256"],
            snapshot_identity=meta["snapshot_identity"], prefix_qualification=meta["prefix_qualification"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(reference, separators=(",", ":")) + "\n")
    print(f"{args.out}: {end - args.start} positions, top-{args.top_k}, ref NLL {reference['nll']:.4f}, "
          f"{args.out.stat().st_size / 1024:.0f} KiB")


if __name__ == "__main__":
    main()
