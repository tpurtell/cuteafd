#!/usr/bin/env python3
"""Teacher-KL cross-check for GLM 5.3 Flash: a served engine's full-vocabulary rows for a teacher panel.

Opt-in and run by hand; not a gate or a dashboard tier. cuteafd's quality and precision gate is
``cuteafd bench fidelity run`` against the public fidelity dataset; this is a second reference
for changes that move numerics. It scores the engine against published BF16-teacher logits
(``brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits``) with ``harness/klgate.py`` from
github.com/hughmadden/glm53f-afd (MIT, standard library only):

    klgate.py plan --teacher TEACHER --out plan.json
    glmf-teacher-kl.py --url URL --plan plan.json --dump-root DUMP_ROOT --path prefill --out rows-bf16
    klgate.py score --teacher TEACHER --engine rows-bf16 --json bf16-prefill.json
    (the same for the baseline arm, then)
    klgate.py compare bf16-prefill.json f32-prefill.json --margin 0.002

For every window of the plan (token ids and the rows the teacher holds; format below),
``POST /v1/bench/probe`` runs the window's ids cold, without drafts, with ``score_from`` 1 and
``score_positions`` the plan's rows, and the engine streams each of those rows' full
log-probabilities into a new leaf under its ``CUTEAFD_PROBE_DUMP_ROOT`` (``PROBE_DUMP_ROOT`` in
the launcher configuration). This script turns each window's rows into
``<out>/<window_id>.safetensors`` and removes the leaf.

Paths (``--path``): ``decode`` scores in verify steps of ``--verify-rows`` rows (the decode
kernels); ``prefill`` in the prompt chunks a prefill runs (GLM 5.3 Flash: the chunked KDA
recurrence, prefill GEMMs and lanes). Prefill-path scoring requires the server to advertise
``prefill_scoring="streamed"``; without that capability the script stops before the first window.

Plan (JSON): ``{"vocab": V, "windows": [{"window_id", "tokens": [ids], "tokens_sha256",
"positions": [rows]}]}`` where row ``r`` is the distribution after ``tokens[0..r]`` (inclusive),
predicting ``tokens[r + 1]``; ``tokens_sha256`` is the sha256 of the ids as little-endian u32.

Output per window (safetensors): ``positions`` I32 [k] (the plan's rows, ascending) and
``logits`` F32 [k, V]: the engine's log-probabilities, which a log-softmax leaves as they are,
so a KL scorer that log-softmaxes logits reads them unchanged. Metadata: ``window_id``,
``tokens_sha256``, ``plan_sha256``, ``score_path`` and ``engine`` (the server's identity and
settings, JSON). ``run.json`` records the run; a rerun into the same directory skips finished
windows. Standard library only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import sys
import time
import urllib.error
import urllib.request
from array import array


def tokens_sha256(tokens: list[int]) -> str:
    """sha256 of the ids as little-endian u32 (the plan's ``tokens_sha256``)."""
    return hashlib.sha256(struct.pack(f"<{len(tokens)}I", *tokens)).hexdigest()


def probe_spec(window: dict, path: str, verify_rows: int, leaf: str) -> dict:
    """The probe of one window: the plan's row r is the probe position r + 1 (the token it predicts)."""
    spec = {"prompt_ids": window["tokens"], "cold": True, "no_speculation": True, "score_from": 1,
            "score_path": path, "score_positions": [r + 1 for r in window["positions"]], "dump_rows": leaf,
            "top_k": 1}
    if path == "decode":
        spec["verify_rows"] = verify_rows
    return spec


def read_safetensors(path: str) -> tuple[dict, dict, bytes]:
    """(metadata, tensors' header entries, data) of a safetensors file."""
    with open(path, "rb") as f:
        size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(size))
        data = f.read()
    return header.pop("__metadata__", {}) or {}, header, data


def write_safetensors(path: str, tensors: list[tuple[str, str, list[int], bytes]], meta: dict[str, str]) -> None:
    """Writes ``tensors`` (name, dtype, shape, little-endian bytes) with string metadata, atomically."""
    header: dict[str, object] = {"__metadata__": meta}
    offset = 0
    for name, dtype, shape, data in tensors:
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + len(data)]}
        offset += len(data)
    text = json.dumps(header, separators=(",", ":")).encode()
    text += b" " * (-(8 + len(text)) % 8)
    partial = path + ".partial"
    with open(partial, "wb") as f:
        f.write(struct.pack("<Q", len(text)) + text)
        for _, _, _, data in tensors:
            f.write(data)
    os.replace(partial, path)


def collect(dump: str, rows: list[int], vocab: int) -> bytes:
    """The dumped log-probabilities of probe positions ``r + 1`` for ``rows``, in that order, F32 LE."""
    entries = {}
    with open(os.path.join(dump, "manifest.jsonl")) as f:
        for line in f:
            entry = json.loads(line)
            if entry.get("tensor") != "log_probs" or entry.get("dtype") != "F32" or entry.get("byte_order") != "little":
                raise ValueError(f"{dump}: unexpected row {entry}")
            if entry["vocab_size"] != vocab:
                raise ValueError(f"{dump}: vocabulary {entry['vocab_size']}, the plan's {vocab}")
            entries[entry["position"]] = entry["file"]
    missing = [r for r in rows if r + 1 not in entries]
    if missing:
        raise ValueError(f"{dump}: no row for {len(missing)} plan rows, e.g. {missing[:5]}")
    out = bytearray()
    for r in rows:
        _, header, data = read_safetensors(os.path.join(dump, entries[r + 1]))
        tensor = header["log_probs"]
        begin, end = tensor["data_offsets"]
        if tensor["dtype"] != "F32" or tensor["shape"] != [vocab] or end - begin != 4 * vocab:
            raise ValueError(f"{dump}: row {r} is not F32 [{vocab}]")
        out += data[begin:end]
    return bytes(out)


def request(url: str, payload: dict | None, timeout: float, api_key: str | None) -> dict:
    headers = {"content-type": "application/json"}
    if api_key:
        headers["authorization"] = f"Bearer {api_key}"
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, method="GET" if payload is None else "POST", headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"HTTP {error.code}: {error.read()[:400]!r}") from None


def score_window(args, window: dict, vocab: int, plan_sha256: str, leaf: str) -> dict:
    """Scores one window into ``<out>/<window_id>.safetensors``; returns its run record."""
    rows = sorted(window["positions"])
    if tokens_sha256(window["tokens"]) != window["tokens_sha256"]:
        raise ValueError(f"{window['window_id']}: tokens_sha256 does not match its ids")
    if not rows or rows[0] < 0 or rows[-1] + 1 >= len(window["tokens"]):
        raise ValueError(f"{window['window_id']}: rows must lie in [0, {len(window['tokens']) - 1})")
    dump = os.path.join(args.dump_root, leaf)
    started = time.time()
    body = {"messages": [{"role": "user", "content": "score"}], "max_tokens": 1}
    reply = request(f"{args.url}/v1/bench/probe",
                    {"body": body, "spec": probe_spec(window, args.path, args.verify_rows, leaf)},
                    args.timeout, args.api_key)
    record = reply.get("probe") or {}
    if record.get("error"):
        raise RuntimeError(f"{window['window_id']}: probe error {record['error']}")
    if record.get("score_path") != args.path or record.get("scored") != len(rows):
        raise RuntimeError(f"{window['window_id']}: the engine scored {record.get('scored')} rows on the "
                           f"{record.get('score_path')} path, the plan asks {len(rows)} on {args.path}")
    scored = time.time()
    logits = collect(dump, rows, vocab)
    meta = {"window_id": window["window_id"], "tokens_sha256": window["tokens_sha256"], "plan_sha256": plan_sha256,
            "score_path": args.path, "engine": json.dumps(reply.get("server", {}), sort_keys=True)}
    if args.path == "decode":
        meta["verify_rows"] = str(args.verify_rows)
    positions = array("i", rows)
    if sys.byteorder != "little":
        positions.byteswap()
    write_safetensors(os.path.join(args.out, f"{window['window_id']}.safetensors"),
                      [("positions", "I32", [len(rows)], positions.tobytes()),
                       ("logits", "F32", [len(rows), vocab], logits)], meta)
    if not args.keep_dumps:
        shutil.rmtree(dump)
    return {"window_id": window["window_id"], "tokens": len(window["tokens"]), "rows": len(rows),
            "score_s": round(scored - started, 3), "write_s": round(time.time() - scored, 3), "server": reply.get("server")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", required=True, help="the served engine, e.g. http://localhost:8000")
    parser.add_argument("--plan", required=True, help="klgate.py plan's output")
    parser.add_argument("--dump-root", required=True, help="the server's CUTEAFD_PROBE_DUMP_ROOT, as seen here")
    parser.add_argument("--out", required=True, help="new or partly written directory of <window>.safetensors")
    parser.add_argument("--path", choices=("decode", "prefill"), default="decode")
    parser.add_argument("--verify-rows", type=int, default=64, help="decode path: rows per verify step")
    parser.add_argument("--windows", default="", help="comma-separated window ids (default every window)")
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--api-key", default=os.environ.get("CUTEAFD_API_KEY"),
                        help="the server's API key, off the local network (default $CUTEAFD_API_KEY)")
    parser.add_argument("--keep-dumps", action="store_true")
    args = parser.parse_args()
    args.url = args.url.rstrip("/")
    with open(args.plan, "rb") as f:
        raw = f.read()
    plan, plan_sha256 = json.loads(raw), hashlib.sha256(raw).hexdigest()
    vocab = int(plan["vocab"])
    wanted = set(filter(None, args.windows.split(",")))
    windows = [w for w in plan["windows"] if not wanted or w["window_id"] in wanted]
    if wanted - {w["window_id"] for w in windows}:
        raise SystemExit(f"windows not in the plan: {sorted(wanted - {w['window_id'] for w in windows})}")
    if args.path == "prefill":
        status = request(f"{args.url}/v1/bench/status", None, args.timeout, args.api_key)
        if status.get("prefill_scoring") != "streamed":
            raise SystemExit("prefill-path scoring requires server capability prefill_scoring=streamed")
    os.makedirs(args.out, exist_ok=True)
    run_path = os.path.join(args.out, "run.json")
    run = {"plan": args.plan, "plan_sha256": plan_sha256, "url": args.url, "score_path": args.path,
           "verify_rows": args.verify_rows if args.path == "decode" else None, "windows": [], "complete": False}
    if os.path.exists(run_path):
        with open(run_path) as f:
            previous = json.load(f)
        if previous.get("plan_sha256") != plan_sha256 or previous.get("score_path") != args.path:
            raise SystemExit(f"{args.out} holds another plan's or path's run")
        run["windows"] = previous["windows"]
    done = {w["window_id"] for w in run["windows"]}
    started = time.time()
    for index, window in enumerate(windows):
        if window["window_id"] in done and os.path.exists(os.path.join(args.out, f"{window['window_id']}.safetensors")):
            continue
        leaf = f"{os.path.basename(os.path.abspath(args.out))}-{window['window_id']}-{os.getpid()}"
        entry = score_window(args, window, vocab, plan_sha256, leaf)
        run["windows"].append(entry)
        run["server"] = entry.pop("server")
        with open(run_path + ".partial", "w") as f:
            json.dump(run, f, indent=1)
        os.replace(run_path + ".partial", run_path)
        print(f"[{index + 1}/{len(windows)}] {window['window_id']}: {entry['rows']} rows, scored in "
              f"{entry['score_s']:.2f} s, written in {entry['write_s']:.2f} s ({time.time() - started:.0f} s)", flush=True)
    run["complete"] = True
    with open(run_path + ".partial", "w") as f:
        json.dump(run, f, indent=1)
    os.replace(run_path + ".partial", run_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
