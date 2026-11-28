#!/usr/bin/env python3
"""Greedy byte-exactness tooling.

  greedy-exactness.py client FIXTURE   wip-cards --client hook: sends the
      lossless fixture at temperature 0 at each concurrency level and writes
      greedy.json to $CUTEAFD_CLIENT_OUT.
  greedy-exactness.py compare A.json B.json   per (level, id) byte equality of
      content and reasoning_content; exits 1 on any mismatch.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import urllib.request

FIELDS = ("content", "reasoning_content")


def read_api_key() -> str:
    path = os.environ.get("CUTEAFD_API_KEY_FILE")
    if path and os.path.exists(path):
        with open(path) as f:
            return f.read().strip()
    return os.environ.get("CUTEAFD_API_KEY", "")


def get_json(url: str, key: str) -> dict:
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode())


def complete_one(base_url: str, key: str, model: str, case: dict, level: int, max_tokens: int = 128) -> dict:
    prompt = f"[{case['nonce']}:c{level}] {case['prompt']}"
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": max_tokens,
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read().decode())
    choice = data["choices"][0]
    msg = choice["message"]
    return {
        "level": level,
        "id": case["id"],
        "finish_reason": choice.get("finish_reason"),
        "content": msg.get("content") or "",
        "reasoning_content": msg.get("reasoning_content") or "",
        "completion_tokens": data.get("usage", {}).get("completion_tokens", 0),
    }


def parse_levels(text: str) -> list[int]:
    levels = [int(p) for p in text.split(",") if p.strip()]
    if not levels or any(n < 1 for n in levels):
        raise argparse.ArgumentTypeError(f"bad --levels {text!r}")
    return levels


def run_client(args: argparse.Namespace) -> int:
    base_url = os.environ["CUTEAFD_SMOKE_URL"].rstrip("/")
    key = read_api_key()
    out_dir = os.environ.get("CUTEAFD_CLIENT_OUT", ".")
    with open(args.fixture) as f:
        cases = json.load(f)
    model = get_json(f"{base_url}/v1/models", key)["data"][0]["id"]
    stats_before = get_json(f"{base_url}/v1/stats", key)
    results = []
    for level in args.levels:
        with concurrent.futures.ThreadPoolExecutor(max_workers=level) as pool:
            futs = [pool.submit(complete_one, base_url, key, model, c, level, args.max_tokens) for c in cases]
            results.extend(f.result() for f in futs)
    stats_after = get_json(f"{base_url}/v1/stats", key)
    payload = {
        "model": model,
        "cases": results,
        "total_tokens": sum(r["completion_tokens"] for r in results),
        "stats_before": stats_before,
        "stats_after": stats_after,
    }
    out_path = os.path.join(out_dir, "greedy.json")
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"greedy_client: {len(results)} responses, {payload['total_tokens']} tokens -> {out_path}")
    return 0


def first_divergence(a: str, b: str) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def load_cases(path: str) -> dict[tuple[int, str], dict]:
    with open(path) as f:
        data = json.load(f)
    return {(c["level"], c["id"]): c for c in data["cases"]}


def same(x: dict, y: dict) -> bool:
    return all(x.get(k, "") == y.get(k, "") for k in FIELDS)


def run_compare(args: argparse.Namespace) -> int:
    a, b = load_cases(args.a), load_cases(args.b)
    keys = sorted(set(a) | set(b))
    ids = sorted({k[1] for k in keys})
    tally: dict[int, list[int]] = {}
    bad = 0
    for level, cid in keys:
        t = tally.setdefault(level, [0, 0])
        t[1] += 1
        if (level, cid) not in a or (level, cid) not in b:
            bad += 1
            print(f"MISSING level={level} id={cid} in {'A' if (level, cid) not in a else 'B'}")
            continue
        x, y = a[(level, cid)], b[(level, cid)]
        if same(x, y):
            t[0] += 1
            continue
        bad += 1
        for fld in FIELDS:
            xs, ys = x.get(fld, ""), y.get(fld, "")
            if xs == ys:
                continue
            i = first_divergence(xs, ys)
            lo = max(0, i - args.context)
            print(f"MISMATCH level={level} id={cid} field={fld} offset={i} (len A={len(xs)} B={len(ys)})")
            print(f"  A: {xs[lo:i + args.context]!r}")
            print(f"  B: {ys[lo:i + args.context]!r}")
        # does B's output at this level equal A's output at another level?
        matches = [lv for lv in sorted({k[0] for k in a}) if lv != level
                   and (lv, cid) in a and same(y, a[(lv, cid)])]
        print(f"  B@c{level} equals A at levels: {matches if matches else 'none'}")
    for level in sorted(tally):
        ok, n = tally[level]
        print(f"level c{level}: {ok}/{n} byte-exact")
    print(f"total: {sum(t[0] for t in tally.values())}/{sum(t[1] for t in tally.values())} ({len(ids)} prompts)")
    return 1 if bad else 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("client", help="send the fixture at temperature 0")
    c.add_argument("fixture")
    c.add_argument("--levels", type=parse_levels, default=[1, 4, 10])
    c.add_argument("--max-tokens", type=int, default=128)
    c.set_defaults(fn=run_client)
    d = sub.add_parser("compare", help="diff two greedy.json files")
    d.add_argument("a")
    d.add_argument("b")
    d.add_argument("--context", type=int, default=40, help="characters of context around a divergence")
    d.set_defaults(fn=run_compare)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
