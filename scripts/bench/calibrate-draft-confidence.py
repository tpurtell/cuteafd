#!/usr/bin/env python3
"""Replay fixed-K7 requests, then fit the shared six-feature selector prior.

Launch the desired drafter with SPECULATION_TRACE=/abs/trace.jsonl,
COPY_DRAFTS=off, COPY_DRAFT_POLICY=off and SPECULATOR_DRAFTS=7.
Leave DRAFT_CONFIDENCE=off for a capture independent of existing fits.
This tool never launches a server:
--replay-url optionally sends 32 fixed, deterministic code requests to one
already running; --trace fits recorded outcomes without hardware. Output is
--output-dir/draft-confidence.<numerics>.json, beside the drafter config.
Target quant is training provenance, never part of --key.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import urllib.request

import numpy as np

FEATURE_ORDER = ["history_logit", "log1p_margin", "candidate_probability_logit",
                 "candidate_entropy", "log1p_unary_rank", "position_over_7"]


def transform(prior, position, selector):
    margin, probability, entropy, rank = selector
    if not (np.isfinite([prior, *selector]).all() and 0 <= prior <= 1 and 1 <= position <= 7
            and margin >= 0 and 0 < probability <= 1 and -1e-5 <= entropy <= 2.80
            and 0 <= rank < 16 and rank == int(rank)):
        raise ValueError("invalid selector features")

    def logit(p):
        p = np.clip(p, 1e-4, 1 - 1e-4)
        return np.log(p / (1 - p))

    return [logit(prior), np.log1p(margin), logit(probability), entropy,
            np.log1p(rank), position / 7]


def observations(paths, key):
    x, y, requests = [], [], []
    for path in paths:
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row.get("type") != "draft_confidence":
                continue
            if row["key"] != key:
                raise ValueError(f"trace key {row['key']} differs from {key}")
            proposed, accepted, observed = row["proposed"], row["accepted"], row["observed"]
            if not (all(type(value) is int for value in (proposed, accepted, observed))
                    and 0 <= accepted <= proposed <= 7
                    and accepted <= observed <= min(proposed, accepted + 1)):
                raise ValueError("invalid or uncensored verify outcome")
            # Fixed-K7 only. Ignore the last cycle truncated by the output/context budget.
            if proposed != 7:
                continue
            if min(len(row["history"]), len(row["features"])) < observed:
                raise ValueError("trace misses observed selector features")
            for i in range(observed):
                x.append(transform(row["history"][i], i + 1, row["features"][i]))
                y.append(float(i < accepted))
                requests.append((str(path), row["request"]))
    if not x or len(set(y)) != 2:
        raise ValueError("fit needs verified fixed-K7 positions with both acceptance outcomes")
    return np.asarray(x), np.asarray(y), requests


def probability(design, beta):
    return 1 / (1 + np.exp(-np.clip(design @ beta, -40, 40)))


def logloss(p, y):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log1p(-p)))


def fit(x, y, regularization=10.0):
    if regularization <= 0 or not np.isfinite(regularization):
        raise ValueError("regularization must be finite and positive")
    mean, scale = x.mean(axis=0), np.maximum(x.std(axis=0), 1e-6)
    design = np.column_stack([np.ones(len(x)), (x - mean) / scale])
    beta = np.zeros(7)
    penalty = np.diag([0.0] + [regularization] * 6)

    def objective(b):
        z = design @ b
        return float(np.sum(np.logaddexp(0, z) - y * z) + 0.5 * b @ penalty @ b)

    for iteration in range(100):
        p = probability(design, beta)
        gradient = design.T @ (p - y) + penalty @ beta
        hessian = design.T @ (design * (p * (1 - p))[:, None]) + penalty
        step = np.linalg.solve(hessian, gradient)
        fraction, before = 1.0, objective(beta)
        while objective(beta - fraction * step) > before and fraction > 1e-6:
            fraction *= 0.5
        beta -= fraction * step
        if np.max(np.abs(fraction * step)) < 1e-8:
            break
    return dict(mean=mean.tolist(), scale=scale.tolist(), intercept=float(beta[0]),
                coefficients=beta[1:].tolist()), logloss(probability(design, beta), y), iteration + 1


def replay(url, count, max_tokens, requests_file=None):
    if count < 1 or max_tokens < 8:
        raise ValueError("replay needs positive requests and at least 8 output tokens")
    headers = {"Content-Type": "application/json"}
    if os.environ.get("CUTEAFD_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["CUTEAFD_API_KEY"]
    with urllib.request.urlopen(urllib.request.Request(url.rstrip("/") + "/v1/models", headers=headers), timeout=30) as response:
        model = json.load(response)["data"][0]["id"]
    tasks = ["merge sorted intervals", "implement an LRU cache", "parse a CSV row", "binary search",
             "topological sort", "validate a JSON schema", "copy-on-write pages", "retry with backoff"]
    payloads = ([json.loads(line) for line in requests_file.read_text().splitlines()] if requests_file else
                [{"messages": [{"role": "user", "content": f"Implement {tasks[i % len(tasks)]} in Python. "
                               f"Use type hints and {2 + i // len(tasks)} assert examples. Explain corner cases. "
                               "Return one code block."}]} for i in range(count)])
    if len(payloads) != count:
        raise ValueError(f"replay corpus has {len(payloads)} requests, expected {count}")
    for i, payload in enumerate(payloads):
        payload.update(model=model, temperature=0, max_tokens=max_tokens, stream=False)
        request = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions",
                                         data=json.dumps(payload).encode(), headers=headers)
        with urllib.request.urlopen(request, timeout=600) as response:
            result = json.load(response)
        if not result.get("choices"):
            raise ValueError(f"request {i} returned no choices")
        print(f"replayed {i + 1}/{count}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key", required=True)
    parser.add_argument("--trace", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--replay-url")
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument("--requests-file", type=Path)
    parser.add_argument("--max-tokens", type=int, default=320)
    parser.add_argument("--regularization", type=float, default=10.0)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_-]+/[A-Za-z0-9_-]+", args.key):
        parser.error("key must be family/drafter/numerics")
    if args.replay_url:
        replay(args.replay_url, args.requests, args.max_tokens, args.requests_file)
    x, y, requests = observations(args.trace, args.key)
    model, loss, iterations = fit(x, y, args.regularization)
    document = dict(schema="cuteafd-draft-confidence-v1", key=args.key, feature_order=FEATURE_ORDER, **model,
                    training=dict(requests=len(set(requests)), observed_positions=len(y), logloss=loss,
                                  iterations=iterations, regularization=args.regularization,
                                  traces=[dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in args.trace]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / f"draft-confidence.{args.key.split('/')[2]}.json"
    path.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(output=str(path), **document["training"]), indent=2))


if __name__ == "__main__":
    main()
