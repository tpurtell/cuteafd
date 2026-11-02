#!/usr/bin/env python3
"""Replay the shipped loop detector (rules R and F) over the local opencode
corpus: every assistant step the production replica served since 2026-10-22 22:00Z.

This is the long-generation control the review asked for: the fidelity windows
are capped at 1,024 generated tokens, so rule R can only fire on periods <= ~170
there and rule F (>= 385 identical tokens) cannot fire at all. The opencode
corpus has real 32k-token generations, so both rules are exercised.

Read-only on the opencode DB (`mode=ro`). Reuses the DB reader from
`loop-replay.py` and the detector port from `loop-abort-fidelity-harness.py`
(both beside this file).

Run:
  python3 -I rust/crates/cuteafd-daemon/tests/loop-abort-corpus-replay.py
"""
import collections
import importlib.util
import os
import statistics
import sys
import time

from tokenizers import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
REPLAY = os.environ.get("LOOP_REPLAY", os.path.join(HERE, "loop-replay.py"))
HARNESS = os.environ.get("LOOP_HARNESS", os.path.join(HERE, "loop-abort-fidelity-harness.py"))
CONTROL_SAMPLE = 500


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    replay = load(REPLAY, "loop_replay")
    harness = load(HARNESS, "loop_harness")
    db = replay.sqlite3.connect(replay.DB, uri=True)
    tk = Tokenizer.from_file(replay.TOK)
    provider = getattr(replay, "PROVIDER", None)
    q = ("select m.id, json_extract(m.data,'$.modelID'), json_extract(m.data,'$.agent') from message m "
         "where m.time_created>=? and json_extract(m.data,'$.role')='assistant' "
         + ("and json_extract(m.data,'$.providerID')=? " if provider else "")
         + "and json_extract(m.data,'$.modelID') in (?,?)")
    args = (replay.T0, *((provider,) if provider else ()), *replay.MODELS)
    msgs = db.execute(q, args).fetchall()
    rows = []
    for mid, model, agent in msgs:
        for si, s in enumerate(replay.steps_of(db, mid)):
            f = s["finish"]
            if f is None:
                continue
            rows.append({"mid": mid, "step": si, "model": model, "agent": agent, "pieces": s["pieces"],
                         "otok": (f.get("tokens") or {}).get("output"), "reason": f.get("reason"),
                         "chars": sum(len(t) for _, t in s["pieces"])})
    looped = [r for r in rows if r["otok"] and r["otok"] >= replay.CAP and r["reason"] == "length"]
    control = [r for r in rows if r not in looped and r["chars"] >= replay.MIN_CHARS]
    print(f"messages {len(msgs)} steps {len(rows)} reasons {dict(collections.Counter(r['reason'] for r in rows))}")
    print(f"capped(>= {replay.CAP} output tokens, length) {len(looped)}; control steps >= {replay.MIN_CHARS} chars "
          f"{len(control)}; shorter steps (cannot fire) {len(rows) - len(looped) - len(control)}")
    t0 = time.time()
    ids = {}
    for r in looped + control:
        ids[(r["mid"], r["step"])] = [t for _, tx in r["pieces"] for t in tk.encode(tx, add_special_tokens=False).ids]
    print(f"tokenized {len(ids)} steps in {time.time() - t0:.0f}s")

    def fire_at(step_ids, fast_span=384):
        guard = harness.RepeatGuard(512, 8, fast_span, 8)
        return harness.feed(guard, step_ids)

    # Rule R over the whole corpus (cheap: one 32-token hash per token).
    lr = [(r, f) for r in looped for f in [fire_at(ids[(r["mid"], r["step"])], 0)] if f]
    cr = [(r, f) for r in control for f in [fire_at(ids[(r["mid"], r["step"])], 0)] if f]
    ft = sorted(f[0] for _, f in lr)
    print(f"R(W=32,run=512,copies=8): capped fires {len(lr)}/{len(looped)} "
          f"at token p50 {ft[len(ft)//2]} p90 {ft[int(.9 * len(ft))]} max {ft[-1]} "
          f"(mean saved {statistics.mean(1 - t / 32000 for t in ft):.0%} of 32000) | "
          f"control fires {len(cr)}/{len(control)}: "
          + " ".join(f"{r['mid'][-6:]}({r['agent']},{r['otok']}tok,{r['reason']},fire@{f[0]})"
                     for r, f in cr))

    # Rule F is O(256) per token in this port, so run it over the capped loops
    # (the long generations that can reach it) and the longest control steps.
    sample = sorted(control, key=lambda r: r["chars"], reverse=True)[:CONTROL_SAMPLE]
    lf = [(r, f) for r in looped for f in [fire_at(ids[(r["mid"], r["step"])])] if f]
    cf = [(r, f) for r in sample for f in [fire_at(ids[(r["mid"], r["step"])])] if f]
    ft = sorted(f[0] for _, f in lf)
    print(f"R+F(W=32,run=512,copies=8; F span=384,copies=8): capped fires {len(lf)}/{len(looped)} "
          f"at token p50 {ft[len(ft)//2]} p90 {ft[int(.9 * len(ft))]} max {ft[-1]} "
          f"(mean saved {statistics.mean(1 - t / 32000 for t in ft):.0%} of 32000) | "
          f"control fires {len(cf)}/{len(sample)} (longest {CONTROL_SAMPLE}): "
          + " ".join(f"{r['mid'][-6:]}({r['agent']},{r['otok']}tok,{r['reason']},fire@{f[0]},{f[1]})"
                     for r, f in cf))
    return 0


if __name__ == "__main__":
    sys.exit(main())
