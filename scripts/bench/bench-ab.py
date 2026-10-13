#!/usr/bin/env python3
"""Interleaved A/B serving comparison between engine checkouts.

Each arm is a checkout with its own run.sh/stop.sh (for example ds41rt v15 and
this repo). Sessions run in ABBA order per RTX layout; every session stops all
arms, relaunches its arm with --restart, discards a warmup battery, then
measures a greedy decode battery and the code concurrency sweep (distinct
per-request nonces by default; --identical-prompts opts out). Results land
under runs/ab/LABEL/ and the summary prints per-arm medians and B/A ratios.

  scripts/bench/bench-ab.py --label p0-parity \
      --arm ds41rt=/home/tj/Developer/ds41rt --arm cuteafd=/home/tj/Developer/cuteafd \
      --layouts 1 2 --sessions 4

--battery agentic replays a recording of scripts/bench/bench-agentic-session.py instead (identical
prompts in every session; C1 and C4 by default), e.g. prefix cache on vs off on MiMo:

  scripts/bench/bench-ab.py --label mimo-prefix --battery agentic --agentic-recording REC \
      --launch 'scripts/launch/run-family.sh --restart' --layouts 1 --sessions 4 \
      --arm off=CHECKOUT --arm on=CHECKOUT \
      --arm-arg 'off=--config mimo-off.config' --arm-arg 'on=--config mimo-on.config'
"""
from __future__ import annotations

import argparse
import json
import math
import shlex
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TOKENIZER_DEFAULT = (
    "/mnt/sparknest/hf-home/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/"
    "dba1be0a40aa45a94ad051997016db3960a90277/tokenizer.json"
)


def run(cmd: list[str], cwd: Path | None = None, log: Path | None = None, timeout: int = 3600) -> None:
    print(f"$ {' '.join(cmd)}", flush=True)
    with open(log, "a") if log else subprocess.DEVNULL as sink:  # type: ignore[arg-type]
        result = subprocess.run(cmd, cwd=cwd, stdout=sink, stderr=subprocess.STDOUT, timeout=timeout)
    if result.returncode:
        raise SystemExit(f"command failed ({result.returncode}): {' '.join(cmd)}; see {log}")


def stop_all(arms: dict[str, Path], log: Path) -> None:
    for path in arms.values():
        subprocess.run(["./stop.sh"], cwd=path, stdout=open(log, "a"), stderr=subprocess.STDOUT)


def session(arm: str, path: Path, arms: dict[str, Path], rtx: int, out: Path, args) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    log = out / "session.log"
    stop_all(arms, log)
    started = time.monotonic()
    extra = [*args.run_arg, *[w for a in args.arm_arg if a.split("=", 1)[0] == arm for w in shlex.split(a.split("=", 1)[1])]]
    run([*shlex.split(args.launch.format(rtx=rtx)), *extra], cwd=path, log=log, timeout=1800)
    ready_s = time.monotonic() - started
    py = str(REPO / ".venv/bin/python")
    if args.battery == "agentic":
        return agentic_session(arm, rtx, ready_s, out, log, py, args)
    decode = [py, str(REPO / "scripts/bench/deepseek_v41/bench-release-decode.py"), "--base-url", args.base_url,
              "--tokenizer", args.tokenizer, "--nonce-seed", str(args.nonce_seed)]
    run([*decode, "--label", f"{arm}-warmup", "--output", str(out / "warmup.json")], log=log)
    run([*decode, "--label", arm, "--repeats", str(args.repeats), "--output", str(out / "decode.json")], log=log)
    run([py, str(REPO / "scripts/bench/deepseek_v41/bench-concurrent-api.py"), "--base-url", args.base_url,
         "--case", "code", "--concurrency", *map(str, args.concurrency), "--repeats", "1",
         "--nonce", f"ab-{args.nonce_seed}", "--prompt-label", "ab", "--label", arm,
         *([] if args.identical_prompts else ["--distinct-prompts"]),
         "--output", str(out / "concurrent.json")], log=log)
    decode_report = json.loads((out / "decode.json").read_text())
    concurrent_report = json.loads((out / "concurrent.json").read_text())
    return {
        "arm": arm,
        "rtx": rtx,
        "ready_s": round(ready_s, 1),
        "weighted_decode_tps": decode_report["median_weighted_observed_decode_tokens_per_second"],
        **{f"code_c{s['concurrency']}_tps": s["median_aggregate_tps"] for s in concurrent_report["summaries"]},
    }


AGENTIC_KEYS = ("ttft_first_s", "ttft_later_s", "hit_ratio_later", "prefill_tps_later", "decode_tps",
                "full_turn_reused", "prompt_reused", "tool_call_validity", "session_wall_s")


def agentic_session(arm: str, rtx: int, ready_s: float, out: Path, log: Path, py: str, args) -> dict:
    """Replay the recording (warm-up pass discarded, then --repeats measured passes)."""
    bench = [py, str(REPO / "scripts/bench/bench-agentic-session.py"), "replay", "--recording", str(args.agentic_recording),
             "--base-url", args.base_url, "--concurrency", *map(str, args.concurrency), *args.agentic_arg]
    # The warm-up only warms the path (connections, workspaces): 64 tokens per turn.
    run([*bench, "--label", f"ab-{args.nonce_seed}-warmup", "--concurrency", "1", "--repeats", "1",
         "--max-tokens", "64", "--output", str(out / "agentic-warmup.json")], log=log, timeout=6 * 3600)
    run([*bench, "--label", f"ab-{args.nonce_seed}", "--repeats", str(args.repeats),
         "--output", str(out / "agentic.json")], log=log, timeout=24 * 3600)
    summary = json.loads((out / "agentic.json").read_text())["summary"]
    row = {"arm": arm, "rtx": rtx, "ready_s": round(ready_s, 1)}
    for level, values in summary.items():
        for key in AGENTIC_KEYS:
            row[f"{level}_{key}"] = values.get(key)
    return row


def summarize(rows: list[dict], arms: list[str]) -> str:
    keys = [k for k in rows[0] if k.endswith("_tps") or k == "ready_s" or k.endswith(AGENTIC_KEYS)]
    lines = []
    for rtx in sorted({r["rtx"] for r in rows}):
        lines.append(f"\n{rtx} RTX")
        lines.append(f"{'metric':<30}" + "".join(f"{a:>12}" for a in arms) + f"{'B/A':>9}")
        for key in keys:
            medians = []
            for arm in arms:
                values = [r[key] for r in rows if r["arm"] == arm and r["rtx"] == rtx and r.get(key) is not None]
                medians.append(statistics.median(values) if values else math.nan)
            ratio = medians[1] / medians[0] if len(medians) == 2 and medians[0] else math.nan
            lines.append(f"{key:<30}" + "".join(f"{m:>12.3f}" if abs(m) < 10 else f"{m:>12.1f}" for m in medians)
                         + f"{ratio:>9.3f}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label", required=True)
    parser.add_argument("--arm", action="append", required=True, help="NAME=CHECKOUT (exactly two)")
    parser.add_argument("--layouts", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--sessions", type=int, default=4, help="per layout, ABBA order")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 16])
    parser.add_argument("--identical-prompts", action="store_true",
                        help="decode battery only: opt out of distinct per-request nonces in the concurrency sweep")
    parser.add_argument("--nonce-seed", type=int, default=20260929)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--tokenizer", default=TOKENIZER_DEFAULT)
    parser.add_argument("--run-arg", action="append", default=[], help="extra run.sh argument (repeatable)")
    parser.add_argument("--arm-arg", action="append", default=[],
                        help="NAME=ARG: launcher argument for one arm only, shell-split, e.g. 'prune=--config FILE' (repeatable)")
    parser.add_argument("--launch", default="./run.sh --set RTX_GPUS={rtx} --restart",
                        help="launcher run in each arm's checkout ({rtx} is the layout), e.g. 'scripts/launch/run-family.sh --restart'; checkouts before #82 take './run.sh --rtx-gpus {rtx} --restart'")
    parser.add_argument("--battery", choices=["decode", "agentic"], default="decode")
    parser.add_argument("--agentic-recording", type=Path, help="bench-agentic-session.py record output (--battery agentic)")
    parser.add_argument("--agentic-arg", action="append", default=[],
                        help="extra bench-agentic-session.py replay argument (repeatable), e.g. --model=ID")
    args = parser.parse_args()
    if args.battery == "agentic":
        if args.agentic_recording is None:
            raise SystemExit("--battery agentic needs --agentic-recording")
        args.agentic_recording = args.agentic_recording.resolve()
        if args.concurrency == [1, 4, 16]:
            args.concurrency = [1, 4]
    arms = dict(item.split("=", 1) for item in args.arm)
    arms = {name: Path(path).resolve() for name, path in arms.items()}
    if len(arms) != 2:
        raise SystemExit("exactly two --arm values are required")
    names = list(arms)
    root = REPO / "runs" / "ab" / args.label
    root.mkdir(parents=True, exist_ok=True)
    order = [names[0], names[1], names[1], names[0]]
    rows: list[dict] = []
    results = root / "sessions.jsonl"
    for rtx in args.layouts:
        for index in range(args.sessions):
            arm = order[index % 4]
            row = session(arm, arms[arm], arms, rtx, root / f"rtx{rtx}-s{index}-{arm}", args)
            rows.append(row)
            with results.open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
    stop_all(arms, root / "stop.log")
    table = summarize(rows, names)
    (root / "summary.txt").write_text(table + "\n")
    print(table)


if __name__ == "__main__":
    sys.exit(main())
