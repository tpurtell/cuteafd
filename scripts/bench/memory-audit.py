#!/usr/bin/env python3
"""Tabulate `cuteafd::memory` ledger reports from coordinator or expertd logs.

Usage: memory-audit.py LOG [LOG ...] [--last | --sequence N] [--json]

Each log line written by `shared::memory_report` carries one JSON report:
per device the runtime's used bytes, the ledger's tracked bytes by category,
the untracked rest (CUDA context, modules, cuBLAS, graph executables),
weights by tensor stem and resident format, and dual-format tensors.
"""
import argparse
import json
import re
import sys
from collections import defaultdict

ANSI = re.compile(r"\x1b\[[0-9;]*m")
GIB = float(1 << 30)

# Tensor stem -> weight group (coarse, family-agnostic).
GROUPS = [
    (r"embed_tokens|\bembed\b|wte", "embedding"),
    (r"lm_head|\bhead\b", "lm_head"),
    (r"indexer|\.index|wk\b|weights_proj|w_ik|w_iq", "indexer"),
    (r"shared_expert|shared_experts", "shared_expert"),
    (r"\.mlp\.experts|\.experts\.", "routed_expert"),
    (r"\.mlp\.gate\.|router|e_score_correction|tid2eid", "router"),
    (r"mtp|nextn|eh_proj|enorm|hnorm|dspark", "speculator"),
    (r"norm", "norm"),
    (r"hc_|\.hc\.", "hyper_connection"),
    (r"self_attn|\.attn\.|attention|kv_b|q_a|q_b|kv_a|o_proj|qkv|wo_|wq|wkv|linear_attn|kda|compressor", "attention"),
    (r"\.mlp\.|ffn|gate_up|down_proj|up_proj|gate_proj", "dense_ffn"),
]


def category(scope):
    """Planner category of a ledger scope (mirrors cuteafd_core::memory_layout::Category::of_scope)."""
    root = scope.split("/")[0]
    simple = {"weights": "weights", "embedding": "embedding", "experts": "experts", "local-experts": "experts",
              "drafter": "drafter", "kv": "kv", "prefix": "prefix", "workspace": "workspace", "sampler": "workspace",
              "spark-intake": "workspace", "probe": "workspace", "transport": "transport", "peer-split": "transport",
              "staging": "staging", "mapped-table": "tables", "ple": "tables"}
    if root in simple:
        return simple[root]
    if root == "v41":
        stage = scope[4:]
        low = stage.lower()
        if "draft" in low:
            return "drafter"
        if "engram" in low:
            return "tables"
        if "weights" in low:
            return "weights"
        if low.startswith("kv") or "kv cache" in low:
            return "kv"
        if "prefix" in low or "snapshot" in low:
            return "prefix"
        if "expert" in low:
            return "experts"
        if "transport" in low or "tp2" in low:
            return "transport"
        if "workspace" in low or "lane" in low or "vision" in low:
            return "workspace"
    return "runtime"


def group(stem):
    for pattern, name in GROUPS:
        if stem and re.search(pattern, stem):
            return name
    return "other"


def reports(path):
    with open(path, errors="replace") as handle:
        for line in handle:
            if "memory ledger" not in line:
                continue
            line = ANSI.sub("", line)
            start = line.find("report=")
            if start < 0:
                continue
            text = line[start + len("report="):]
            try:
                value, _ = json.JSONDecoder().raw_decode(text)
            except json.JSONDecodeError:
                continue
            seq = re.search(r"sequence=(\d+)", line)
            value["_sequence"] = int(seq.group(1)) if seq else None
            yield value


def gib(n):
    return f"{n / GIB:8.2f}"


def summarize(report):
    out = {"stage": report.get("stage"), "devices": [], "pinned": report.get("pinned", {})}
    weights = defaultdict(int)
    for space, device, scope, tensor, fmt, nbytes, _count in report.get("weights", []):
        weights[(space, device, group(tensor), fmt or "native")] += nbytes
    for dev in report.get("devices", []):
        entry = dict(dev)
        cats = defaultdict(int)
        for scope, nbytes in dev.get("scopes", {}).items():
            cats[category(scope)] += nbytes
        cats["runtime"] += max(dev.get("untracked", 0), 0)
        entry["categories"] = dict(cats)
        entry["weights"] = {f"{g}/{f}": b for (s, d, g, f), b in sorted(weights.items())
                            if s != "pinned" and d == dev["device"]}
        out["devices"].append(entry)
    out["dual_formats"] = report.get("dual_formats", [])
    out["host"] = report.get("host", {})
    return out


def render(path, summary):
    print(f"== {path} (stage {summary['stage']})")
    for dev in summary["devices"]:
        print(f"device {dev['device']}: used {gib(dev.get('used', 0))} GiB of {gib(dev.get('total', 0))}, "
              f"tracked {gib(dev['tracked'])}, untracked {gib(dev.get('untracked', 0))}, peak tracked {gib(dev['peak'])}")
        print("   by category: " + ", ".join(f"{c} {n / GIB:.2f}" for c, n in
                                             sorted(dev["categories"].items(), key=lambda kv: -kv[1])))
        for scope, nbytes in sorted(dev["scopes"].items(), key=lambda kv: -kv[1]):
            print(f"   {scope:<44} {gib(nbytes)}")
        if dev["weights"]:
            print("   weights by group/format:")
            for key, nbytes in sorted(dev["weights"].items(), key=lambda kv: -kv[1]):
                print(f"     {key:<42} {gib(nbytes)}")
    pinned = summary["pinned"]
    if pinned.get("tracked"):
        print(f"pinned host: {gib(pinned['tracked'])} GiB (peak {gib(pinned.get('peak', 0))})")
        for scope, nbytes in sorted(pinned.get("scopes", {}).items(), key=lambda kv: -kv[1]):
            print(f"   {scope:<44} {gib(nbytes)}")
    host = summary.get("host") or {}
    if host:
        print("host: " + ", ".join(f"{k} {gib(v).strip()}" for k, v in host.items()))
    for dual in summary["dual_formats"]:
        print(f"DUAL device {dual['device']}: {dual['tensor']} resident as {', '.join(dual['formats'])}")


GROWTH_GROUPS = {"graph growth"}
READY = re.compile(r"API is ready|is ready listen=")


def ready_reports(path):
    """(the at-ready report, last report) of one log. The at-ready report is the one serve logs at
    readiness (stage "ready", before any request captures lazy graphs or loads more code); older logs
    without it fall back to the first periodic report after the API is ready."""
    ready_seen, first, tagged, last = False, None, None, None
    with open(path, errors="replace") as handle:
        for line in handle:
            line = ANSI.sub("", line)
            if "memory ledger" not in line:
                if READY.search(line):
                    ready_seen = True
                continue
            start = line.find("report=")
            try:
                value, _ = json.JSONDecoder().raw_decode(line[start + len("report="):])
            except json.JSONDecodeError:
                continue
            if value.get("stage") == "ready" and tagged is None:
                tagged = value
                continue
            if ready_seen and first is None:
                first = value
            last = value
    return tagged or first, last


def ready_ledger(path):
    """Per ledger device: categories at ready (tracked from the last report; runtime from the at-ready
    report's untracked bytes, before lazy graph captures grow it)."""
    first, last = ready_reports(path)
    if last is None:
        return {}
    at_ready = {str(d["device"]): d for d in summarize(first or last)["devices"]}
    out = {}
    for dev in summarize(last)["devices"]:
        cats = dict(dev["categories"])
        cats["runtime"] = cats.get("runtime", 0) - max(dev.get("untracked", 0), 0) + max(
            at_ready.get(str(dev["device"]), dev).get("untracked", 0), 0)
        out[str(dev["device"])] = cats
    return out


def ready_gate(args):
    plan = json.load(open(args.compare))["memory_layout"] if args.compare else None
    if plan is None:
        print("--ready needs --compare PLAN", file=sys.stderr)
        return 2
    names = {("rtx" if d["kind"] == "rtx" else "spark") + str(d["index"]): d for d in plan["devices"]}
    mapping = dict(pair.split("=") for pair in args.device_map.split(","))
    worst = 0.0
    for path in args.logs:
        ledger = ready_ledger(path)
        for planned, ledger_id in mapping.items():
            if planned not in names or ledger_id not in ledger:
                continue
            predicted, growth = defaultdict(int), 0
            for item in names[planned]["items"]:
                if item["group"] in GROWTH_GROUPS:
                    growth += item["bytes"]
                else:
                    predicted[item["category"]] += item["bytes"]
            measured = ledger[ledger_id]
            total_p, total_m = sum(predicted.values()), sum(measured.values())
            diff = (total_p - total_m) / (1 << 20)
            worst = max(worst, abs(diff))
            print(f"== {path} {planned} (ledger device {ledger_id}) at ready: planned {total_p / (1 << 20):.0f} MiB, "
                  f"ledger {total_m / (1 << 20):.0f} MiB, diff {diff:+.0f} MiB (growth items {growth / (1 << 20):.0f} MiB)")
            for cat in sorted(set(predicted) | set(measured)):
                p_, m_ = predicted.get(cat, 0), measured.get(cat, 0)
                print(f"   {cat:<12} {p_ / (1 << 20):9.0f} {m_ / (1 << 20):9.0f} {(p_ - m_) / (1 << 20):+8.0f}")
    print(f"worst device difference {worst:.0f} MiB (tolerance {args.tolerance_mib:.0f})")
    return 0 if worst <= args.tolerance_mib else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+")
    parser.add_argument("--sequence", type=int)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--compare", help="`cuteafd plan --layout --json` output to compare against")
    parser.add_argument("--device-map", default="rtx0=0,rtx1=1",
                        help="planner device=ledger device pairs for the logs given (e.g. spark0=0)")
    parser.add_argument("--ready", action="store_true",
                        help="placement gate: the ledger at ready vs the plan without its growth items; exit 1 when a "
                             "device differs by more than --tolerance-mib. Tracked bytes come from the last report (fixed-"
                             "size buffers some families allocate on their first request, e.g. prefill workspaces, count: "
                             "admission must have reserved them); untracked (context, modules, graphs) from the first "
                             "report at or after the API is ready, before lazily captured graphs grow it")
    parser.add_argument("--tolerance-mib", type=float, default=64.0)
    args = parser.parse_args()
    if args.ready:
        sys.exit(ready_gate(args))
    results = {}
    for path in args.logs:
        found = list(reports(path))
        if args.sequence is not None:
            found = [r for r in found if r["_sequence"] == args.sequence]
        if not found:
            print(f"{path}: no memory ledger reports", file=sys.stderr)
            continue
        results[path] = summarize(found[-1])
    if args.compare:
        plan = json.load(open(args.compare))["memory_layout"]
        names = {}
        for device in plan["devices"]:
            prefix = "rtx" if device["kind"] == "rtx" else "spark"
            names[f"{prefix}{device['index']}"] = device
        mapping = dict(pair.split("=") for pair in args.device_map.split(","))
        for path, summary in results.items():
            ledger = {str(d["device"]): d for d in summary["devices"]}
            for planned, ledger_id in mapping.items():
                if planned not in names or ledger_id not in ledger:
                    continue
                predicted = defaultdict(int)
                for item in names[planned]["items"]:
                    predicted[item["category"]] += item["bytes"]
                measured = ledger[ledger_id]["categories"]
                print(f"== {path} {planned} (ledger device {ledger_id}): predicted vs measured GiB")
                total_p = total_m = 0
                for cat in sorted(set(predicted) | set(measured)):
                    p_, m_ = predicted.get(cat, 0), measured.get(cat, 0)
                    total_p += p_; total_m += m_
                    err = (p_ - m_) / m_ * 100 if m_ else float("nan")
                    print(f"   {cat:<12} {p_ / GIB:8.2f} {m_ / GIB:8.2f} {err:+7.1f}%")
                err = (total_p - total_m) / total_m * 100 if total_m else float("nan")
                print(f"   {'total':<12} {total_p / GIB:8.2f} {total_m / GIB:8.2f} {err:+7.1f}%")
        return
    if args.json:
        json.dump(results, sys.stdout, indent=1)
        print()
    else:
        for path, summary in results.items():
            render(path, summary)


if __name__ == "__main__":
    main()
