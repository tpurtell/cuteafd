#!/usr/bin/env python3
"""Coordinator GPU selection, shared by every family's launch.

COORDINATOR_GPUS is the one explicit selector: one or two host GPU indices, PCI
bus ids or GPU UUIDs, primary first. The primary is logical GPU 0 (entry, head
and V4.1's 3/2 ownership share), the second logical GPU 1. Unset selects the
lowest-index GPU, plus the next one when two are wanted.

  resolve  the ordered GPUs a launch uses (every family, and ./wip.sh)
  select   DeepSeek V4.1's one- or two-RTX choice; an explicit pair is checked,
           never replaced by the free-memory pick
  admit    after the deployment plan: each selected GPU holds its logical
           device's planned fixed owners (the KV pool sizes itself)
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation


MIB = 1 << 20
GIB = 1 << 30
GROUP_BYTES = 455_680
DEFAULT_POOL_TOKENS = 2 * 1_048_576

# Fixed owners include weights, execution workspaces, snapshots, CUDA contexts,
# and the constant (window/state) part of the C16 cache. These are conservative
# K7 measurements; K5 needs slightly less memory. Per-group storage follows the
# 3/2 compressed-source ownership split on logical GPUs 0/1. The fixed totals
# embed the default 20 native TP2 routed layers, so the format-aware model
# subtracts them and adds the checkpoint's own per-layer bytes.
DUAL_FIXED_WITH_HEADROOM = (93_078_948_736, 96_299_387_520)
DUAL_GROUP_BYTES = (270_336, 180_224)
# Native packed TP2 rank weights per routed layer; staging remains reserved.
DUAL_ROUTED_LAYER_BYTES = 3_609_722_880
DUAL_FIXED_EX_EXPERTS = (
    DUAL_FIXED_WITH_HEADROOM[0] - 20 * DUAL_ROUTED_LAYER_BYTES,
    DUAL_FIXED_WITH_HEADROOM[1] - 20 * DUAL_ROUTED_LAYER_BYTES,
)
# Per-TP2-rank routed-layer bytes by expert checkpoint format. The native
# value is the K7 measurement; EXL3 values come from the v7 K2 dual plan
# (69,122,129,920 B / 40 layers, ~2% slack) and the v5 K3.25 residency
# measurements. Attention, draft, vision and cache owners are format
# independent, so the fixed base is shared.
DUAL_LAYER_BYTES_BY_FORMAT = {
    "native": DUAL_ROUTED_LAYER_BYTES,
    # ModelOpt NVFP4 W4A4: 9,953,280 bytes per rank per expert (E2M1 payload
    # plus E4M3 K16 scales) x 384 experts, as measured in the startup plan.
    # Placing with the native constant instead over-fills the card.
    "nvfp4": 3_822_059_520,
}


class SelectionError(ValueError):
    pass


SELECTOR = re.compile(
    r"(?P<index>[0-9]+)"
    r"|(?P<uuid>GPU-[0-9A-Fa-f]{8}-(?:[0-9A-Fa-f]{4}-){3}[0-9A-Fa-f]{12})"
    r"|(?P<pci>[0-9A-Fa-f]{4,8}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.[0-7])",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Gpu:
    index: int
    uuid: str
    pci: str
    total_mib: int
    free_mib: int
    reclaim_mib: int = 0

    @property
    def effective_free_mib(self) -> int:
        return min(self.total_mib, self.free_mib + self.reclaim_mib)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nvidia-smi", default="nvidia-smi")
    commands = parser.add_subparsers(dest="command", required=True)
    resolve_parser = commands.add_parser("resolve", help="print INDEX UUID PCI per selected GPU, primary first")
    resolve_parser.add_argument("--selectors", default="", help="COORDINATOR_GPUS")
    resolve_parser.add_argument("--count", choices=("auto", "1", "2"), default="auto",
                                help="GPUs the launch asks for; an explicit list must name that many")
    select = commands.add_parser("select", help="DeepSeek V4.1's one- or two-RTX selection, as JSON")
    select.add_argument("--selectors", default="", help="COORDINATOR_GPUS")
    select.add_argument("--mode", choices=("auto", "1", "2"), required=True)
    select.add_argument("--concurrency", type=int, required=True)
    select.add_argument("--max-context-tokens", type=int, required=True)
    select.add_argument("--retained-turns", type=int, required=True)
    select.add_argument("--kv-pool-size", default="")
    select.add_argument("--memory-reservation", default="")
    select.add_argument("--compact-spark", "--compact-spark-tp2", dest="compact_spark",
                        action="store_true",
                        help="single-RTX compact Spark EXL3 layout (TP2 or TP3): "
                             "the ceiling is capped by physical VRAM; "
                             "--compact-spark-tp2 is the historical spelling")
    select.add_argument("--minimum-expert-layers", type=int, choices=range(1, 41), default=20)
    select.add_argument("--expert-format", choices=tuple(DUAL_LAYER_BYTES_BY_FORMAT), default="native")
    select.add_argument("--reclaim-pid", action="append", type=int, default=[])
    admit = commands.add_parser("admit", help="check the selected GPUs against a deployment plan")
    admit.add_argument("--uuids", required=True, help="selected GPU UUIDs, logical order")
    admit.add_argument("--deployment", required=True, help="deployment plan JSON file")
    admit.add_argument("--reclaim-pid", action="append", type=int, default=[])
    return parser.parse_args()


def run(command: list[str]) -> str:
    try:
        return subprocess.run(command, check=True, text=True, capture_output=True).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        detail = getattr(error, "stderr", "").strip()
        raise SelectionError(f"GPU query failed: {' '.join(command)}{': ' + detail if detail else ''}") from error


def inventory(tool: str, reclaim_pids: set[int]) -> list[Gpu]:
    raw = run([tool, "--query-gpu=index,uuid,pci.bus_id,memory.total,memory.free", "--format=csv,noheader,nounits"])
    reclaim: dict[str, int] = {}
    if reclaim_pids:
        apps = run([tool, "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"])
        for line in apps.splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) != 3 or not fields[1].isdigit() or not fields[2].isdigit():
                continue
            if int(fields[1]) in reclaim_pids:
                reclaim[fields[0]] = reclaim.get(fields[0], 0) + int(fields[2])
    result = []
    for line in raw.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 5:
            raise SelectionError(f"invalid nvidia-smi GPU row: {line}")
        index, uuid, pci, total, free = fields
        if not (index.isdigit() and total.isdigit() and free.isdigit()):
            raise SelectionError(f"non-numeric nvidia-smi GPU row: {line}")
        result.append(Gpu(int(index), uuid, pci, int(total), int(free), reclaim.get(uuid, 0)))
    if not result:
        raise SelectionError("nvidia-smi reported no GPUs")
    return result


def describe(gpus: list[Gpu]) -> str:
    return "; ".join(f"GPU {gpu.index} {gpu.uuid} {gpu.pci}: {gpu.effective_free_mib}/{gpu.total_mib} MiB "
                     "effective-free/total" for gpu in gpus)


def pci_key(value: str) -> tuple[int, str]:
    domain, rest = value.split(":", 1)
    return int(domain, 16), rest.lower()


def resolve(gpus: list[Gpu], selectors: str, count: str) -> tuple[list[Gpu], bool]:
    """The ordered GPUs a launch uses and whether COORDINATOR_GPUS named them."""
    text = selectors.strip()
    if not text:
        ordered = sorted(gpus, key=lambda gpu: gpu.index)
        return ordered[:1 if count == "1" else 2], False
    entries = [entry.strip() for entry in text.split(",")]
    if not 1 <= len(entries) <= 2 or not all(entries):
        raise SelectionError(f"COORDINATOR_GPUS must name one or two GPUs, primary first, not {text!r}")
    chosen = []
    for entry in entries:
        match = SELECTOR.fullmatch(entry)
        if match is None:
            raise SelectionError(f"COORDINATOR_GPUS entry {entry!r} is not a host GPU index, PCI bus id or GPU UUID")
        if match["index"] is not None:
            found = [gpu for gpu in gpus if gpu.index == int(entry)]
        elif match["uuid"] is not None:
            found = [gpu for gpu in gpus if gpu.uuid.lower() == entry.lower()]
        else:
            found = [gpu for gpu in gpus if pci_key(gpu.pci) == pci_key(entry)]
        if not found:
            raise SelectionError(f"COORDINATOR_GPUS={text} names GPU {entry}, but nvidia-smi reports no such "
                                 f"physical GPU on this host ({describe(gpus)})")
        chosen.append(found[0])
    if len(chosen) == 2 and chosen[0].uuid == chosen[1].uuid:
        raise SelectionError(f"COORDINATOR_GPUS={text} names GPU {chosen[0].index} twice; "
                             "the two coordinator GPUs must differ")
    if count != "auto" and len(chosen) != int(count):
        raise SelectionError(f"COORDINATOR_GPUS={text} names {len(chosen)} GPU(s), but RTX_GPUS/COORDINATOR_SPLIT "
                             f"ask for {count}; name exactly the GPUs to use")
    return chosen, True


def p2p_read_pairs(tool: str) -> set[tuple[int, int]]:
    raw = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", run([tool, "topo", "-p2p", "r"]))
    lines = [line.split() for line in raw.splitlines() if line.strip()]
    header = next((line for line in lines if len(line) >= 2 and all(re.fullmatch(r"GPU\d+", item) for item in line)), None)
    if header is None:
        raise SelectionError("cannot parse nvidia-smi peer-access matrix")
    pairs: set[tuple[int, int]] = set()
    for row in lines:
        if not row or not re.fullmatch(r"GPU\d+", row[0]) or len(row) < len(header) + 1:
            continue
        source = int(row[0][3:])
        for target_name, status in zip(header, row[1:]):
            if status == "OK":
                pairs.add((source, int(target_name[3:])))
    return pairs


def parse_bytes(value: str) -> int:
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]{1,6})?)(B|MB|GB|MiB|GiB)?", value)
    if not match:
        raise SelectionError(f"invalid KV pool size: {value}")
    scale = {None: 1, "B": 1, "MB": 10**6, "GB": 10**9, "MiB": MIB, "GiB": GIB}[match.group(2)]
    try:
        result = int(Decimal(match.group(1)) * scale)
    except InvalidOperation as error:
        raise SelectionError(f"invalid KV pool size: {value}") from error
    if result <= 0:
        raise SelectionError(f"memory size rounds to zero bytes: {value}")
    return result


def reservation_bytes(value: str, total_mib: int) -> int:
    if value.endswith("%"):
        try:
            percent = Decimal(value[:-1])
        except InvalidOperation as error:
            raise SelectionError(f"invalid memory reservation: {value}") from error
        result = int(total_mib * MIB * percent / Decimal(100))
    else:
        result = parse_bytes(value)
    if result <= 0 or result > total_mib * MIB:
        raise SelectionError(f"memory reservation exceeds device total or rounds to zero: {value}")
    return result


def desired_groups(args: argparse.Namespace) -> int:
    minimum = 2 * args.concurrency + 2 * args.retained_turns
    if args.kv_pool_size:
        groups = parse_bytes(args.kv_pool_size) // GROUP_BYTES
        if groups < minimum:
            raise SelectionError(
                f"explicit KV pool provides {groups} groups but minimum admission needs {minimum}"
            )
        return groups
    if args.memory_reservation:
        return minimum
    groups = math.ceil(args.max_context_tokens / 512) * args.concurrency
    return max(min(groups, DEFAULT_POOL_TOKENS // 512) + args.concurrency + 2 * args.retained_turns,
               math.ceil(args.max_context_tokens / 512) + minimum)


def required_mib(role: int, groups: int, minimum_expert_layers: int = 20,
                 expert_format: str = "native") -> int:
    layer_bytes = DUAL_LAYER_BYTES_BY_FORMAT[expert_format]
    required = (DUAL_FIXED_EX_EXPERTS[role] + DUAL_GROUP_BYTES[role] * groups
                + minimum_expert_layers * layer_bytes)
    return math.ceil(required / MIB)


def fits(gpu: Gpu, role: int, groups: int, reservation: str, minimum_expert_layers: int = 20,
         expert_format: str = "native") -> bool:
    required = required_mib(role, groups, minimum_expert_layers, expert_format)
    if gpu.total_mib < required or gpu.effective_free_mib < required:
        return False
    return not reservation or reservation_bytes(reservation, gpu.total_mib) >= required * MIB


def peer_pairs(tool: str, gpus: list[Gpu], required: bool) -> set[tuple[int, int]]:
    if len(gpus) < 2:
        return set()
    try:
        return p2p_read_pairs(tool)
    except SelectionError:
        if required:
            raise
        return set()


def bidirectional(peers: set[tuple[int, int]], first: Gpu, second: Gpu) -> bool:
    return (first.index, second.index) in peers and (second.index, first.index) in peers


def select(args: argparse.Namespace) -> dict:
    gpus = inventory(args.nvidia_smi, set(args.reclaim_pid))
    chosen, explicit = resolve(gpus, args.selectors, args.mode)
    primary = chosen[0]
    mode = str(len(chosen)) if explicit else args.mode
    if args.compact_spark:
        if mode != "1" or not args.memory_reservation:
            raise SelectionError("compact Spark requires mode 1 and a memory ceiling")
        if parse_bytes(args.memory_reservation) > 32 * GIB:
            raise SelectionError("compact Spark memory ceiling exceeds 32GiB")
    decision = "explicit" if explicit else "forced"
    if mode == "1":
        if args.kv_pool_size:
            parse_bytes(args.kv_pool_size)
        if args.memory_reservation:
            # Compact runtime clamps its absolute ceiling to physical total;
            # a nominal 32GiB board can report slightly less usable VRAM.
            reservation = args.memory_reservation
            if args.compact_spark:
                reservation = str(min(parse_bytes(reservation), primary.total_mib * MIB))
            reservation_bytes(reservation, primary.total_mib)
        return {"count": 1, "groups": None, "required_mib": None,
                "gpus": [primary.__dict__ | {"effective_free_mib": primary.effective_free_mib}],
                "decision": decision}
    groups = desired_groups(args)
    requirements = [required_mib(role, groups, args.minimum_expert_layers, args.expert_format)
                    for role in (0, 1)]
    if explicit:
        # The named pair is the pair: check it, never substitute another card.
        second = chosen[1]
        if not bidirectional(peer_pairs(args.nvidia_smi, gpus, True), primary, second):
            raise SelectionError(f"COORDINATOR_GPUS={args.selectors}: GPUs {primary.index} and {second.index} "
                                 "lack bidirectional peer reads")
        unfit = [f"GPU {gpu.index} ({gpu.uuid}) as logical GPU {role} needs {requirements[role]} MiB, "
                 f"has {gpu.effective_free_mib}/{gpu.total_mib} MiB effective-free/total"
                 for role, gpu in enumerate((primary, second))
                 if not fits(gpu, role, groups, args.memory_reservation, args.minimum_expert_layers,
                             args.expert_format)]
        if unfit:
            raise SelectionError(f"COORDINATOR_GPUS={args.selectors} cannot serve two-RTX V4.1: " + "; ".join(unfit))
        selected = [primary, second]
    else:
        peers = peer_pairs(args.nvidia_smi, gpus, args.mode == "2")
        candidates = sorted(
            (
                gpu for gpu in gpus
                if gpu.uuid != primary.uuid
                and bidirectional(peers, primary, gpu)
                and fits(gpu, 1, groups, args.memory_reservation, args.minimum_expert_layers,
                         args.expert_format)
            ),
            key=lambda gpu: (gpu.effective_free_mib, gpu.total_mib, -gpu.index),
            reverse=True,
        )
        dual = fits(primary, 0, groups, args.memory_reservation, args.minimum_expert_layers,
                    args.expert_format) and bool(candidates)
        if args.mode == "2" and not dual:
            raise SelectionError(
                f"forced two-RTX mode is infeasible; logical GPUs need {requirements[0]}/{requirements[1]} MiB; "
                + describe(gpus)
            )
        count = 2 if args.mode == "2" or (args.mode == "auto" and dual) else 1
        selected = [primary] + ([candidates[0]] if count == 2 else [])
        if args.mode == "auto":
            decision = "automatic-dual" if count == 2 else "automatic-single"
    return {
        "count": len(selected),
        "groups": groups if len(selected) == 2 else None,
        "required_mib": requirements if len(selected) == 2 else None,
        "gpus": [gpu.__dict__ | {"effective_free_mib": gpu.effective_free_mib} for gpu in selected],
        "decision": decision,
    }


def admit(args: argparse.Namespace) -> None:
    """Each selected GPU holds the plan's fixed owners for its logical device.

    The KV pool sizes itself into what remains, and with Spark ranks so do the
    RTX-onboarded expert layers; everything else must fit the GPU's free memory
    now. A GPU another job occupies fails here, before any container starts."""
    gpus = inventory(args.nvidia_smi, set(args.reclaim_pid))
    selected = []
    for uuid in (uuid for uuid in args.uuids.split(",") if uuid):
        found = [gpu for gpu in gpus if gpu.uuid == uuid]
        if not found:
            raise SelectionError(f"selected coordinator GPU {uuid} disappeared from nvidia-smi")
        selected.append(found[0])
    try:
        with open(args.deployment) as stream:
            deployment = json.load(stream)
        flexible = {"kv"} | ({"experts"} if deployment["spark_ranks"] else set())
        devices = sorted((device for device in deployment["memory_layout"]["devices"] if device["kind"] == "rtx"),
                         key=lambda device: device["index"])
        required = [sum(item["bytes"] for item in device["items"] if item["category"] not in flexible)
                    for device in devices]
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise SelectionError(f"the deployment plan names no coordinator memory layout: {error}") from error
    if len(required) != len(selected):
        raise SelectionError(f"the deployment plans {len(required)} coordinator GPU(s) for {len(selected)} selected")
    unfit = [f"GPU {gpu.index} ({gpu.uuid}) as logical GPU {logical} needs {math.ceil(need / MIB)} MiB "
             f"for its fixed owners, has {gpu.effective_free_mib}/{gpu.total_mib} MiB effective-free/total"
             for logical, (gpu, need) in enumerate(zip(selected, required))
             if gpu.effective_free_mib * MIB < need]
    if unfit:
        raise SelectionError("coordinator GPU cannot hold this deployment: " + "; ".join(unfit)
                             + "; free it or name other GPUs with COORDINATOR_GPUS")


def main() -> int:
    args = parse_args()
    try:
        if args.command == "resolve":
            chosen, _ = resolve(inventory(args.nvidia_smi, set()), args.selectors, args.count)
            for gpu in chosen:
                print(gpu.index, gpu.uuid, gpu.pci)
        elif args.command == "select":
            print(json.dumps(select(args)))
        else:
            admit(args)
        return 0
    except SelectionError as error:
        print(f"cuteafd release: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
