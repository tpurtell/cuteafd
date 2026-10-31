#!/usr/bin/env python3
"""Check exported slice execution through the production expert C ABI."""

import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(_Path(__file__).resolve().parents[2] / _d) for _d in ("lib",)]  # sibling tool dirs

import argparse
import json
from spark_packages import native_spark_roles


CAPACITIES = (1, 16, 80, 256, 1024, 4096)


def repeat_equal(replay, snapshot, repeats):
    """Compare raw bytes, not float equality (which hides signed-zero changes)."""
    first = None
    exact = True
    for _ in range(repeats):
        replay()
        current = snapshot()
        if first is None:
            first = current
        else:
            exact &= current == first
    return exact


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-lib", required=True)
    parser.add_argument("--width", type=int, choices=(64, 128, 192), required=True)
    parser.add_argument(
        "--capacities",
        help="Only check these native variants (default: all for repeat equality; 1,16,80 otherwise)",
    )
    parser.add_argument("--full-backbone", action="store_true", help="Check full-width RTX backbone geometry")
    parser.add_argument("--local-entrypoints", action="store_true", help="Use namespaced full-backbone symbols in a combined library")
    family = parser.add_mutually_exclusive_group()
    family.add_argument("--spark-tp", type=int, choices=tuple(native_spark_roles()))
    family.add_argument("--tp2", action="store_true")
    parser.add_argument("--require-ordered", action="store_true",
                        help="Reject token-atomic ABI 3 (candidate correctness gate)")
    parser.add_argument("--repeat-equality", type=int, metavar="N",
                        help="Require identical output bytes over N fixed-input replays (N >= 3)")
    options = parser.parse_args(argv)
    if options.local_entrypoints and not options.full_backbone:
        parser.error("local entrypoints require --full-backbone")
    if options.full_backbone and (options.spark_tp or options.tp2):
        parser.error("select only one expert family")
    selected = options.capacities or (
        ",".join(map(str, CAPACITIES)) if options.repeat_equality is not None else "1,16,80")
    capacities = {int(x) for x in selected.split(",")}
    allowed = set(CAPACITIES if options.repeat_equality is not None else (1, 16, 80))
    if not capacities or not capacities <= allowed:
        parser.error(f"capacities must be a nonempty subset of {sorted(allowed)}")
    if options.repeat_equality is not None and options.repeat_equality < 3:
        parser.error("repeat equality requires at least 3 replays")
    return options, capacities


def main():
    options, capacities = parse_args()
    import torch
    import _pinned_sparkinfer  # noqa: F401
    from _v41_expert_native import Native, library
    from tests.moe.test_v41_grouped_slices import _check_grouped_slices

    checked = set()
    repeat_records = []
    lib = library(options.native_lib, local=options.local_entrypoints,
                  tp2=options.tp2, spark_tp=options.spark_tp)
    owners = {}
    full_weights = None
    resident_weights = None

    def coded_weights(weights, experts, intermediate, hidden):
        from b12x.moe._shared.scale_codes import pack_scale_codes

        def logical_scales(packed, rows, columns, half=None):
            # Undo the oracle's N256/K128 scale swizzle before lossless coding.
            stored = packed.view(torch.uint8).reshape(
                experts, -1, (columns + 3) // 4, 32, 8, 4)
            grid = stored.permute(0, 1, 3, 4, 2, 5).reshape(
                experts, -1, ((columns + 3) // 4) * 4)[..., :columns]
            if half is not None:
                padded = (half + 127) // 128 * 128
                grid = torch.cat((grid[:, :half], grid[:, padded:padded + half]), 1)
            return grid[:, :rows].contiguous()

        w13, s13, w2, s2 = weights
        s13 = pack_scale_codes(
            logical_scales(s13, intermediate * 2, hidden // 32, intermediate),
            tensor_name="oracle.w13.scale", gated_half_rows=intermediate)
        s2 = pack_scale_codes(logical_scales(s2, hidden, intermediate // 32),
                             tensor_name="oracle.w2.scale")
        return [w13, s13.view(torch.uint32).flatten(), w2, s2.view(torch.uint32).flatten()]

    def check(**case):
        nonlocal full_weights, resident_weights
        rows = case["rows"]
        weights, wire, ids_cpu, routing = case["native_inputs"]
        if resident_weights is None:
            resident_weights = coded_weights(weights, 128 if options.full_backbone else 32,
                                            case["intermediate"], case["hidden"])
        weights = resident_weights
        capacity = (case["capacity"] if options.repeat_equality is not None else
                    1 if rows == 1 else 16 if rows <= 16 else 80)
        if capacity not in capacities:
            return
        if options.full_backbone:
            # The independent oracle owns 128 experts. Place its exact packed
            # bytes in the final third of a 384-expert arena to exercise IDs
            # through 383 without changing the expected numerical result.
            if full_weights is None:
                full_weights = []
                for weight in weights:
                    arena = torch.zeros(weight.numel() * 3, dtype=weight.dtype, device=weight.device)
                    arena[weight.numel() * 2:].copy_(weight.reshape(-1))
                    full_weights.append(arena)
            weights = full_weights
            ids_cpu = ids_cpu + 256
        if capacity not in owners:
            ids = torch.empty(capacity, 6, dtype=torch.int32, device="cuda")
            rw = torch.empty(capacity, 6, device="cuda")
            native = Native(lib, capacity, weights, wire, ids, rw,
                            full_backbone=options.full_backbone, tp2=options.tp2,
                            spark_tp=options.spark_tp)
            assert not options.require_ordered or native.info.abi_version == 2, (
                capacity, native.info.abi_version, "ordered ABI 2 required")
            ids.fill_(-1)
            rw.zero_()
            native.run(1)
            graph = torch.cuda.CUDAGraph()
            # Rows stay at capacity; invalid routes mask inactive input rows.
            with torch.cuda.graph(graph):
                native.run(capacity)
            owners[capacity] = native, ids, rw, graph
        native, ids, rw, graph = owners[capacity]
        if options.repeat_equality is not None:
            graph.reset()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                native.run(rows)
            owners[capacity] = native, ids, rw, graph
        ids.fill_(-1)
        ids[:rows].copy_(ids_cpu)
        rw.zero_()
        rw[:rows].copy_(routing)
        native.output.fill_(float("nan"))
        allocated = torch.cuda.memory_allocated()
        graph.replay()
        assert torch.cuda.memory_allocated() == allocated
        if options.repeat_equality is not None:
            live_output = native.output[:rows if native.token_accumulation else rows * 6]
            actual = live_output if native.token_accumulation else live_output.reshape(rows, 6, -1).sum(1)
            expected = case["expected"]
            assert bool(torch.isfinite(live_output).all()) and bool(actual.norm() > 0)
            relative = ((actual - expected).norm() / expected.norm()).item()
            cosine = torch.nn.functional.cosine_similarity(
                actual.flatten(), expected.flatten(), dim=0).item()
            assert relative < 0.01 and cosine > 0.9999, (capacity, rows, relative, cosine)
            oracle_exact = None
            if not native.token_accumulation:
                reference = torch.zeros_like(native.output)
                summed = torch.zeros_like(case["out"][0, :rows * 6])
                for plane in case["out"][:, :rows * 6]:
                    summed += plane
                pairs = case["pair_gpu"]
                reference[pairs[:, 0] * 6 + pairs[:, 1]] = summed
                oracle_exact = torch.equal(live_output.view(torch.uint8),
                                           reference[:rows * 6].view(torch.uint8))
                assert oracle_exact, (capacity, rows, "ordered oracle bytes differ")
            exact = repeat_equal(
                graph.replay,
                lambda: native.output.detach().cpu().numpy().tobytes(),
                options.repeat_equality,
            )
            record = dict(capacity=capacity, rows=rows, abi=native.info.abi_version,
                          repeats=options.repeat_equality, exact=exact,
                          ordered_oracle_exact=oracle_exact,
                          scratch_bytes=native.storage.numel())
            repeat_records.append(record)
            print("REPEAT " + json.dumps(record), flush=True)
            checked.add(capacity)
            return
        reference = torch.zeros_like(native.output)
        pairs = case["pair_gpu"]
        # Same-width independently launched grouped compute with CPU metadata.
        partial = case["out"][:, : rows * 6]
        summed = torch.zeros_like(partial[0])
        for plane in partial:
            summed += plane
        reference[pairs[:, 0] * 6 + pairs[:, 1]] = summed
        assert torch.equal(native.output, reference), case["case"]
        checked.add(capacity)
        print(
            "NATIVE "
            + json.dumps(
                dict(
                    case=case["case"],
                    rows=rows,
                    width=options.width,
                    exact=True,
                    scratch_bytes=native.storage.numel(),
                )
            ),
            flush=True,
        )

    intermediate = (2304 if options.full_backbone else 1152 if options.tp2 else
                    2304 // options.spark_tp if options.spark_tp else 576)
    if options.repeat_equality is not None:
        for capacity in sorted(capacities):
            row_counts = sorted({capacity, 128} if capacity == 256 else {capacity})
            _check_grouped_slices(options.width, after_case=check, n=intermediate,
                                  topk=6, capacity=capacity, row_counts=row_counts)
            _, _, _, graph = owners.pop(capacity)
            graph.reset()
    else:
        _check_grouped_slices(options.width, after_case=check, n=intermediate, topk=6)
    assert checked == capacities, (checked, capacities)
    for _, _, _, graph in owners.values():
        graph.reset()
    assert all(record["exact"] for record in repeat_records), repeat_records


if __name__ == "__main__":
    main()
