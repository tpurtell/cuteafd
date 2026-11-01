#!/usr/bin/env python3
"""Issue #928 evidence: dump/compare aligned native outputs across fork seals.

Uses the existing paired-screen builder, not a new benchmark or kernel API.
The b12x source is chosen externally by PYTHONPATH; no source is patched.
"""
import argparse
import hashlib
import json
from pathlib import Path


def digest(tensor):
    import torch

    return hashlib.sha256(tensor.detach().contiguous().view(torch.uint8)
                          .cpu().numpy().tobytes()).hexdigest()


def dump(args):
    import b12x
    import torch
    from benchmarks.bench_cuteafd_fp8_moe import native_pair_arm

    expected_source = Path(args.source).resolve()
    assert Path(b12x.__file__).resolve().parent.parent == expected_source
    torch.backends.cuda.matmul.allow_tf32 = False
    widths = [int(value) for value in args.widths.split(",")]
    result = dict(fork=args.fork, tree_sha256=args.tree_sha256, source=str(expected_source),
                  seed=args.seed, cells={})
    for width in widths:
        assert width > 0 and width % 32 == 0
        generator = torch.Generator().manual_seed(args.seed + width)
        experts, hidden, topk = 16, 5120, 6
        weights, scales = {}, {}
        for name, shape in (("w1", (experts, width, hidden // 2)),
                            ("w3", (experts, width, hidden // 2)),
                            ("w2", (experts, hidden, width // 2))):
            weights[name] = torch.randint(0, 256, shape, dtype=torch.uint8, generator=generator)
            scales[name] = torch.randint(121, 124, (*shape[:-1], shape[-1] // 16),
                                         dtype=torch.uint8, generator=generator)
        weight_hashes = {name: digest(value) for name, value in
                         [(f"weight.{k}", v) for k, v in weights.items()] +
                         [(f"scale.{k}", v) for k, v in scales.items()]}
        for capacity in (1, 16, 80):
            torch.manual_seed(args.seed + capacity)
            x = (torch.randn(capacity, hidden, device="cuda") * .5).bfloat16()
            ids = torch.rand(capacity, experts, device="cuda").topk(topk, -1).indices.int()
            routing = torch.rand(capacity, topk, device="cuda")
            routing /= routing.sum(-1, keepdim=True)
            # Old API only: aligned/padded storage, ordered coded tile192.
            run, output, owned = native_pair_arm(x, ids, routing, weights, scales,
                                                  192, False, False, True, qualify=True)
            run()
            torch.cuda.synchronize()
            snapshot = output.detach().cpu().clone()
            assert torch.isfinite(snapshot).all() and torch.count_nonzero(snapshot)
            key = f"s{width}c{capacity}"
            result["cells"][key] = dict(weights=weight_hashes,
                inputs={"x": digest(x), "ids": digest(ids), "routing": digest(routing)},
                output=snapshot, output_sha256=digest(snapshot), tile=192,
                stored=(width + 127) // 128 * 128, exact_storage=False,
                poison_live_graph=True)
            print(json.dumps(dict(cell=key, fork=args.fork,
                                  output_sha256=digest(snapshot))), flush=True)
            del owned, run, output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, args.output)


def compare(args):
    import torch

    old = torch.load(args.old, weights_only=True, map_location="cpu")
    new = torch.load(args.new, weights_only=True, map_location="cpu")
    assert old["seed"] == new["seed"] and old["cells"].keys() == new["cells"].keys()
    cells = {}
    for key, a in old["cells"].items():
        b = new["cells"][key]
        assert a["inputs"] == b["inputs"] and a["weights"] == b["weights"], key
        equal = torch.equal(a["output"], b["output"])
        cells[key] = dict(torch_equal=equal, old_sha256=a["output_sha256"],
                          new_sha256=b["output_sha256"],
                          max_abs=float((a["output"] - b["output"]).abs().max()))
        assert equal, (key, cells[key])
    report = dict(old={k: old[k] for k in ("fork", "tree_sha256")},
                  new={k: new[k] for k in ("fork", "tree_sha256")},
                  seed=old["seed"], cells=cells)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    emit = sub.add_parser("dump")
    emit.add_argument("--source", required=True)
    emit.add_argument("--fork", required=True)
    emit.add_argument("--tree-sha256", required=True)
    emit.add_argument("--widths", default="384,576,768,1152,2304")
    emit.add_argument("--seed", type=int, default=92811)
    emit.add_argument("--output", type=Path, required=True)
    check = sub.add_parser("compare")
    check.add_argument("--old", type=Path, required=True)
    check.add_argument("--new", type=Path, required=True)
    check.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dump(args) if args.command == "dump" else compare(args)


if __name__ == "__main__":
    main()
