#!/usr/bin/env python3
"""Export generic unweighted EXL3 reply epilogues through the native dispatcher."""
from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import _pinned_sparkinfer


def programs(width=4096, topk=8, max_rows=4096):
    from b12x.moe._shared.kernels.w4a16.expert_partial import compile_expert_partial_aot
    return [(format, "expert_partial", {"max_rows": max_rows},
             partial(compile_expert_partial_aot, width=width, topk=topk, format=format))
            for format in ("bf16", "nvfp4", "fp8")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--width", type=int, default=4096)
    parser.add_argument("--topk", type=int, default=8)
    args = parser.parse_args()
    from export_b12x_dsv4_aot import main as export
    sys.argv = [sys.argv[0], "--geometry", "expert_partial", "--output-dir", str(args.output_dir),
                "--expert-partial-width", str(args.width), "--expert-partial-topk", str(args.topk)]
    export()


if __name__ == "__main__":
    main()
