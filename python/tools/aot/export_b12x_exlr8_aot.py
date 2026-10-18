#!/usr/bin/env python3
"""Export EXLR8 A16 ABI v1 with the generic native program table."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import _pinned_sparkinfer


def programs(hidden=6144, units=16, top_k=8, max_groups=512):
    from b12x.integration.cuteafd.exlr8 import compile_exlr8_a16_aot
    return [("a16_decode", "exlr8_a16", {"max_rows": 64},
             lambda: compile_exlr8_a16_aot(hidden=hidden, units=units,
                                          top_k=top_k, max_groups=max_groups))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    from export_b12x_dsv4_aot import main as export
    sys.argv = [sys.argv[0], "--geometry", "exlr8",
                "--output-dir", str(args.output_dir)]
    export()


if __name__ == "__main__":
    main()
