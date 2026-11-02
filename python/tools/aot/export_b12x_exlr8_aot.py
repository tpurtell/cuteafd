#!/usr/bin/env python3
"""Export the fork's EXLR8 programs for one role (all, coordinator or spark)."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import _pinned_sparkinfer


def programs(hidden=6144, units=16, top_k=8, max_groups=512, role="all", experts=256,
             swiglu_limit=0):
    from b12x.integration.cuteafd.exlr8_glue import exlr8_export_programs
    return exlr8_export_programs(role, hidden=hidden, units=units,
                                  top_k=top_k, max_groups=max_groups, experts=experts,
                                  swiglu_limit=swiglu_limit)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--role", choices=("all", "coordinator", "spark"), default="all")
    args = parser.parse_args()
    from export_b12x_dsv4_aot import main as export
    sys.argv = [sys.argv[0], "--geometry", "exlr8", "--exlr8-role", args.role,
                "--output-dir", str(args.output_dir)]
    export()


if __name__ == "__main__":
    main()
