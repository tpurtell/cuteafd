#!/usr/bin/env python3
"""Publish a completed immutable run through verified, resumable HTTP LFS batches."""
from __future__ import annotations

import os
from pathlib import Path
import runpy
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hub_requirements import require_qualified_hub, use_http_lfs


def main():
    if "--publish-only" not in sys.argv[1:]:
        raise SystemExit("This entry point requires --publish-only")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    use_http_lfs()
    require_qualified_hub()  # Checked before the producer imports anything heavy.
    # Keep the original producer and its plan hashes unchanged on recovery;
    # its completion-marker checks and remote checksum audit still run.
    runpy.run_path(str(Path(__file__).with_name("run.py")), run_name="__main__")


if __name__ == "__main__":
    main()
