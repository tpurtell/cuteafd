#!/usr/bin/env python3
"""Write an official-model-names file from the installed Claude Code and Codex CLIs.

`cuteafd gateway --official-model-names-file FILE` reads the output so the
model ids that Claude Code's discovery (`GET /v1/models`, claude-* ids only)
and Codex's picker expect can be refreshed without a rebuild. The built-in
lists in rust/crates/cuteafd-api/src/gateway/models.rs come from this script
(Claude Code 2.1.289, Codex 0.161).

Sources:
  Claude Code: quoted "claude-{opus,sonnet,haiku,fable}-..." strings in the
               installed binary (`claude` on PATH, resolved through symlinks).
  Codex:       slugs in $CODEX_HOME/models_cache.json (default ~/.codex), the
               model list Codex fetched for its picker; hidden slugs included.
Neither source is contacted over the network.

Usage: official-model-names.py [--claude PATH] [--codex-cache PATH] [--json] > names.txt
"""
import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

CLAUDE_ID = re.compile(rb'"(claude-(?:opus|sonnet|haiku|fable)-[0-9][0-9a-z.-]*)"')


def claude_ids(binary: Path) -> list[str]:
    counts: dict[str, int] = {}
    with binary.open("rb") as handle:
        data = handle.read()
    for match in CLAUDE_ID.finditer(data):
        model = match.group(1).decode()
        counts[model] = counts.get(model, 0) + 1
    # Ids referenced many times are the live picker/default set; one-off
    # mentions are usually retired models in migration tables.
    return sorted((m for m, c in counts.items() if c >= 5), key=lambda m: (-counts[m], m))


def codex_ids(cache: Path) -> list[str]:
    data = json.loads(cache.read_text())
    models = data.get("models", data) if isinstance(data, dict) else data
    return [m.get("slug") or m.get("id") for m in models if isinstance(m, dict) and (m.get("slug") or m.get("id"))]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--claude", type=Path, help="Claude Code binary (default: claude on PATH)")
    parser.add_argument("--codex-cache", type=Path,
                        default=Path(os.environ.get("CODEX_HOME", Path.home() / ".codex")) / "models_cache.json")
    parser.add_argument("--json", action="store_true", help='write {"replace": false, "models": [...]}')
    args = parser.parse_args()
    ids: list[str] = []
    claude = args.claude or (Path(shutil.which("claude")).resolve() if shutil.which("claude") else None)
    if claude and claude.is_file():
        found = claude_ids(claude)
        # The npm launcher may be a small script; the real binary sits beside it.
        if not found:
            for candidate in claude.parent.glob("claude*"):
                if candidate.is_file() and candidate.stat().st_size > 10_000_000:
                    found = claude_ids(candidate)
                    break
        ids += found
    else:
        print("claude binary not found; skipping Claude Code ids", file=sys.stderr)
    if args.codex_cache.is_file():
        ids += codex_ids(args.codex_cache)
    else:
        print(f"{args.codex_cache} not found; skipping Codex ids", file=sys.stderr)
    ids = list(dict.fromkeys(ids))
    if args.json:
        print(json.dumps({"replace": False, "models": ids}, indent=2))
    else:
        print("# official model names (scripts/gateway/official-model-names.py)")
        print("\n".join(ids))
    return 0 if ids else 1


if __name__ == "__main__":
    sys.exit(main())
