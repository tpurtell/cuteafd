#!/usr/bin/env python3
"""Offline harness: run the LD-01 loop detector over good-output transcripts.

Goal (TJ, issue #948): before the loop abort is on by default, show the detector
never fires on good output from the fidelity dataset.

This is a faithful Python port of
`rust/crates/cuteafd-daemon/src/shared/repeat_guard.rs` (rules R and F). It is
NOT the shipped detector; it exists so the evidence run can be done without a
GPU or a live engine. The port is validated against the crate's own fixture
(`repeat_guard_fixture.json`) by `--fixture`.

Modes
  --fixture            run the three crate fixtures and assert the Rust tests'
                       expected outcomes (no dataset needed).
  --dataset DIR        walk DIR for *.json / *.jsonl transcripts, feed each
                       transcript's generated token ids through the detector,
                       and report transcript count, fire count and the exact
                       file/position of every fire.
  --fidelity DIR       read the published fidelity dataset's
                       `<config>/windows.json` (exact token ids + roles) and
                       feed each window's assistant-generated (`gen`) tokens
                       through the detector. This is the corpus TJ asked for:
                       reference generations from the official checkpoint, i.e.
                       known-good output. `--config` picks the config subdir
                       (default `deepseek_v41-v2_20261005`); `--all-roles` feeds
                       the whole window (ctx + gen) as a secondary check.
  --tokenizer PATH     optional: tokenize a text field with a HF tokenizer.json
                       (needs the `tokenizers` package) instead of reading ids.

Dataset layout is NOT assumed: pass --tokens-field (default: first of
`generated_ids`, `token_ids`, `tokens`, `ids`) and --text-field (default:
`text`). If the real fidelity dataset stores safetensors windows, extract the
generated-row token ids first and feed them as JSON.

Read-only. Writes nothing.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

WINDOW = 32
FAST_WINDOW = 256
MAX_ENTRIES = 65_536


class RepeatGuard:
    """Port of the Rust RepeatGuard (rules R and F)."""

    def __init__(self, run: int, copies: int, fast_span: int, fast_copies: int):
        self.run = run
        self.copies = min(max(copies, 2), 64)
        self.fast_span = fast_span
        self.fast_copies = min(max(fast_copies, 2), 64)
        self.ring = [0] * WINDOW
        self.cursor = 0
        self.counts: dict[tuple[int, ...], list[int]] = {}
        self.consecutive = 0
        self.first_seen_at: int | None = None
        self.pushed = 0
        self.fast_ring = [0] * FAST_WINDOW
        self.fast_cursor = 0
        self.fast_len = 0
        self.fast_run = [0] * (FAST_WINDOW + 1)
        self.fast_period: int | None = None
        self.last_hit: tuple[str, int | None] | None = None

    def _ring_tuple(self) -> tuple[int, ...]:
        return tuple(self.ring[(self.cursor + i) % WINDOW] for i in range(WINDOW))

    def push(self, token: int) -> tuple[str, int | None] | None:
        self.pushed += 1
        pushed = self.pushed
        hit = None

        if self.run != 0:
            self.ring[self.cursor] = token
            self.cursor = (self.cursor + 1) % WINDOW
            if pushed >= WINDOW:
                key = self._ring_tuple()
                entry = self.counts.get(key)
                seen = entry is not None and entry[0] >= self.copies - 1
                if entry is not None:
                    if entry[0] < self.copies:
                        entry[0] += 1
                elif len(self.counts) < MAX_ENTRIES:
                    self.counts[key] = [1, pushed - WINDOW]
                if seen:
                    if self.consecutive == 0:
                        self.first_seen_at = self.counts[key][1]
                    self.consecutive += 1
                    if self.consecutive >= self.run:
                        hit = ("Repeat", None)
                else:
                    self.consecutive = 0
                    self.first_seen_at = None

        if self.fast_span != 0:
            fired = None
            for period in range(1, FAST_WINDOW + 1):
                matches = (
                    self.fast_len >= period
                    and token == self.fast_ring[(self.fast_cursor + FAST_WINDOW - period) % FAST_WINDOW]
                )
                if matches:
                    self.fast_run[period] += 1
                else:
                    self.fast_run[period] = 0
                if fired is None:
                    threshold = max((self.fast_copies - 1) * period, self.fast_span)
                    if self.fast_run[period] >= threshold:
                        fired = period
            self.fast_ring[self.fast_cursor] = token
            self.fast_cursor = (self.fast_cursor + 1) % FAST_WINDOW
            if self.fast_len < FAST_WINDOW:
                self.fast_len += 1
            if fired is not None:
                self.fast_period = fired
                if hit is None:
                    hit = ("Fast", fired)

        self.last_hit = hit
        return hit

    def onset(self) -> int | None:
        if self.last_hit is None:
            return None
        kind, period = self.last_hit
        if kind == "Fast":
            assert period is not None
            run = self.fast_run[period]
            return self.pushed - (run + period)
        return self.first_seen_at


def feed(guard: RepeatGuard, ids: list[int]) -> tuple[int, tuple[str, int | None]] | None:
    for i, token in enumerate(ids):
        hit = guard.push(token)
        if hit is not None:
            return i + 1, hit
    return None


def block(length: int, seed: int) -> list[int]:
    x = seed
    out = []
    for _ in range(length):
        x = (x * 6_364_136_223_846_793_005 + 1_442_695_040_888_963_407) & ((1 << 64) - 1)
        out.append((x >> 33) & 0xFFFFFFFF)
    return out


def repeat(unit: list[int], copies: int) -> list[int]:
    return unit * copies


def rust_test_fixture() -> dict[str, list[int]]:
    """The synthetic streams the Rust unit tests build in `repeat_guard.rs`.

    The checked-in `repeat_guard_fixture.json` is a stub (the Rust tests generate
    these streams from `block()`/`repeat()`), so reproduce them here exactly.
    """
    looped = block(200, 101) + repeat(block(640, 102), 19)
    a = block(1500, 103)
    draft_then_write = a + block(1500, 104) + a + block(500, 105)
    fast_loop = block(300, 106) + repeat(block(24, 107), 80)
    return {"looped": looped, "draft_then_write": draft_then_write, "fast_loop": fast_loop}


def run_fixture(fixture_path: pathlib.Path) -> int:
    data = json.loads(fixture_path.read_text())
    if not {"looped", "draft_then_write", "fast_loop"} <= set(data):
        print(f"note: {fixture_path} is a stub; using the Rust tests' generated fixture")
        data = rust_test_fixture()
    failures = 0

    def check(name: str, ok: bool, detail: str) -> None:
        nonlocal failures
        print(f"{'PASS' if ok else 'FAIL'} {name}: {detail}")
        if not ok:
            failures += 1

    g = RepeatGuard(512, 8, 0, 8)
    got = feed(g, data["looped"])
    check(
        "looped fires Repeat",
        got is not None and got[1][0] == "Repeat" and 3_127 < got[0] < 32_000,
        f"at={got[0] if got else None} hit={got[1] if got else None}",
    )

    g = RepeatGuard(512, 8, 384, 8)
    got = feed(g, data["draft_then_write"])
    check("draft_then_write never fires", got is None, f"got={got}")

    g = RepeatGuard(0, 8, 384, 8)
    got = feed(g, data["fast_loop"])
    check(
        "fast_loop fires Fast",
        got is not None and got[1][0] == "Fast" and 294 < got[0] < 32_000,
        f"at={got[0] if got else None} hit={got[1] if got else None}",
    )

    # Rule-level checks copied from the Rust unit tests.
    g = RepeatGuard(512, 8, 384, 8)
    got = feed(g, repeat(block(30, 5), 20))
    check(
        "fast eight copies fire at span",
        got is not None and got[1] == ("Fast", 30) and got[0] == 30 + 384,
        f"got={got}",
    )

    g = RepeatGuard(512, 8, 384, 8)
    got = feed(g, [7] * 384)
    check("fast 384 same tokens does not fire", got is None, f"got={got}")
    g = RepeatGuard(512, 8, 384, 8)
    got = feed(g, [7] * 385)
    check(
        "fast 385 same tokens fires period 1",
        got is not None and got[1] == ("Fast", 1) and got[0] == 385,
        f"got={got}",
    )

    g = RepeatGuard(512, 8, 384, 8)
    got = feed(g, repeat(block(600, 3), 9))
    check(
        "eighth copy fires Repeat at run",
        got is not None and got[1][0] == "Repeat" and got[0] == 7 * 600 + 31 + 512,
        f"got={got}",
    )

    print(f"fixture: {failures} FAIL")
    return 1 if failures else 0


def iter_transcripts(root: pathlib.Path):
    for path in sorted(root.rglob("*")):
        if path.suffix in (".json", ".jsonl"):
            yield path


def extract_ids(obj, tokens_field: str, text_field: str, tokenizer):
    if isinstance(obj, list) and obj and isinstance(obj[0], int):
        return obj
    if isinstance(obj, dict):
        for key in (tokens_field, "generated_ids", "token_ids", "tokens", "ids"):
            value = obj.get(key)
            if isinstance(value, list) and value and isinstance(value[0], int):
                return value
        text = obj.get(text_field)
        if isinstance(text, str) and tokenizer is not None:
            return tokenizer.encode(text).ids
    return None


def run_dataset(root: pathlib.Path, tokens_field: str, text_field: str, tokenizer_path: str | None) -> int:
    tokenizer = None
    if tokenizer_path:
        try:
            from tokenizers import Tokenizer  # type: ignore[import-not-found]  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise SystemExit("--tokenizer needs the `tokenizers` package") from exc

        tokenizer = Tokenizer.from_file(tokenizer_path)

    transcripts = 0
    fires = 0
    for path in iter_transcripts(root):
        try:
            if path.suffix == ".jsonl":
                rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            else:
                rows = [json.loads(path.read_text())]
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            print(f"SKIP {path}: {exc}")
            continue
        for index, row in enumerate(rows):
            ids = extract_ids(row, tokens_field, text_field, tokenizer)
            if ids is None:
                continue
            transcripts += 1
            guard = RepeatGuard(512, 8, 384, 8)
            got = feed(guard, ids)
            if got is not None:
                fires += 1
                print(
                    f"FIRE {path}#{index} at token {got[0]} rule={got[1][0]} "
                    f"period={got[1][1]} onset={guard.onset()} tokens={len(ids)}"
                )

    print(f"transcripts={transcripts} fires={fires}")
    return 0


def run_fidelity(root: pathlib.Path, config: str, all_roles: bool) -> int:
    """Run the detector over the published fidelity dataset's windows.

    Each window stores exact token ids and a per-token role (`ctx` = prompt /
    repository source, `gen` = assistant-generated). The engine detector only
    ever sees generated tokens, so the primary corpus is the `gen` span of each
    window. `--all-roles` feeds the whole window as a secondary check.
    """
    windows_path = root / config / "windows.json"
    if not windows_path.is_file():
        raise SystemExit(f"missing {windows_path} (pass --fidelity <dataset-dir> --config <name>)")
    data = json.loads(windows_path.read_text())
    windows = data.get("windows")
    if not isinstance(windows, list):
        raise SystemExit(f"{windows_path}: no `windows` list")

    transcripts = 0
    fires = 0
    total_tokens = 0
    eligible = 0
    eligible_fires = 0
    max_run = 0
    max_copies = 0
    for window in windows:
        tokens = window.get("tokens")
        roles = window.get("roles")
        if not isinstance(tokens, list) or not isinstance(roles, list) or len(tokens) != len(roles):
            print(f"SKIP {window.get('id')}: tokens/roles missing or mismatched")
            continue
        if all_roles:
            ids = tokens
        else:
            ids = [t for t, r in zip(tokens, roles) if r == "gen"]
        if not ids:
            continue
        transcripts += 1
        total_tokens += len(ids)
        # Rule R needs WINDOW + run = 544 generated tokens before it can fire.
        can_fire = len(ids) >= WINDOW + 512
        if can_fire:
            eligible += 1
        guard = RepeatGuard(512, 8, 384, 8)
        got = None
        for i, token in enumerate(ids):
            hit = guard.push(token)
            max_run = max(max_run, guard.consecutive)
            if hit is not None:
                got = (i + 1, hit)
                break
        max_copies = max(max_copies, max((s[0] for s in guard.counts.values()), default=0))
        if got is not None:
            fires += 1
            if can_fire:
                eligible_fires += 1
            print(
                f"FIRE {window.get('id')} at token {got[0]} rule={got[1][0]} "
                f"period={got[1][1]} onset={guard.onset()} tokens={len(ids)}"
            )

    print(
        f"transcripts={transcripts} fires={fires} tokens={total_tokens} "
        f"rule_r_eligible={eligible} rule_r_eligible_fires={eligible_fires} "
        f"max_consecutive_seen={max_run} max_shingle_copies={max_copies}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=pathlib.Path, help="run the crate fixture self-test")
    parser.add_argument("--dataset", type=pathlib.Path, help="directory of transcript JSON/JSONL")
    parser.add_argument("--fidelity", type=pathlib.Path, help="fidelity dataset root (contains <config>/windows.json)")
    parser.add_argument("--config", default="deepseek_v41-v2_20261005", help="fidelity config subdir")
    parser.add_argument("--all-roles", action="store_true", help="fidelity: feed ctx+gen instead of gen only")
    parser.add_argument("--tokens-field", default="generated_ids")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--tokenizer", help="HF tokenizer.json for text transcripts")
    args = parser.parse_args()

    if args.fixture:
        return run_fixture(args.fixture)
    if args.dataset:
        return run_dataset(args.dataset, args.tokens_field, args.text_field, args.tokenizer)
    if args.fidelity:
        return run_fidelity(args.fidelity, args.config, args.all_roles)
    parser.error("pass --fixture, --dataset or --fidelity")
    return 2


if __name__ == "__main__":
    sys.exit(main())
