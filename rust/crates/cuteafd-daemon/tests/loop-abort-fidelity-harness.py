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
  --registered DIR     replay every registered text config, using its pinned
                       revision in fidelity_dataset.rs; write --evidence JSON.
  --tokenizer PATH     optional: tokenize a text field with a HF tokenizer.json
                       (needs the `tokenizers` package) instead of reading ids.

Dataset layout is NOT assumed: pass --tokens-field (default: first of
`generated_ids`, `token_ids`, `tokens`, `ids`) and --text-field (default:
`text`). If the real fidelity dataset stores safetensors windows, extract the
generated-row token ids first and feed them as JSON.

Input transcripts are read-only. --registered writes only --evidence.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import re
import json
import pathlib
import sys

WINDOW = 32
FAST_WINDOW = 256
MIN_REPEAT_PERIOD = FAST_WINDOW + 1
MAX_ENTRIES = 65_536
RUN = 512
COPIES = 8
FAST_SPAN = 1024
FAST_COPIES = 8


def rule_r_min_tokens():
    return WINDOW + (COPIES - 1) * MIN_REPEAT_PERIOD + RUN - 1


def rule_f_min_tokens():
    return 1 + max(FAST_COPIES - 1, FAST_SPAN)


def shipped_guard():
    return RepeatGuard(RUN, COPIES, FAST_SPAN, FAST_COPIES)


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
                period = (pushed - entry[2]) if entry is not None else None
                if entry is not None:
                    if entry[0] < self.copies:
                        entry[0] += 1
                    entry[2] = pushed
                elif len(self.counts) < MAX_ENTRIES:
                    self.counts[key] = [1, pushed, pushed]
                if seen:
                    if self.consecutive == 0:
                        self.first_seen_at = self.counts[key][1]
                    self.consecutive += 1
                    # Rule R is the long-block detector: only periods beyond
                    # rule F's window may fire it (see the Rust module doc).
                    if self.consecutive >= self.run and period is not None and period >= MIN_REPEAT_PERIOD:
                        hit = ("Repeat", period)
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

    The full checked-in fixture is also consumed by Rust's fixture test.
    Reproduce its deterministic streams here to cross-check its contents.
    """
    looped = block(200, 101) + repeat(block(640, 102), 19)
    a = block(1500, 103)
    draft_then_write = a + block(1500, 104) + a + block(500, 105)
    fast_loop = block(300, 106) + repeat(block(24, 107), 80)
    return {"looped": looped, "draft_then_write": draft_then_write, "fast_loop": fast_loop}


def run_fixture(fixture_path: pathlib.Path) -> int:
    data = json.loads(fixture_path.read_text())
    if {name: data.get(name) for name in rust_test_fixture()} != rust_test_fixture():
        raise ValueError(f"{fixture_path}: differs from the Rust tests' generated fixture")
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

    g = RepeatGuard(512, 8, 1024, 8)
    got = feed(g, data["draft_then_write"])
    check("draft_then_write never fires", got is None, f"got={got}")

    g = RepeatGuard(0, 8, 1024, 8)
    got = feed(g, data["fast_loop"])
    check(
        "fast_loop fires Fast",
        got is not None and got[1][0] == "Fast" and 294 < got[0] < 32_000,
        f"at={got[0] if got else None} hit={got[1] if got else None}",
    )

    # Rule-level checks copied from the Rust unit tests.
    g = RepeatGuard(0, 8, 1024, 8)
    got = feed(g, repeat(block(30, 5), 40))
    check(
        "fast eight copies fire at span",
        got is not None and got[1] == ("Fast", 30) and got[0] == 30 + 1024,
        f"got={got}",
    )

    # A legitimate 400-token identical run (a zero array / table row) must not
    # abort at the shipped defaults.
    g = RepeatGuard(512, 8, 1024, 8)
    got = feed(g, [0] * 400)
    check("legitimate 400 identical tokens do not fire", got is None, f"got={got}")

    # V2: rule R is gated to periods > FAST_WINDOW, so a period-1 run waits for
    # rule F's 1,024-token span (without the gate rule R would fire at 550).
    g = RepeatGuard(512, 8, 1024, 8)
    got = feed(g, [0] * 1024)
    check("1,024 identical tokens do not fire", got is None, f"got={got}")

    g = RepeatGuard(512, 8, 1024, 8)
    got = feed(g, [0] * 1025)
    check(
        "1,025 identical tokens fire rule F at the span",
        got is not None and got[1] == ("Fast", 1) and got[0] == 1 + 1024,
        f"got={got}",
    )

    g = RepeatGuard(512, 8, 1024, 8)
    got = feed(g, repeat(block(8, 12), 200))
    check(
        "period-8 repeat waits for rule F",
        got is not None and got[1] == ("Fast", 8) and got[0] == 8 + 1024,
        f"got={got}",
    )

    g = RepeatGuard(512, 8, 1024, 8)
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
            guard = shipped_guard()
            got = feed(guard, ids)
            if got is not None:
                fires += 1
                print(
                    f"FIRE {path}#{index} at token {got[0]} rule={got[1][0]} "
                    f"period={got[1][1]} onset={guard.onset()} tokens={len(ids)}"
                )

    print(f"transcripts={transcripts} fires={fires}")
    return int(fires != 0 or transcripts == 0)


def fidelity_metrics(windows_path: pathlib.Path, all_roles: bool = False) -> dict:
    data = json.loads(windows_path.read_text())
    windows = data.get("windows")
    if not isinstance(windows, list) or not windows:
        raise ValueError(f"{windows_path}: no windows")
    metrics = dict(transcripts=0, tokens=0, fires=0, rule_r_eligible=0,
                   rule_f_eligible=0, max_consecutive_seen=0, max_shingle_copies=0)
    for window in windows:
        tokens, roles = window.get("tokens"), window.get("roles")
        if (not isinstance(tokens, list) or not isinstance(roles, list)
                or len(tokens) != len(roles) or any(r not in ("ctx", "gen") for r in roles)
                or any(type(t) is not int or not 0 <= t <= 0xFFFFFFFF for t in tokens)):
            raise ValueError(f"{windows_path}#{window.get('id')}: malformed tokens/roles")
        ids = tokens if all_roles else [t for t, role in zip(tokens, roles) if role == "gen"]
        if not ids:
            continue
        metrics["transcripts"] += 1
        metrics["tokens"] += len(ids)
        metrics["rule_r_eligible"] += int(len(ids) >= rule_r_min_tokens())
        metrics["rule_f_eligible"] += int(len(ids) >= rule_f_min_tokens())
        guard = shipped_guard()
        for i, token in enumerate(ids):
            hit = guard.push(token)
            metrics["max_consecutive_seen"] = max(metrics["max_consecutive_seen"], guard.consecutive)
            if hit is not None:
                raise ValueError(f"FIRE {windows_path}#{window.get('id')} at token {i + 1} "
                                 f"rule={hit[0]} period={hit[1]} onset={guard.onset()}")
        metrics["max_shingle_copies"] = max(metrics["max_shingle_copies"],
                                          max((v[0] for v in guard.counts.values()), default=0))
    if not metrics["transcripts"]:
        raise ValueError(f"{windows_path}: no generated transcripts")
    return metrics


def run_fidelity(root: pathlib.Path, config: str, all_roles: bool) -> int:
    print(json.dumps(fidelity_metrics(root / config / "windows.json", all_roles), sort_keys=True))
    return 0


def registered_configs(registry: pathlib.Path):
    constants = dict(re.findall(r'pub const (\w+): &str = "([^"\n]+)";', registry.read_text()))
    for name, config in constants.items():
        if not name.endswith("CONFIG") or name.startswith("RETIRED_") or name.endswith("MEDIA_CONFIG"):
            continue
        revision = constants[name.removesuffix("CONFIG") + "REVISION"]
        yield config, revision


def run_registered(cache: pathlib.Path, output: pathlib.Path) -> int:
    root = pathlib.Path(__file__).resolve().parents[4]
    registry = root / "rust/crates/cuteafd-bench/src/fidelity_dataset.rs"
    configs = []
    for config, revision in registered_configs(registry):
        path = cache / revision / config / "windows.json"
        metrics = fidelity_metrics(path)
        print(config, json.dumps(metrics, sort_keys=True), flush=True)
        configs.append(dict(config=config, revision=revision,
                            windows_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), **metrics))
    fixture = root / "rust/crates/cuteafd-daemon/src/shared/repeat_guard_fixture.json"
    if run_fixture(fixture):
        return 1
    detector = root / "rust/crates/cuteafd-daemon/src/shared/repeat_guard.rs"
    evidence = {
        "schema": "cuteafd.loop_abort.fidelity_evidence/2",
        "generated": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "detector": dict(module=str(detector.relative_to(root)),
                         sha256=hashlib.sha256(detector.read_bytes()).hexdigest(),
                         rule_r=dict(window=WINDOW, run=RUN, copies=COPIES,
                                     min_period=MIN_REPEAT_PERIOD, min_tokens=rule_r_min_tokens()),
                         rule_f=dict(window=FAST_WINDOW, span=FAST_SPAN, copies=FAST_COPIES,
                                     min_tokens=rule_f_min_tokens()), max_entries=MAX_ENTRIES),
        "harness": dict(path=str(pathlib.Path(__file__).resolve().relative_to(root)),
                        sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()),
        "fidelity": dict(repository="wrldsuksgo2mars/cuteafd-fidelity",
                         registry=str(registry.relative_to(root)),
                         registry_sha256=hashlib.sha256(registry.read_bytes()).hexdigest(),
                         configs_run=len(configs), configs=configs,
                         note="Every registered text config, pinned to fidelity_dataset.rs. "
                              "Only assistant-generated tokens are fed, with a fresh guard per window. "
                              "Neither rule can fire inside a 1024-token generation window: "
                              "Rule R's periodic-stream minimum and Rule F's minimum are derived "
                              "from the shipped constants. Longer windows are reported as eligible; "
                              "synthetic boundary tests exercise actual loop detection."),
        "fixture": dict(path=str(fixture.relative_to(root)),
                        sha256=hashlib.sha256(fixture.read_bytes()).hexdigest(),
                        streams={k: len(v) for k, v in rust_test_fixture().items()},
                        validated=True),
        "corpus_replay": "Dropped: private production inputs are unavailable; no old control metrics retained.",
        "reproduce": ["python3 -I rust/crates/cuteafd-daemon/tests/loop-abort-fidelity-harness.py "
                      "--registered <fidelity-cache/repository> --evidence "
                      "rust/crates/cuteafd-daemon/tests/loop-abort-fidelity-evidence.json"],
    }
    output.write_text(json.dumps(evidence, indent=2) + "\n")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registered", type=pathlib.Path, help="replay all registered text configs from revision/config/windows.json cache")
    parser.add_argument("--evidence", type=pathlib.Path, help="write regenerated evidence (requires --registered)")
    parser.add_argument("--fixture", type=pathlib.Path, help="run the crate fixture self-test")
    parser.add_argument("--dataset", type=pathlib.Path, help="directory of transcript JSON/JSONL")
    parser.add_argument("--fidelity", type=pathlib.Path, help="fidelity dataset root (contains <config>/windows.json)")
    parser.add_argument("--config", default="deepseek_v41-v2_20261005", help="fidelity config subdir")
    parser.add_argument("--all-roles", action="store_true", help="fidelity: feed ctx+gen instead of gen only")
    parser.add_argument("--tokens-field", default="generated_ids")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--tokenizer", help="HF tokenizer.json for text transcripts")
    args = parser.parse_args()

    if args.registered:
        if not args.evidence:
            parser.error("--registered requires --evidence")
        return run_registered(args.registered, args.evidence)
    if args.evidence:
        parser.error("--evidence requires --registered")
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
