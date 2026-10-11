#!/usr/bin/env python3
"""CPU HTTP driver for V4.1 stage-2 prefix gates; no model or hardware launch.

Partial replay NLL and authoritative odd-frontier/image-state checks run through
the in-process golden probe, not this HTTP driver. Optional HTTP image isolation
needs a known frontier (--image-start-token); otherwise it is marked blocked.
--reference-url is optional and only adds an informational cold-match note.
"""
from __future__ import annotations

import argparse
import base64
import concurrent.futures
from contextlib import contextmanager
import copy
import hashlib
import json
import mimetypes
from pathlib import Path
import runpy
import statistics
import threading
import time
import urllib.error
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parents[3]
API = runpy.run_path(str(ROOT / "scripts/qualify-ds41-native-api.py"))
PREFIX = runpy.run_path(str(ROOT / "scripts/qualify/qualify-prefix-cache.py"))


def cached(result):
    usage = result["usage"]
    hit = usage.get("prompt_tokens_details", {}).get("cached_tokens")
    alias = usage.get("prompt_cache_hit_tokens")
    if hit is None:
        hit = alias
    if hit is None or not 0 <= hit <= usage["prompt_tokens"]:
        raise ValueError("missing/invalid cached token accounting")
    if alias is not None and alias != hit:
        raise ValueError("cached token fields disagree")
    return hit


def parity_prompt(prompt, parity, send, attempts=16):
    """Measure, then try repeated one-token suffixes; never guess token parity."""
    tried = []
    for n in range(attempts):
        padded = prompt + " x" * n
        result = send(padded)
        count = result["usage"]["prompt_tokens"]
        tried.append(dict(padding=n, prompt_tokens=count, parity=count % 2))
        if count % 2 == parity:
            return padded, result, tried
    raise ValueError(f"cannot arrange parity {parity}: {tried}")


def flatten(value, prefix=""):
    out = {}
    if isinstance(value, dict):
        for key, child in value.items():
            out.update(flatten(child, f"{prefix}.{key}" if prefix else key))
    elif isinstance(value, (int, float, bool)):
        out[prefix] = value
    return out


def delta(before, after):
    a, b = flatten(before), flatten(after)
    return {key: b[key] - a[key] for key in a.keys() & b.keys()}


def compare_reports(candidate, baseline):
    """Emit a before/after table and reject failed or incompatible gate reports."""
    problems, rows = [], []
    if candidate.get("command") != baseline.get("command"):
        problems.append("report commands differ")
    for key in ("model", "nonce", "config"):
        if candidate.get(key) != baseline.get(key):
            problems.append(f"report {key} differs")
    if not baseline.get("pass"):
        problems.append("baseline gate failed")
    old_cases = {c["name"]: c for c in baseline.get("cases", [])}
    new_cases = {c["name"]: c for c in candidate.get("cases", [])}
    if old_cases.keys() != new_cases.keys():
        problems.append("case sets differ")
    if not candidate.get("pass"):
        problems.append("candidate gate failed")
    for name in sorted(old_cases.keys() | new_cases.keys()):
        old, new = old_cases.get(name, {}), new_cases.get(name, {})
        a, b = old.get("metrics", {}), new.get("metrics", {})
        for key in sorted(a.keys() | b.keys()):
            before, after = a.get(key), b.get(key)
            rows.append(dict(case=name, metric=key, before=before, after=after))
        if not new.get("pass"):
            problems.append(f"{name}: case failed or missing")
        rows.append(dict(case=name, metric="pass", before=old.get("pass"), after=new.get("pass")))
    return dict(command="compare", **{"pass": not problems}, problems=problems, rows=rows)


class Driver:
    def __init__(self, args):
        self.args = args
        self.report = dict(schema=1, command=args.command, base_url=args.base_url,
                           nonce=args.nonce, model=args.model, cases=[], **{"pass": False})
        self.report["config"] = {key: getattr(args, key) for key in
            ("timeout", "entries", "sessions", "max_turns", "odd_frontier", "image_start_token")
            if hasattr(args, key)}
        # Reuse the existing SSE parser with this driver's bounded opener.
        self.stream = PREFIX["stream"]
        self.stream.__globals__["open_request"] = self.open_request
        self.report["limitations"] = [
            "HTTP scoring/cold probes are in-process only; no generated token IDs",
            "exact greedy replies do not independently prove internal state byte identity",
            "image frontier must be supplied until HTTP exports media spans",
        ]

    def save(self):
        self.args.output.parent.mkdir(parents=True, exist_ok=True)
        self.args.output.write_text(json.dumps(self.report, indent=2) + "\n")

    @contextmanager
    def open_request(self, base, body, timeout=None):
        started = time.monotonic()
        headers = {"Content-Type": "application/json"}
        if self.args.api_key:
            headers["Authorization"] = "Bearer " + self.args.api_key
        request = urllib.request.Request(base.rstrip("/") + "/v1/chat/completions",
                                         data=json.dumps(body).encode(), headers=headers)
        with urllib.request.urlopen(request, timeout=self.args.timeout) as response:
            timer = threading.Timer(max(0.001, self.args.timeout - (time.monotonic() - started)),
                                    PREFIX["hang_up"], [response])
            timer.start()
            try:
                yield response
            finally:
                timer.cancel()

    def body(self, prompt, max_tokens=64):
        return dict(API["payload"](prompt), model=self.args.model, max_tokens=max_tokens)

    def send(self, body, base=None):
        started = time.monotonic()
        result = self.stream(base or self.args.base_url, body)
        if not result.get("usage") or not result.get("finish_reason"):
            raise ValueError("incomplete completion")
        cached(result)
        result["wall_seconds"] = time.monotonic() - started
        return result

    def get(self, path):
        headers = {"Authorization": "Bearer " + self.args.api_key} if self.args.api_key else {}
        request = urllib.request.Request(self.args.base_url.rstrip("/") + path, headers=headers)
        with urllib.request.urlopen(request, timeout=min(self.args.timeout, 10)) as response:
            return json.load(response)

    @contextmanager
    def console_viewer(self):
        # Snapshot GETs do not count as viewers: keep SSE open so KV gauges
        # are actually produced. Drain frames to avoid backpressure.
        headers = {"Authorization": "Bearer " + self.args.api_key} if self.args.api_key else {}
        url = self.args.base_url.rstrip("/") + "/v1/console/events"
        response = urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=20)
        stopped = threading.Event()
        def drain():
            try:
                for _ in response:
                    if stopped.is_set():
                        break
            except (OSError, ValueError):
                pass
        thread = threading.Thread(target=drain, daemon=True)
        thread.start()
        try:
            yield
        finally:
            stopped.set()
            PREFIX["hang_up"](response)
            thread.join(timeout=5)
            response.close()

    def case(self, name, fn):
        case = dict(name=name, **{"pass": False}, metrics={})
        self.report["cases"].append(case)
        self.save()
        try:
            fn(case)
        except Exception as error:
            case["pass"] = False
            case["error"] = f"{type(error).__name__}: {error}"
        self.save()
        print(f"{name}: {'PASS' if case['pass'] else 'FAIL'}", flush=True)

    def tag(self, name):
        return "Session " + hashlib.sha256(f"{self.args.nonce}/{name}".encode()).hexdigest()[:24] + ". "

    def reference(self, body, case):
        if self.args.reference_url:
            try:
                result = self.send(body, self.args.reference_url)
                case["reference"] = result
                case["reference_cold"] = cached(result) == 0
                return result
            except Exception as error:
                case["reference_error"] = str(error)
        return None

    def exact_case(self, case, body, first=None, suffix_kind="short"):
        first = first or self.send(body)
        again = self.send(body)
        case.update(request=body, cold=first, repeat=again)
        tokens = first["usage"]["prompt_tokens"]
        identical = (first["content"], first["reasoning"]) == (again["content"], again["reasoning"])
        case["metrics"].update(prompt_tokens=tokens, parity=tokens % 2,
                                cached_tokens=cached(again), byte_exact=identical)
        follow = copy.deepcopy(body)
        follow["messages"] += [dict(role="assistant", content=first["content"],
                                    reasoning_content=first["reasoning"]),
                               dict(role="user", content=(" x" * 256 if suffix_kind == "long" else "") +
                                    " Continue briefly.")]
        continued = self.send(follow)
        frontier = tokens + first["usage"]["completion_tokens"] - 1
        suffix = continued["usage"]["prompt_tokens"] - frontier
        case["continuation"] = continued
        case["metrics"].update(continuation_cached_tokens=cached(continued), turn_frontier=frontier,
                                suffix_tokens=suffix)
        reference = self.reference(body, case)
        if reference:
            case["reference_greedy_match"] = reference["content"] == first["content"]
        case["pass"] = (cached(again) == tokens and identical and cached(continued) >= frontier
                        and (suffix >= 128 if suffix_kind == "long" else 0 < suffix < 128))

    def exact(self):
        for size, count in (("short", 12), ("long", 4096)):
            for parity in ((0, 1) if self.args.odd_frontier else (None,)):
                for suffix in ("short", "long"):
                    name = f"text_{size}_{parity if parity is not None else 'observed'}_{suffix}_suffix"
                    def run(case, name=name, size=size, count=count, parity=parity, suffix=suffix):
                        prompt = self.tag(name) + " x" * count + "\nWrite a short Python identity function."
                        if parity is None:
                            padded, first, tried = prompt, self.send(self.body(prompt)), []
                        else:
                            padded, first, tried = parity_prompt(prompt, parity, lambda p: self.send(self.body(p)))
                        case["padding_trials"] = tried
                        self.exact_case(case, self.body(padded), first, suffix)
                        if size == "short" and first["usage"]["prompt_tokens"] >= 128:
                            case["pass"] = False
                        if size == "long" and first["usage"]["prompt_tokens"] < 2048:
                            case["pass"] = False
                    self.case(name, run)
        if self.args.image_start_token is not None:
            self.case("image", self.image_case)
        else:
            self.report["blocked"] = ["optional image isolation: supply --image and --image-start-token",
                                      "authoritative odd-frontier/image state and partial NLL: in-process probe"]

    def image_case(self, case):
        images = sorted(p for p in (ROOT / "scripts/fixtures").rglob("*")
                        if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
        path = self.args.image or (images[0] if images else None)
        if path is None:
            raise ValueError("no scripts/fixtures image: supply --image PATH")
        other = self.args.different_image or ROOT / "rust/crates/cuteafd-api/src/openai/fixtures/black.png"
        if path.read_bytes() == other.read_bytes():
            raise ValueError("images must differ: supply --different-image PATH")
        def data_url(p):
            mime = mimetypes.guess_type(str(p))[0] or "image/png"
            return "data:" + mime + ";base64," + base64.b64encode(p.read_bytes()).decode()
        body = self.body("")
        body["messages"][0]["content"] = [dict(type="text", text=self.tag("image") + "Describe this image briefly."),
                                            dict(type="image_url", image_url=dict(url=data_url(path)))]
        self.exact_case(case, body)
        # Avoid embedding the fixture in the report.
        case["request"]["messages"][0]["content"][1]["image_url"]["url"] = "fixture:" + str(path)
        case["fixtures"] = [dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                            for p in (path, other)]
        changed = copy.deepcopy(body)
        changed["messages"][0]["content"][1]["image_url"]["url"] = data_url(other)
        result = self.send(changed)
        case["changed_image"] = result
        start = self.args.image_start_token
        case["metrics"].update(changed_image_cached_tokens=cached(result), image_start_token=start)
        if start is None:
            case["pass"] = False
            raise ValueError("HTTP does not expose image frontier; supply --image-start-token INDEX")
        case["pass"] = case["pass"] and cached(result) < start + 1

    def host(self):
        def run(case):
            case["stats_before"] = self.get("/v1/stats")
            body = self.body(self.tag("host_early") + " x" * 256 + "\nWrite a Python identity function.")
            first = self.send(body)
            case["cold"] = first
            fillers = []
            case["fillers"] = fillers
            for i in range(self.args.entries * 2 + 2):
                result = self.send(self.body(self.tag(f"host_fill_{i}") + " x" * 256 + "\nReply OK.", 8))
                fillers.append(result["usage"])
            case["stats_before_restore"] = self.get("/v1/stats")
            again = self.send(body)
            case["repeat"] = again
            case["stats_after"] = self.get("/v1/stats")
            counters = delta(case["stats_before_restore"], case["stats_after"])
            case["counter_deltas"] = counters
            metrics = {key: value for key, value in counters.items()
                       if ("host" in key or "prefix" in key) and ("restore" in key or "promot" in key)}
            identical = (first["content"], first["reasoning"]) == (again["content"], again["reasoning"])
            restore_counts = [v for k, v in metrics.items()
                              if k.endswith((".restores", ".host_restores", ".host_promotions", ".promotions"))]
            observed = any(v > 0 for v in restore_counts) if restore_counts else None
            metrics.update(cached_tokens=cached(again), byte_exact=identical, host_restore_observed=observed)
            case["metrics"] = metrics
            require = getattr(self.args, "require_host_path", False)
            verified = observed is True if require else observed is not False
            case["pass"] = (cached(again) == first["usage"]["prompt_tokens"] and identical and verified)
            case["host_path_verified"] = observed
            if observed is False:
                case["error"] = "full hit did not exercise host restore; check --entries and host cache budget"
            elif observed is None:
                case["note"] = "restore telemetry absent: exact hit passed, host tier unverified"
                if require:
                    case["error"] = "--require-host-path: no host restore/promotion counter in /v1/stats"
        self.case("host_restore", run)

    def admission(self):
        with self.console_viewer():
            for name, sizes in (("c16_code", [128] * 16), ("long_mix", [65536] * 4 + [128] * 8)):
                self.case(name, lambda case, sizes=sizes: self.admission_case(case, sizes))

    def admission_case(self, case, sizes):
        # Calibrate the repeated suffix using the server tokenizer, not chars/4.
        count = max(sizes)
        calibration = self.send(self.body(self.tag("calibration_" + case["name"]) + " x" * count, 1))
        measured = calibration["usage"]["prompt_tokens"]
        case["calibration"] = dict(repetitions=count, prompt_tokens=measured)
        bodies = []
        for i, target in enumerate(sizes):
            repeats = max(1, round(count * target / measured))
            body = self.body(self.tag(f"{case['name']}_{i}") + " x" * repeats +
                             "\nImplement a Python CSV parser with tests and explain the code.", 1024)
            bodies.append(body)
        samples, errors = [], []
        start = threading.Event()
        stop = threading.Event()
        def request(body):
            start.wait(self.args.timeout)
            return self.send(body)
        def sample():
            while not stop.is_set():
                item = dict(seconds=time.monotonic() - began)
                for key, path in (("stats", "/v1/stats"), ("console", "/v1/console/snapshot")):
                    try:
                        item[key] = self.get(path)
                    except Exception as error:
                        item[key + "_error"] = str(error)
                item.update(sample_metrics(item.get("stats", {}), item.get("console", {})))
                samples.append(item)
                stop.wait(1)
        before = self.get("/v1/stats")
        began = time.monotonic()
        sampler = threading.Thread(target=sample, daemon=True)
        results = []
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(bodies)) as pool:
                futures = [pool.submit(request, body) for body in bodies]
                sampler.start()
                start.set()
                for future in futures:
                    try:
                        results.append(future.result())
                    except Exception as error:
                        errors.append(str(error))
        finally:
            stop.set()
            if sampler.ident:
                sampler.join(timeout=25)
        after = self.get("/v1/stats")
        active = [s["admitted"] for s in samples if s["admitted"] is not None]
        util = [s["pool_utilization"] for s in samples if s["pool_utilization"] is not None]
        case.update(samples=samples, requests=results, errors=errors, stats_before=before,
                    stats_after=after, counter_deltas=delta(before, after))
        case["metrics"].update(admitted_max=max(active) if active else None,
            admitted_mean=statistics.mean(active) if active else None,
            pool_utilization_max=max(util) if util else None,
            pool_utilization_mean=statistics.mean(util) if util else None)
        for key, value in case["counter_deltas"].items():
            if key.startswith("admission."):
                case["metrics"][key] = value
        long_counts = [r["usage"]["prompt_tokens"] for r in results if r["usage"]["prompt_tokens"] > 32768]
        case["pass"] = (not errors and len(results) == len(bodies) and bool(active) and bool(util)
                        and (case["name"] != "long_mix" or
                             len(long_counts) == 4 and all(60000 <= n <= 72000 for n in long_counts)))

    def agentic(self):
        bench = runpy.run_path(str(ROOT / "scripts/bench/bench-agentic-session.py"))
        stream_turn = bench["stream_turn"]
        stream_turn.__globals__["open_request"] = lambda base, body, api_key=None: self.open_request(base, body)
        def bounded_turn(*args, **kwargs):
            result = stream_turn(*args, **kwargs)
            if not result["done"] or not result["finish_reason"] or not result["usage"]:
                raise ValueError("incomplete agentic SSE turn")
            return result
        bench["record_session"].__globals__["stream_turn"] = bounded_turn
        options = argparse.Namespace(model=self.args.model, base_url=self.args.base_url,
            api_key=self.args.api_key, label=self.args.nonce, max_turns=self.args.max_turns,
            max_tokens=1024, reasoning_effort="high", extra_body=None, echo_reasoning="on")
        for seed in range(self.args.sessions):
            def run(case, seed=seed):
                session = bench["record_session"](options, seed, 0)
                case["session"] = session
                turns = session["turns"]
                case["hit_tokens_per_turn"] = [t["hit_tokens"] for t in turns]
                case["metrics"]["hit_tokens_total"] = sum(t["hit_tokens"] or 0 for t in turns)
                for turn in turns:
                    case["metrics"][f"turn_{turn['turn']}_hit_tokens"] = turn["hit_tokens"]
                case["pass"] = bool(turns) and all(t["hit_tokens"] is not None and
                    t["finish_reason"] is not None and not t["invalid_tool_calls"] for t in turns)
            self.case(f"agentic_{seed}", run)
        self.report["hit_tokens_total"] = sum(c["metrics"].get("hit_tokens_total", 0) for c in self.report["cases"])


def sample_metrics(stats, console):
    totals = stats.get("totals") or {}
    gauges = console.get("g") or {}
    if "requests_admitted" in totals and "requests_retired" in totals:
        admitted = max(0, totals["requests_admitted"] - totals["requests_retired"])
    else:
        admitted = gauges.get("active")
    kv = gauges.get("kv") or stats.get("kv") or {}
    prefix = stats.get("prefix") or {}
    pages, free = kv.get("pages", prefix.get("pages")), kv.get("free", prefix.get("pages_free"))
    utilization = (pages - free) / pages if pages and free is not None else None
    return dict(admitted=admitted, pool_utilization=utilization, kv_pages_total=pages,
                kv_pages_free=free, kv_pages_active=kv.get("active"))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("exact", "host", "admission", "agentic"):
        s = sub.add_parser(name)
        s.add_argument("--base-url", required=True)
        s.add_argument("--output", type=Path, required=True)
        s.add_argument("--model", default=API["MODEL"])
        s.add_argument("--api-key")
        s.add_argument("--timeout", type=float, default=1800)
        s.add_argument("--nonce", default=uuid.uuid4().hex, help="use the same fresh namespace for both builds")
        s.add_argument("--reference-url")
        if name == "exact":
            s.add_argument("--odd-frontier", action="store_true", help="optional both-parity HTTP coverage")
            s.add_argument("--image", type=Path)
            s.add_argument("--different-image", type=Path)
            s.add_argument("--image-start-token", type=int)
        if name == "host":
            s.add_argument("--entries", type=int, default=20)
            s.add_argument("--require-host-path", action="store_true",
                           help="host: fail unless a restore/promotion counter in /v1/stats advanced")
        if name == "agentic":
            s.add_argument("--sessions", type=int, choices=(1, 2), default=2)
            s.add_argument("--max-turns", type=int, default=4)
    s = sub.add_parser("compare")
    s.add_argument("--candidate", required=True, type=Path)
    s.add_argument("--baseline", required=True, type=Path)
    s.add_argument("--output", type=Path)
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.command == "compare":
        result = compare_reports(json.loads(args.candidate.read_text()), json.loads(args.baseline.read_text()))
        print("Case / metric | before -> after")
        for row in result["rows"]:
            print(f"{row['case']} / {row['metric']} | {row['before']} -> {row['after']}")
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
        for problem in result["problems"]:
            print("FAIL: " + problem)
        return 0 if result["pass"] else 1
    if args.timeout <= 0:
        p.error("--timeout must be positive")
    if args.command == "host" and args.entries < 1:
        p.error("--entries must be positive")
    driver = Driver(args)
    driver.save()
    try:
        getattr(driver, args.command)()
        driver.report["pass"] = bool(driver.report["cases"]) and all(c["pass"] for c in driver.report["cases"])
    except Exception as error:
        driver.report["error"] = f"{type(error).__name__}: {error}"
    driver.save()
    return 0 if driver.report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
