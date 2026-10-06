#!/usr/bin/env python3
"""Pin the deterministic 64-window fidelity recipe using a checkpoint-precision arm.

Requires the authenticated POST /v1/bench/probe endpoint. No external teachers,
local assistant transcripts, or benchmark answers are read. Generation provenance
is an explicit --arm-manifest (checkpoint, head='bf16', activations/kv/state
='bf16' or 'family_native' or 'checkpoint', speculation=false, prefix_cache=false).
This is attestation, not hardware validation; fixed family-native formats stay native.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import pathlib
import sys
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python/reference"))
from fidelity_windows import MAX_TOKENS, SET_SCHEMA, bucket, canonical, set_hash, validate_set

LEGACY = {"deepseek_v41": "deepseek-v4.1-flash", "mimo_v2": "mimo-v2.6-pro",
          "qwen4": "qwen3.8-flash-next", "glm5_flash": "glm-5.3-flash",
          "deepseek_v4": "deepseek-v4-flash-0731", "glm5": "glm-5.3"}
REPO_FILES = ["rust/crates/cuteafd-api/src/openai/probe.rs", "scripts/bench/bench-agentic-session.py",
              "python/reference/families/deepseek_v41/golden.py", "rust/crates/cuteafd-bench/src/client.rs",
              "scripts/bench/make-fidelity-reference.py", "rust/crates/cuteafd-bench/src/reference.rs",
              "rust/crates/cuteafd-bench/src/panels/agentic.rs", "rust/crates/cuteafd-bench/src/panels/quality.rs",
              "rust/crates/cuteafd-daemon/src/shared/probe.rs", "native/shared/cuda/norm.cu"]
ADDED_TASKS = [
    "Add a focused regression test documenting the tie-breaking rule for negative half-cent scaling; use the tools to inspect and edit the fixture.",
    "Add a CSV regression test for an escaped quote in a description, and confirm the parser handles the case.",
    "Inspect the exchange-rate cache and add tests documenting behavior exactly at the expiry boundary.",
    "Add tests for leap-year February and a December-to-January month transition in the fixture date helpers.",
    "Use the fixture tools to explain how money arithmetic and reporting interact; identify one small safe improvement and test it.",
    "Inspect all fixture importers and propose a minimal patch preserving quoted fields; add tests and run the suite.",
]
MATH = ["Explain and prove why binary search terminates, including the integer boundary cases.",
        "Derive the variance of a paired difference of two Bernoulli outcomes.",
        "Prove an amortized bound for a dynamic array that doubles capacity.",
        "Explain the log-sum-exp identity and derive its numerically stable implementation."]
STRUCTURED = ["Invent a fictional person and describe them.",
              "Create an example e-commerce order with several line items.",
              "Classify this support message: 'My invoice shows a double charge for March and I need a refund before Friday.' Give category, priority, a summary and follow-up actions."]


def load_agentic():
    spec = importlib.util.spec_from_file_location("fidelity_agentic", ROOT / "scripts/bench/bench-agentic-session.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class BucketMismatch(ValueError):
    pass


class ContextCap(ValueError):
    pass


class ProbeClient:
    def __init__(self, base: str, token: str | None, timeout: float):
        self.url, self.token, self.timeout = base.rstrip("/") + "/v1/bench/probe", token, timeout

    def __call__(self, body: dict) -> dict:
        spec = {"cold": True, "no_speculation": True, "record_rows": 0, "top_k": 1}
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["x-cuteafd-bench"] = self.token
            headers["Authorization"] = "Bearer " + self.token
        request = urllib.request.Request(self.url, canonical({"body": body, "spec": spec}), headers)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                chat = json.load(response)
        except urllib.error.HTTPError as error:
            detail = error.read(4096).decode("utf-8", errors="replace")
            if error.code == 400 and "prompt of " in detail and "outside 1..16384" in detail:
                raise ContextCap(f"generation probe HTTP {error.code}: {detail}") from error
            raise ValueError(f"generation probe HTTP {error.code}: {detail}") from error
        record = chat.get("probe")
        if not record or record.get("error") or not record.get("engine"):
            detail = record.get("error") if record else "missing probe record"
            raise ValueError(f"server did not honor the generation probe: {detail}")
        if not record.get("cold") or not record.get("no_speculation") or record.get("cached_tokens", 0):
            raise ValueError("generation must be cold, speculation-free")
        if not record.get("prompt_ids") or not record.get("generated"):
            raise ValueError("probe must return prompt_ids and generated token IDs")
        return chat


def arm_policy(arm: dict, checkpoint: str) -> None:
    if arm.get("checkpoint") != checkpoint:
        raise ValueError("arm checkpoint differs from requested checkpoint")
    if arm.get("head") != "bf16" or any(arm.get(k) not in ("bf16", "family_native", "checkpoint") for k in ("activations", "kv", "state")):
        raise ValueError("generation arm requires a BF16 head and checkpoint-precision/native activations/kv/state")
    if arm.get("speculation") is not False or arm.get("prefix_cache") is not False:
        raise ValueError("generation arm must disable speculation and prefix cache")


def source(root: pathlib.Path, path: str) -> dict:
    resolved = (root / path).resolve()
    if pathlib.Path(path).is_absolute() or not resolved.is_relative_to(root.resolve()):
        raise ValueError("source paths must stay inside the repository")
    data = resolved.read_bytes()
    return {"path": path, "sha256": hashlib.sha256(data).hexdigest()}


def legacy_window(reference: dict) -> dict:
    if reference.get("schema") == "cuteafd.fidelity.reference/2":
        matches = [w for w in reference.get("windows", []) if w.get("id") == "legacy"]
        if len(matches) != 1:
            raise ValueError("schema 2 reference needs exactly one legacy window")
        legacy = matches[0]
    elif reference.get("schema") == "cuteafd.bench.reference/1":
        legacy = reference
    else:
        raise ValueError("unsupported legacy reference schema")
    start, tokens = legacy.get("score_from"), legacy.get("tokens")
    if type(start) is not int or start < 1 or not isinstance(tokens, list):
        raise ValueError("invalid legacy reference tokens/score_from")
    end = start + 512
    if len(tokens) < end or end > MAX_TOKENS:
        raise ValueError("legacy reference has fewer than 512 positions or exceeds the 16K cap")
    if any(type(t) is not int or not 0 <= t <= 2147483647 for t in tokens[:end]):
        raise ValueError("invalid legacy reference token id")
    return {"id": "legacy", "block": "E", "bucket": bucket(start),
            "tokens": tokens[:end], "roles": ["ctx"] * end, "score_from": start}


def build_set(*, family: str, model: str, checkpoint: str, version: str, arm: dict,
              tokenizer, probe, root: pathlib.Path = ROOT, recordings: list[dict] | None = None,
              files: list[str] | None = None, legacy_reference: dict | None = None,
              legacy_provenance: dict | None = None) -> dict:
    arm_policy(arm, checkpoint)
    agentic = load_agentic()
    files = files or REPO_FILES
    sources = [source(root, path) for path in files]
    corpus = "\n\n".join(f"File: {path}\n{(root / path).read_text()}" for path in files)
    corpus_ids = tokenizer.encode(corpus, add_special_tokens=False).ids
    if len(corpus_ids) < 4096:
        raise ValueError("repository source list needs at least 4096 tokens")
    if (legacy_reference is None) != (legacy_provenance is None):
        raise ValueError("explicit legacy reference requires its provenance")
    if legacy_reference is None:
        legacy_path = root / "rust/crates/cuteafd-bench/references" / (LEGACY[family] + ".json")
        legacy_reference = json.loads(legacy_path.read_text())
        legacy_provenance = source(root, str(legacy_path.relative_to(root)))
    elif not isinstance(legacy_provenance, dict) or not legacy_provenance.get("sha256"):
        raise ValueError("explicit legacy reference requires a content hash")
    legacy = legacy_window(legacy_reference)
    windows = [{**legacy, "provenance": legacy_provenance}]
    snapshots = []
    for recording in recordings or []:
        if recording.get("model") != model or recording.get("mode") != "record":
            raise ValueError("recordings must be our record runner's output for the selected model")
        for session in recording["sessions"]:
            for turn in session["turns"]:
                snapshots.append((session["messages"][:turn["messages"]], recording.get("tools", agentic.TOOLS)))

    def generate(name, block, messages, target, tools=None, extra=None):
        # Size by rendered IDs, never truncate a chat template or original history.
        base = copy.deepcopy(messages)
        base.insert(0, {"role": "system", "content": f"Fidelity fixture {name}; think carefully and use the repository tools when appropriate."})
        count = max(0, target * 3 - len(canonical(base)))
        for attempt in range(8):
            body = {"model": model, "messages": copy.deepcopy(base), "temperature": 0,
                    "seed": 0, "max_tokens": 1, "reasoning_effort": "high", "stream": False}
            if count:
                body["messages"].insert(0, {"role": "system", "content": "Repository context:\n" + (corpus * (count // len(corpus) + 1))[:count]})
            if tools:
                body.update(tools=tools, tool_choice="auto")
            body.update(extra or {})
            leading = []
            while body["messages"] and body["messages"][0]["role"] == "system":
                leading.append(body["messages"].pop(0)["content"])
            if leading:
                body["messages"].insert(0, {"role": "system", "content": "\n\n".join(leading)})
            sample = probe(body)
            n = len(sample["probe"]["prompt_ids"])
            if target <= n < target + 384:
                break
            if not count and n >= target:
                if n + 1024 > MAX_TOKENS:
                    raise ContextCap(f"{name}: original history exceeds 16K cap")
                break
            count = max(0, count + (target + 96 - n) * 3)
        else:
            raise ValueError(f"{name}: could not size rendered prompt to target {target}")
        if n + 1024 > MAX_TOKENS:
            raise ContextCap(f"{name}: rendered prompt plus generation exceeds 16K cap")
        body["max_tokens"] = 1024
        chat = probe(body)
        server = chat.get("server", {})
        if server.get("model", checkpoint) != checkpoint or server.get("family", family) != family:
            raise ValueError("probe server checkpoint/family differs from generation manifest")
        settings = server.get("settings", [])
        if family == "deepseek_v41" and isinstance(settings, list):
            for setting in settings:
                if setting.get("name") == "CUTEAFD_V41_FP8_HEAD" and str(setting.get("value")).lower() not in ("off", "false", "0", "none", "bf16"):
                    raise ValueError("server reports an enabled FP8 head, not the checkpoint-precision arm")
        prompt, generated = chat["probe"]["prompt_ids"], chat["probe"]["generated"]
        if prompt != sample["probe"]["prompt_ids"] or len(generated) > 1024:
            raise ValueError("unstable prompt rendering or max_tokens was not honored")
        tokens = prompt + generated
        roles = ["ctx"] * len(prompt) + ["gen"] * len(generated)
        start = len(prompt) + max(0, len(generated) - 512)
        # Short turns remain short: trailing human code supplies ctx rows, not
        # fabricated assistant output. These rows never carry the gen bar.
        padding = max(0, 512 - len(generated))
        tokens += corpus_ids[:padding]
        roles += ["ctx"] * padding
        expected_bucket = "0-2K" if target < 2048 else "2-8K" if target < 8192 else "8-16K"
        if block in ("A", "B") and bucket(start) != expected_bucket:
            raise BucketMismatch(f"{name}: history lands in {bucket(start)}, expected {expected_bucket}; supply a shorter recording/history")
        windows.append({"id": name, "block": block, "bucket": bucket(start),
                        "tokens": tokens, "roles": roles, "score_from": start,
                        "provenance": {"prompt_sha256": hashlib.sha256(canonical(body)).hexdigest(),
                                       "engine": chat["probe"]["engine"], "server": server,
                                       "generated_tokens": len(generated),
                                       "trailing_ctx": padding}})
        return chat

    def fresh_task(i):
        workspace = agentic.Workspace()
        tasks = [task["prompt"] for task in agentic.TASKS] + ADDED_TASKS
        if i < 8:
            # Tool schemas also consume tokens: the short stratum uses the same
            # fixture instructions and tree, without the bundled README/docs.
            system = agentic.SYSTEM_TEMPLATE.split("Repository layout:")[0].format(nonce="fidelity")
            system += "Repository layout:\n" + agentic.tree(workspace.files)
        else:
            system = agentic.system_prompt(workspace.files)
        return workspace, [{"role": "system", "content": system},
                           {"role": "user", "content": tasks[i % len(tasks)]}]

    workspace, messages = None, None
    for i in range(26):
        if snapshots:
            messages, tools = snapshots[i % len(snapshots)]
        else:
            if messages is None:
                workspace, messages = fresh_task(i)
            tools = agentic.TOOLS
        target = 1024 if i < 8 else 4096 if i < 20 else 10240
        try:
            chat = generate(f"a{i:02d}", "A", messages, target, tools)
        except (BucketMismatch, ContextCap) as error:
            if isinstance(error, ContextCap) and windows:
                windows[-1]["provenance"]["ended_at_context_cap"] = True
            if snapshots:
                raise
            # A growing tool history can leave the scheduled length stratum.
            # Start the next deterministic fixture task instead of truncating it.
            workspace, messages = fresh_task(i)
            chat = generate(f"a{i:02d}", "A", messages, target, tools)
        if not snapshots:
            malformed = 0
            for recovery in range(3):
                calls = chat.get("tool_calls", [])
                assistant = {"role": "assistant", "content": chat.get("content", ""),
                             "reasoning_content": chat.get("reasoning", "")}
                if not calls:
                    messages = None
                    break
                assistant["tool_calls"] = calls
                messages.append(assistant)
                errors = 0
                for call in calls:
                    output, error = workspace.execute(call["function"]["name"], call["function"]["arguments"])
                    errors += int(error is not None)
                    messages.append({"role": "tool", "tool_call_id": call["id"], "content": output})
                malformed += errors
                if not errors:
                    break
                if recovery == 2:
                    # Retain the final actual model turn, but do not carry a stuck
                    # invalid-call loop into the next scheduled fixture window.
                    messages = None
                    break
                prior = windows.pop()
                try:
                    chat = generate(f"a{i:02d}", "A", messages, target, tools)
                except (BucketMismatch, ContextCap) as error:
                    if isinstance(error, ContextCap):
                        prior["provenance"]["ended_at_context_cap"] = True
                    # Error history cannot be truncated to force a length stratum.
                    windows.append(prior)
                    messages = None
                    break
            windows[-1]["provenance"]["malformed_tool_calls"] = malformed
            windows[-1]["provenance"]["tool_error_retries"] = recovery
        windows[-1]["provenance"].setdefault("ended_at_context_cap", False)
    for i in range(12):
        generate(f"b{i:02d}", "B", [{"role": "user", "content": f"Explain invariants and propose a small safe modification to {files[i % len(files)]}.\n" + (root / files[i % len(files)]).read_text()}], 10240 + (i % 3) * 1024)
    for i in range(10):
        ids = tokenizer.encode((root / files[i % len(files)]).read_text(), add_special_tokens=False).ids
        if len(ids) < 1024:
            raise ValueError(f"{files[i % len(files)]}: plain code needs >=1024 tokens; choose a longer file")
        start = min(len(ids) - 512, 512 + (i % 4) * 512)
        tokens = ids[:start + 512]
        windows.append({"id": f"c{i:02d}", "block": "C", "bucket": bucket(start), "tokens": tokens,
                        "roles": ["ctx"] * len(tokens), "score_from": start, "provenance": sources[i % len(files)]})
    for i in range(6):
        prompt = STRUCTURED[i] + " Reply in JSON." if i < 3 else "Select and call the correct tool to read, search or edit the fixture repository. " + agentic.TASKS[(i - 3) % len(agentic.TASKS)]["prompt"]
        generate(f"d{i:02d}", "D", [{"role": "user", "content": prompt}], 1024 + (i % 3) * 512,
                 agentic.TOOLS if i >= 3 else None,
                 {"response_format": {"type": "json_object"}} if i < 3 else None)
    for i, prompt in enumerate(MATH + [f"Explain the repository design in {files[j]} in detail." for j in range(5)]):
        generate(f"e{i:02d}", "E", [{"role": "user", "content": prompt}], 768 + (i % 3) * 512)
    quick = ["legacy", "a00", "a04", "a08", "a12", "a20", "a24", "b00", "b08", "c00", "d00", "e00"]
    servers = [w["provenance"]["server"] for w in windows if "server" in w["provenance"]]
    if servers and any(server != servers[0] for server in servers):
        raise ValueError("server identity/settings changed during set generation")
    manifest = {"schema": SET_SCHEMA, "family": family, "model": model, "checkpoint": checkpoint,
                "version": version, "quick_windows": quick, "generation_arm": arm,
                "generation_server": servers[0] if servers else {}, "precision_policy": "attested; recognized server head setting cross-checked if present",
                "builder_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
                "sources": sources + [source(root, "scripts/bench/bench-agentic-session.py"),
                    source(root, "rust/crates/cuteafd-bench/src/panels/quality.rs")],
                "recording_sha256": [hashlib.sha256(canonical(r)).hexdigest() for r in recordings or []],
                "recipe": {"windows": 64, "positions_per_window": 512, "max_tokens": 1024,
                    "context_cap": MAX_TOKENS, "temperature": 0, "seed": 0, "reasoning_effort": "high",
                    "blocks": {"A": 26, "B": 12, "C": 10, "D": 6, "E": 10},
                    "prompt_roles": "conditioning history ctx; current assistant span gen"}, "windows": windows}
    manifest["set_sha256"] = set_hash(manifest)
    validate_set(manifest)
    return manifest


def build_vision_set(*, family, model, checkpoint, version, arm, probe, fixtures, tokenizer_sha256):
    """Independent media panel: never rewrite a frozen 64-window text publication."""
    import base64
    from fidelity_media import read_fixture, validate_media
    if family not in ("mimo_v2", "qwen4", "glm5_flash"):
        raise ValueError("this family has no supported v2 vision tower")
    arm_policy(arm, checkpoint)
    fixture_manifest = json.loads((fixtures / "fixtures.json").read_text())
    records = fixture_manifest.get("fixtures", [])
    if fixture_manifest.get("schema") != "cuteafd.media.fixtures/1" or len(records) != 8:
        raise ValueError("vision bucket needs eight pinned first-party fixtures")
    coding_tasks = {
        "code": "Transcribe every visible line of code, preserving indentation. Explain each visible "
            "function, input, branch and invariant line by line. Then write a standalone Python "
            "test module with at least eight distinct test cases covering the visible behavior. "
            "Explain each test's purpose and expected result. Mark cropped or unreadable text "
            "as unknown rather than inventing the missing source.",
        "terminal": "Transcribe every visible terminal line exactly. Explain each command and "
            "reported count or exit status. Then write a standalone Python parser for this "
            "transcript and a unittest module with at least eight distinct cases, including "
            "successful runs, failures, malformed lines and contradictory counts. Explain the "
            "expected result of every test; label invented test inputs as synthetic.",
        "chart": "Describe every visible chart element and transcribe all labels and numeric "
            "values. Compute the total and compare the batches. Then write a complete Python "
            "matplotlib program reproducing the visible chart, with comments explaining the "
            "layout and data. Add at least eight unittest cases validating the transcribed "
            "data and calculations, and explain each expected result. Do not invent hidden values.",
        "diagram": "Transcribe every visible node label and describe every arrow in order. "
            "Then write a complete Python program reproducing the diagram and a validator for "
            "the represented pipeline. Add at least eight unittest cases for stage order, "
            "connectivity, missing stages and cycles, explaining each expected result. Label "
            "hypothetical invalid pipelines as synthetic; do not infer unseen stages.",
    }
    windows, servers = [], []
    for item in records:
        fixture = {"path": item["path"], "sha256": item["sha256"]}
        data = read_fixture(fixtures, {"fixture": fixture})
        body = {"model": model, "messages": [{"role": "system", "content": "Inspect the image carefully. Explain only what is visible."},
            {"role": "user", "content": [{"type": "image_url", "image_url": {
                "url": "data:image/png;base64," + base64.b64encode(data).decode(), "detail": "auto"}},
                {"type": "text", "text": item["question"] + "\n\n" + coding_tasks[item["kind"]]
                    + "\nGive a detailed answer of at least 800 words, including complete code "
                    "and tests rather than placeholders. Separate observed facts from proposed code."}]}],
            "max_tokens": 2048, "temperature": 0, "seed": 0, "reasoning_effort": "high", "stream": False}
        chat = probe(body)
        record, server = chat["probe"], chat.get("server", {})
        if server.get("model") != checkpoint or server.get("family") != family:
            raise ValueError("media generation server identity differs from manifest")
        spans = record.get("media")
        if not isinstance(spans, list) or len(spans) != 1:
            raise ValueError("media probe must return one prepared image span; refusing text-only generation")
        span = {k: spans[0][k] for k in ("start", "len", "kind", "key", "grid")}
        span["fixture"] = fixture
        prompt, generated = record["prompt_ids"], record["generated"]
        if not 576 <= len(generated) <= body["max_tokens"]:
            raise ValueError("vision golden needs at least 576 real generated tokens for prefix qualification")
        tokens, roles = prompt + generated, ["ctx"] * len(prompt) + ["gen"] * len(generated)
        start = len(tokens) - 512
        validate_media([span], tokens, roles, start)
        if len(tokens) > MAX_TOKENS:
            raise ValueError("vision window exceeds context cap")
        windows.append({"id": item["id"], "block": "vision", "bucket": bucket(start),
            "tokens": tokens, "roles": roles, "score_from": start, "media": [span],
            "provenance": {"fixture_manifest_sha256": hashlib.sha256(canonical(fixture_manifest)).hexdigest(),
                "prompt_sha256": hashlib.sha256(canonical(body)).hexdigest(), "engine": record["engine"],
                "generated_tokens": len(generated), "server": server}})
        servers.append(server)
    if any(s != servers[0] for s in servers):
        raise ValueError("server changed during media generation")
    manifest = {"schema": SET_SCHEMA, "family": family, "model": model, "checkpoint": checkpoint,
        "version": version, "generation_arm": arm, "generation_server": servers[0],
        "tokenizer_sha256": tokenizer_sha256, "quick_windows": fixture_manifest["quick_windows"],
        "fixtures_sha256": hashlib.sha256((fixtures / "fixtures.json").read_bytes()).hexdigest(),
        "recipe": {"windows": 8, "positions_per_window": 512, "quick_windows": 2,
                   "prompt_roles": "image/text ctx; actual assistant output gen"}, "windows": windows}
    manifest["set_sha256"] = set_hash(manifest)
    return validate_set(manifest)


def seal_qsa_geometry(manifest, requested=None):
    if manifest["family"] != "qwen4":
        if requested is not None:
            raise ValueError("--qsa-key-rows requires the qwen4 family")
        return
    from qwen_media import QSA_KEY_ROWS, qsa_key_rows
    media = any(w.get("media") for w in manifest["windows"])
    observed = max(len(w["tokens"]) for w in manifest["windows"])
    default = QSA_KEY_ROWS if media else max(128, 1 << (observed - 1).bit_length())
    manifest["qsa_key_rows"] = qsa_key_rows(manifest, default if requested is None else requested)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--family", required=True, choices=LEGACY)
    p.add_argument("--version", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tokenizer", required=True, type=pathlib.Path)
    p.add_argument("--arm-manifest", required=True, type=pathlib.Path)
    p.add_argument("--recording", action="append", type=pathlib.Path, default=[])
    p.add_argument("--media-fixtures", type=pathlib.Path, help="build a separate 8-window vision panel from fixtures.json")
    p.add_argument("--qsa-key-rows", type=int, help="Qwen fixed QSA extent sealed into this set (media default 2560; text next power of two)")
    p.add_argument("--file-list", type=pathlib.Path, help="JSON array of first-party repo-relative paths")
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--bench-token")
    p.add_argument("--timeout", type=float, default=1800)
    p.add_argument("--out", type=pathlib.Path)
    a = p.parse_args(argv)
    if not a.version or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in a.version):
        p.error("version must be a safe directory name")
    if a.file_list:
        import subprocess
        files = json.loads(a.file_list.read_text())
        tracked = set(subprocess.check_output(["git", "-C", str(ROOT), "ls-files"], text=True).splitlines())
        if any(path not in tracked or pathlib.Path(path).suffix not in (".rs", ".py", ".cu", ".cuh") for path in files):
            p.error("--file-list may contain only tracked first-party Rust, Python or CUDA files")
        if any(path.startswith("third_party/") for path in files):
            p.error("third-party sources are not part of the fidelity recipe")
    from tokenizers import Tokenizer
    if a.media_fixtures:
        if a.recording or a.file_list:
            p.error("--media-fixtures cannot combine with text recipe recordings/files")
        manifest = build_vision_set(family=a.family, model=a.model, checkpoint=a.checkpoint, version=a.version,
            arm=json.loads(a.arm_manifest.read_text()), probe=ProbeClient(a.base_url, a.bench_token, a.timeout),
            fixtures=a.media_fixtures, tokenizer_sha256=hashlib.sha256(a.tokenizer.read_bytes()).hexdigest())
    else:
        manifest = build_set(family=a.family, model=a.model, checkpoint=a.checkpoint, version=a.version,
            arm=json.loads(a.arm_manifest.read_text()), tokenizer=Tokenizer.from_file(str(a.tokenizer)),
            probe=ProbeClient(a.base_url, a.bench_token, a.timeout),
            recordings=[json.loads(path.read_text()) for path in a.recording],
            files=json.loads(a.file_list.read_text()) if a.file_list else None)
    seal_qsa_geometry(manifest, a.qsa_key_rows)
    manifest["tokenizer_sha256"] = hashlib.sha256(a.tokenizer.read_bytes()).hexdigest()
    manifest["set_sha256"] = set_hash(manifest)
    out = a.out or ROOT / "set" / a.family / a.version / "windows.json"
    if out.exists():
        p.error("set versions are immutable; choose a new --version/output")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(canonical(manifest) + b"\n")
    count = len(manifest["windows"])
    print(f"{out}: {count} windows, {count * 512} rows, sha256 {manifest['set_sha256']}")


if __name__ == "__main__":
    main()
