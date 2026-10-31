"""Fidelity recipe and probe contract tests; no CUDA or external data."""
import collections
import importlib.util
import json
import pathlib
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("fidelity_set_builder", ROOT / "scripts/bench/fidelity-set.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return SimpleNamespace(ids=[(ord(text[i]) % 61) + 1 for i in range(0, len(text), 3)])


@pytest.fixture
def fake_server():
    state = {"requests": [], "mode": "good", "generated": 600, "family": "deepseek_v41", "tool_once": False, "called": False, "malformed_remaining": 0}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            assert self.path == "/v1/bench/probe"
            assert self.headers["x-cuteafd-bench"] == "test-token"
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state["requests"].append(request)
            body, probe_spec = request["body"], request["spec"]
            assert probe_spec["cold"] and probe_spec["no_speculation"]
            assert body["temperature"] == 0 and body["reasoning_effort"] == "high"
            assert body["messages"][0]["role"] == "system"
            assert all(m["role"] != "system" for m in body["messages"][1:])
            if state["mode"] == "http_error":
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b'{"error":"native library not found"}')
                return
            if state["mode"] == "context_cap" and any(m.get("role") == "tool" for m in body["messages"]):
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b'{"error":{"message":"prompt of 16520 tokens is outside 1..16384"}}')
                return
            text = json.dumps(body["messages"], sort_keys=True)
            ids = FakeTokenizer().encode(text).ids
            probe = {"engine": "fake-bf16", "cold": True, "no_speculation": True, "cached_tokens": 0,
                     "prompt_ids": ids, "generated": [4] * min(body["max_tokens"], state["generated"])}
            if state["mode"] == "warm":
                probe["cold"] = False
            if state["mode"] == "missing":
                probe["generated"] = []
            if state["mode"] == "error":
                probe["error"] = "unsupported probe"
            response = {"probe": probe, "content": "a fixture answer", "reasoning": "because", "tool_calls": [],
                        "server": {"model": "test-model", "family": state["family"], "snapshot": "pinned-revision"}}
            if state["tool_once"] and not state["called"] and body["max_tokens"] == 1024 and body.get("tools"):
                response["tool_calls"] = [{"id": "fixture_call", "type": "function",
                    "function": {"name": "list_dir", "arguments": "{\"path\":\".\"}"}}]
                state["called"] = True
            if state["malformed_remaining"] and body["max_tokens"] == 1024 and body.get("tools"):
                response["tool_calls"] = [{"id": "invalid_call", "type": "function",
                    "function": {"name": "list_dir", "arguments": "{}"}}]
                state["malformed_remaining"] -= 1
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state["client"] = builder.ProbeClient(f"http://127.0.0.1:{server.server_port}", "test-token", 10)
    yield state
    server.shutdown()
    server.server_close()
    thread.join()


def policy():
    return {"checkpoint": "test-model", "snapshot_revision": "pinned-revision", "activations": "bf16",
            "head": "bf16", "kv": "bf16", "state": "bf16", "speculation": False, "prefix_cache": False}


def make(state, **kwargs):
    if "legacy_reference" not in kwargs and "legacy_provenance" not in kwargs:
        kwargs.update(legacy_reference={"schema": "cuteafd.bench.reference/1", "score_from": 64,
                                      "tokens": list(range(576))},
                      legacy_provenance={"path": "synthetic-test-anchor", "sha256": "b" * 64})
    return builder.build_set(family=state["family"], model="test-model", checkpoint="test-model",
        version="test1", arm=policy(), tokenizer=FakeTokenizer(), probe=state["client"], **kwargs)


def test_new_backend_sizes_locally_and_records_reference_snapshot(fake_server):
    client = fake_server['client']
    class LocalBackend:
        identity = {'text_source': 'hf'}
        def prompt(self, body):
            return {'probe': {'prompt_ids': FakeTokenizer().encode(json.dumps(body['messages'], sort_keys=True)).ids}}
        def __call__(self, body):
            assert body['max_tokens'] == 1024
            result = client(body)
            result['generation_provenance'] = {'text_source': 'hf', 'content': 'a fixture answer'}
            return result
    arm = {**policy(), 'tokenizer_sha256': 'a' * 64, 'config_sha256': 'b' * 64}
    manifest = builder.build_set(family='deepseek_v41', model='test-model', checkpoint='test-model',
        version='test', arm=arm, tokenizer=FakeTokenizer(), probe=LocalBackend(),
        legacy_reference={'schema': 'cuteafd.fidelity.set/1', 'windows': [
            {'id': 'legacy', 'score_from': 64, 'tokens': list(range(576))}]},
        legacy_provenance={'path': 'legacy.json', 'sha256': 'c' * 64})
    assert len(fake_server['requests']) == 53
    assert manifest['reference_snapshot']['tokenizer_sha256'] == 'a' * 64
    assert manifest['windows'][1]['provenance']['generation']['text_source'] == 'hf'


def test_rebind_text_set_checks_published_hash_and_used_tokens(tmp_path):
    import hashlib
    from tokenizers import Tokenizer, models
    from fidelity_windows import canonical, set_hash
    tokenizer = Tokenizer(models.WordLevel({"a": 0, "b": 1}, unk_token="a"))
    old, new = tmp_path / "old.json", tmp_path / "new.json"
    tokenizer.save(str(old))
    tokenizer.save(str(new))
    manifest = {"schema": "cuteafd.fidelity.set/1", "family": "qwen4", "model": "source",
                "checkpoint": "source", "generation_arm": {"checkpoint": "source"},
                "quick_windows": ["w"], "windows": [{"id": "w", "block": "E", "bucket": "0-2K",
                    "tokens": [0] * 576, "roles": ["ctx"] * 576, "score_from": 64}]}
    manifest["set_sha256"] = set_hash(manifest)
    source = tmp_path / "windows.json"
    source.write_bytes(canonical(manifest))
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    kwargs = dict(family="qwen3_5", model="Qwen/Qwen3.5-9B", version="test", arm={}, source_sha256=digest)
    result = builder.rebind_text_set(source, old, new, **kwargs)
    assert result["windows"] == manifest["windows"]
    assert result["generation_arm"] == manifest["generation_arm"]
    assert result["pinned_text_source"]["used_ids"] == 1
    assert result["family"] == "qwen3_5"
    with pytest.raises(ValueError, match="source hash"):
        builder.rebind_text_set(source, old, new, **{**kwargs, "source_sha256": "0" * 64})
    pinned = {**manifest, "tokenizer_sha256": "0" * 64}
    source.write_bytes(canonical(pinned))
    with pytest.raises(ValueError, match="source tokenizer"):
        builder.rebind_text_set(source, old, new, **{**kwargs,
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
    source.write_bytes(canonical(manifest))
    Tokenizer(models.WordLevel({"a": 1, "b": 0}, unk_token="a")).save(str(new))
    with pytest.raises(ValueError, match="token ids differ"):
        builder.rebind_text_set(source, old, new, **kwargs)


def test_64_window_recipe_is_deterministic_and_hashed(fake_server):
    manifest = make(fake_server)
    again = make(fake_server)
    assert manifest == again
    assert manifest["set_sha256"] == builder.set_hash(manifest)
    assert collections.Counter(w["block"] for w in manifest["windows"]) == {"A": 26, "B": 12, "C": 10, "D": 6, "E": 10}
    assert len(manifest["quick_windows"]) == 12
    assert sum(len(w["tokens"]) - w["score_from"] for w in manifest["windows"]) == 32768
    for window in manifest["windows"]:
        assert len(window["roles"]) == len(window["tokens"]) <= 16384
        assert window["bucket"] == builder.bucket(window["score_from"])
        if window["block"] == "C" or window["id"] == "legacy":
            assert set(window["roles"]) == {"ctx"}
        else:
            assert window["roles"][window["score_from"]:] == ["gen"] * 512
    a_buckets = collections.Counter(w["bucket"] for w in manifest["windows"] if w["block"] == "A")
    assert a_buckets == {"0-2K": 8, "2-8K": 12, "8-16K": 6}
    assert {w["bucket"] for w in manifest["windows"] if w["block"] == "B"} == {"8-16K"}
    assert all(len(s["sha256"]) == 64 for s in manifest["sources"])
    changed = json.loads(json.dumps(manifest))
    changed["windows"][0]["tokens"][0] += 1
    with pytest.raises(ValueError, match="hash mismatch"):
        builder.validate_set(changed)


def test_text_recipe_requires_explicit_sealed_anchor(capsys):
    with pytest.raises(SystemExit) as error:
        builder.main(["--family", "qwen4", "--version", "test", "--model", "test",
                      "--checkpoint", "test", "--tokenizer", "unused", "--arm-manifest", "unused"])
    assert error.value.code == 2
    assert "requires --legacy-reference" in capsys.readouterr().err


def test_explicit_legacy_reference_keeps_variant_tokens_and_provenance(fake_server):
    reference = {"schema": "cuteafd.bench.reference/1", "score_from": 1, "tokens": [7] * 513}
    provenance = {"path": "official-mopd-smoke.json", "sha256": "a" * 64}
    manifest = make(fake_server, legacy_reference=reference, legacy_provenance=provenance)
    assert manifest["windows"][0]["tokens"] == [7] * 513
    assert manifest["windows"][0]["provenance"] == provenance
    with pytest.raises(ValueError, match="requires its provenance"):
        make(fake_server, legacy_reference=reference)
    with pytest.raises(ValueError, match="content hash"):
        make(fake_server, legacy_reference=reference, legacy_provenance={})


@pytest.mark.parametrize("family", ["mimo_v2", "qwen4", "glm5_flash", "deepseek_v4", "glm5"])
def test_other_family_recipes_keep_internal_names(fake_server, family):
    fake_server["family"] = family
    manifest = make(fake_server)
    assert manifest["family"] == family
    assert len(manifest["windows"]) == 64


def test_live_fixture_tools_execute_and_reasoning_is_echoed(fake_server):
    fake_server["tool_once"] = True
    manifest = make(fake_server)
    assert len(manifest["windows"]) == 64
    histories = [request["body"]["messages"] for request in fake_server["requests"]]
    continued = next(messages for messages in histories if any(m.get("role") == "tool" for m in messages))
    tool = next(m for m in continued if m.get("role") == "tool")
    assistant = next(m for m in continued if m.get("role") == "assistant")
    assert tool["tool_call_id"] == "fixture_call" and "README.md" in tool["content"]
    assert assistant["reasoning_content"] == "because"


@pytest.mark.parametrize("invalid_calls, retries", [(1, 1), (3, 2)])
def test_malformed_tool_call_recovers_with_bounded_error_turns(fake_server, invalid_calls, retries):
    fake_server["malformed_remaining"] = invalid_calls
    manifest = make(fake_server)
    assert len(manifest["windows"]) == 64
    window = next(w for w in manifest["windows"] if w["id"] == "a00")
    assert window["provenance"]["malformed_tool_calls"] == invalid_calls
    assert window["provenance"]["tool_error_retries"] == retries
    histories = [request["body"]["messages"] for request in fake_server["requests"]]
    errors = [m for history in histories for m in history if m["role"] == "tool"]
    assert any("path" in m["content"] and "error" in m["content"] for m in errors)
    assert fake_server["malformed_remaining"] == 0


def test_context_cap_keeps_last_complete_turn_and_restarts_without_truncation(fake_server):
    fake_server["mode"] = "context_cap"
    fake_server["tool_once"] = True
    manifest = make(fake_server)
    first = next(w for w in manifest["windows"] if w["id"] == "a00")
    assert first["provenance"]["ended_at_context_cap"]
    assert len(manifest["windows"]) == 64
    assert first["provenance"]["generated_tokens"] == 600
    assert all(len(w["tokens"]) <= builder.MAX_TOKENS for w in manifest["windows"])


def test_short_assistant_turns_get_context_padding_not_fake_gen(fake_server):
    fake_server["generated"] = 73
    manifest = make(fake_server)
    for w in manifest["windows"]:
        if w["id"].startswith(("a", "b", "d", "e")):
            assert w["roles"][w["score_from"]:] == ["gen"] * 73 + ["ctx"] * 439
            assert w["provenance"]["trailing_ctx"] == 439


@pytest.mark.parametrize("mode", ["warm", "missing", "error"])
def test_probe_fails_closed(fake_server, mode):
    fake_server["mode"] = mode
    with pytest.raises(ValueError):
        make(fake_server)


def test_probe_reports_server_rejection(fake_server):
    fake_server["mode"] = "http_error"
    with pytest.raises(ValueError, match="HTTP 400: .*native library not found"):
        make(fake_server)


def test_legacy_recipe_survives_schema2_reference_migration():
    legacy = {"tokens": list(range(640)), "score_from": 64}
    before = builder.legacy_window({"schema": "cuteafd.bench.reference/1", **legacy})
    reference = {"schema": "cuteafd.fidelity.reference/2", "windows": [
        {"id": "a00", "tokens": [9] * 1024, "score_from": 512},
        {"id": "legacy", **legacy}], "tokens": [0], "score_from": 1}
    assert builder.legacy_window(reference) == before
    assert before["tokens"] == legacy["tokens"][:576]
    assert before["roles"] == ["ctx"] * 576
    assert legacy["tokens"] == list(range(640))


@pytest.mark.parametrize("change", ["missing", "duplicate", "schema", "start", "short", "token"])
def test_migrated_legacy_reference_fails_closed(change):
    window = {"id": "legacy", "tokens": [3] * 576, "score_from": 64}
    reference = {"schema": "cuteafd.fidelity.reference/2", "windows": [window]}
    if change == "missing":
        reference["windows"] = []
    elif change == "duplicate":
        reference["windows"].append(dict(window))
    elif change == "schema":
        reference["schema"] = "unknown"
    elif change == "start":
        window["score_from"] = True
    elif change == "short":
        window["tokens"].pop()
    else:
        window["tokens"][0] = -1
    with pytest.raises(ValueError, match="legacy"):
        builder.legacy_window(reference)


def test_arm_provenance_and_sources_fail_closed():
    arm = policy()
    arm["head"] = "fp8"
    with pytest.raises(ValueError, match="BF16"):
        builder.arm_policy(arm, "test-model")
    with pytest.raises(ValueError, match="checkpoint"):
        builder.arm_policy(policy(), "other-model")
    with pytest.raises(ValueError, match="inside"):
        builder.source(ROOT, "../secret.txt")


def test_recording_model_mismatch(fake_server):
    with pytest.raises(ValueError, match="selected model"):
        make(fake_server, recordings=[{"model": "external-teacher", "mode": "record", "sessions": []}])


def test_recording_history_above_16k_is_not_truncated(fake_server):
    recording = {"model": "test-model", "mode": "record", "sessions": [
        {"messages": [{"role": "user", "content": "x" * 60000}], "turns": [{"messages": 1}]}]}
    with pytest.raises(ValueError, match="16K"):
        make(fake_server, recordings=[recording])


# ---- recipe v2: in-window tool continuation and native trailing context (#431) ----

class TemplateRenderer:
    """A checkpoint-like template: each message is a tagged span; ids are characters."""

    @staticmethod
    def message(m):
        text = m.get("reasoning_content", "") + "|" + (m.get("content") or "")
        for call in m.get("tool_calls") or []:
            text += "{" + call["function"]["name"] + ":" + call["function"]["arguments"] + "}"
        return f"<{m['role']}>" + text + f"</{m['role']}>"

    @staticmethod
    def ids(text):
        # About four characters per id, as for a real BPE vocabulary.
        return [sum(map(ord, text[i:i + 4])) % 50000 + 100 for i in range(0, len(text), 4)]

    def render(self, messages, generation=False):
        # Role headers and message bodies are token boundaries, as with special-token templates.
        out = []
        for m in messages:
            head = f"<{m['role']}>"
            out += [7] + self.ids(self.message(m)[len(head):]) if m["role"] == "assistant" else self.ids(self.message(m))
        return out + ([7] if generation else [])

    def conversation_ids(self, body, messages):
        return self.render(messages)


class FakeAPI:
    """Own-text API probe: replies with tool calls for the first ``calls`` turns of a window."""

    def __init__(self, generated=40, calls=2, tool_windows=("a", "b", "d", "e")):
        self.renderer, self.generated, self.calls = TemplateRenderer(), generated, calls
        self.tool_windows, self.identity, self.requests = tool_windows, {"text_source": "api"}, []

    def prompt(self, body):
        return {"probe": {"prompt_ids": self.renderer.render(body["messages"], True)}}

    def __call__(self, body):
        self.requests.append(body)
        name = body["messages"][0]["content"].split("Fidelity fixture ")[-1][:3]
        own = sum(1 for m in body["messages"] if m["role"] == "tool" and m.get("tool_call_id", "").startswith("c"))
        message = {"role": "assistant", "reasoning_content": "r" * 8, "content": "x" * self.generated}
        if name[0] in self.tool_windows and own < self.calls:
            tool = "list_dir" if body.get("tools") else "bash"
            message["tool_calls"] = [{"id": f"c{own}", "type": "function",
                                      "function": {"name": tool, "arguments": "{\"path\":\".\"}"}}]
        prompt = self.renderer.render(body["messages"], True)
        generated = self.renderer.render(body["messages"] + [message])[len(prompt):]
        return {"probe": {"prompt_ids": prompt, "generated": generated,
                          "engine": "api"},
                "content": message["content"], "reasoning": message["reasoning_content"],
                "tool_calls": message.get("tool_calls", []), "server": {}}


def make_api(probe, recipe):
    return builder.build_set(family="mistral_large4", model="test-model", checkpoint="test-model",
        version="t", arm={"checkpoint": "test-model", "snapshot_revision": "r", "config_sha256": "c",
                          "tokenizer_sha256": "t"},
        tokenizer=FakeTokenizer(), probe=probe, recipe=recipe,
        legacy_reference={"schema": "cuteafd.bench.reference/1", "score_from": 64, "tokens": list(range(576))},
        legacy_provenance={"path": "synthetic-test-anchor", "sha256": "b" * 64})


def test_v1_remains_default_and_unversioned(fake_server):
    manifest = make(fake_server)
    assert "version" not in manifest["recipe"]
    assert all("turns" not in w["provenance"] for w in manifest["windows"])


def test_v2_continues_tool_turns_in_window_and_versions_the_recipe():
    probe = FakeAPI(generated=200, calls=3)
    manifest = make_api(probe, "v2")
    version = manifest["recipe"]["version"]
    assert version["id"] == "own-text-v2" and len(version["sha256"]) == 64
    for window in manifest["windows"]:
        if window["id"][0] in "ab" or window["id"] in ("d03", "d04", "d05") or window["id"][0] == "e":
            p = window["provenance"]
            assert len(p["turns"]) > 1 or p.get("ended_at_stratum") or p.get("ended_at_context_cap")
            scored = window["roles"][window["score_from"]:]
            assert len(scored) == 512 and scored.count("gen") == p["scored_generated"]
    # Tool-less windows drop the tools clause and get the unavailable-tool result.
    b00 = [r for r in probe.requests if "Fidelity fixture b00" in r["messages"][0]["content"]]
    assert all("repository tools" not in r["messages"][0]["content"] for r in b00)
    assert any(m.get("content") == builder.V2_TOOL_UNAVAILABLE for m in b00[-1]["messages"])
    a00 = [r for r in probe.requests if "Fidelity fixture a00" in r["messages"][0]["content"]]
    assert "repository tools" in a00[0]["messages"][0]["content"]
    assert any(m["role"] == "tool" and m["content"] != builder.V2_TOOL_UNAVAILABLE for m in a00[-1]["messages"])


def test_v2_earlier_turns_stay_generated_only_on_exact_prefix():
    probe = FakeAPI(generated=800, calls=2)
    manifest = make_api(probe, "v2")
    e00 = next(w for w in manifest["windows"] if w["id"] == "e00")
    assert e00["provenance"]["earlier_turns_ctx"] == 0
    # Tool results stay ctx: the gen ids are exactly the generated assistant spans.
    assert e00["roles"].count("gen") == sum(t["generated_tokens"] for t in e00["provenance"]["turns"])
    assert e00["roles"][e00["score_from"]:].count("gen") >= 400


def test_v2_short_answers_get_native_user_turn_context():
    probe = FakeAPI(generated=30, calls=0)
    manifest = make_api(probe, "v2")
    e00 = next(w for w in manifest["windows"] if w["id"] == "e00")
    padding = e00["provenance"]["trailing_ctx"]
    assert padding > 0
    renderer = TemplateRenderer()
    head = renderer.ids(renderer.message({"role": "user", "content": "File: " + "x" * 64}))[:2]
    assert e00["tokens"][-padding:][:2] == head
    assert set(e00["roles"][-e00["provenance"]["trailing_ctx"]:]) == {"ctx"}
