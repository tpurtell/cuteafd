"""Own-text backend contracts without credentials, CUDA or external APIs."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python/reference"))
from fidelity_generation import APIClient, ChatRenderer, HFClient, parse_qwen_response
from fidelity_windows import canonical


class Tokenizer:
    def get_chat_template(self):
        return "test checkpoint template"

    def encode(self, text, **kwargs):
        return list(text.encode())

    def decode(self, ids, **kwargs):
        return bytes(ids).decode()

    def apply_chat_template(self, messages, tools=None, tokenize=False, add_generation_prompt=False):
        text = ""
        for message in messages:
            text += message["role"] + ":" + (message.get("content") or "")
            if message.get("reasoning_content"):
                text += "<reasoning>" + message["reasoning_content"]
            if message.get("tool_calls"):
                text += json.dumps(message["tool_calls"], sort_keys=True)
            text += "\n"
        return text + ("assistant:" if add_generation_prompt else "")


@pytest.fixture
def api(tmp_path):
    state = {"calls": [], "failures": 0}
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            assert self.path == "/chat/completions"
            assert self.headers["Authorization"] == "Bearer test-private-value"
            state["calls"].append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            if state["failures"]:
                state["failures"] -= 1
                self.send_response(429)
                self.end_headers()
                return
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps({"id": "response-test", "model": "official-model", "provider": "original",
                "usage": {"completion_tokens": 3, "cost": .001},
                "choices": [{"message": state.get("message", {"role": "assistant", "content": "answer", "reasoning_content": "reason"}),
                             "finish_reason": "stop"}]}).encode())
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    key = tmp_path / "key"
    key.write_text("test-private-value")
    key.chmod(0o600)
    client = APIClient(ChatRenderer(Tokenizer()), tmp_path / "cache", checkpoint="official-model", family="qwen3_5",
                       base=f"http://127.0.0.1:{server.server_port}", model="official-model", key_file=key)
    try:
        yield client, state
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def body():
    return {"model": "official-model", "messages": [{"role": "user", "content": "fixture"}],
            "temperature": 0, "seed": 0, "max_tokens": 1024, "reasoning_effort": "high", "stream": False}


def test_api_local_sizing_cache_split_and_provenance(api):
    client, state = api
    request = body()
    prompt = client.prompt(request)["probe"]["prompt_ids"]
    assert state["calls"] == []
    result = client(request)
    assert result["probe"]["prompt_ids"] == prompt
    assert result["content"] == "answer" and result["reasoning"] == "reason"
    assert result["generation_provenance"]["provider"] == "original"
    assert result["generation_provenance"]["usage"]["cost"] == .001
    assert result["generation_provenance"]["generated_ids_full"] == len(result["probe"]["generated"])
    assert result["generation_provenance"]["generated_ids_kept"] == len(result["probe"]["generated"])
    assert client(request) == result
    assert len(state["calls"]) == 1
    cached = next(client.cache.directory.glob("*.json")).read_text()
    assert "test-private-value" not in cached and "Authorization" not in cached
    assert next(client.cache.directory.glob("*.json")).stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("model", ["official-model", "another-model"])
def test_api_caps_full_rendered_ids_and_preserves_raw_response(api, model):
    client, state = api
    client.model = model
    call = {"id": "call0", "type": "function", "function": {"name": "read_file", "arguments": '{"path":"README.md"}'}}
    state["message"] = {"role": "assistant", "content": "answer" * 300,
                        "reasoning_content": "reason" * 300, "tool_calls": [call]}
    request = body()
    full = client.renderer.completion(request, state["message"])
    result = client(request)
    assert result["probe"]["generated"] == full[:1024]
    assert result["tool_calls"] == []
    provenance = result["generation_provenance"]
    assert provenance["generated_ids_full"] == len(full)
    assert provenance["generated_ids_kept"] == 1024
    assert provenance["tool_calls"] == [call]
    cached = json.loads(next(client.cache.directory.glob("*.json")).read_text())
    assert cached["response"]["choices"][0]["message"] == state["message"]
    assert client(request) == result
    assert len(state["calls"]) == 1


def test_roundtrip_reconstructed_span(api):
    client, _ = api
    request = body()
    result = client(request)
    assistant = {"role": "assistant", "content": "answer", "reasoning_content": "reason"}
    full = client.renderer.render(request, assistant)
    ids = result["probe"]["prompt_ids"] + result["probe"]["generated"]
    assert client.renderer.tokenizer.decode(ids) == full


def test_api_native_tools_are_rendered_as_objects(api):
    client, state = api
    call = {"id": "call0", "type": "function", "function": {"name": "read_file", "arguments": '{"path":"README.md"}'}}
    state["message"] = {"role": "assistant", "content": None, "tool_calls": [call]}
    result = client(body())
    assert result["tool_calls"] == [call]
    assert '"arguments": {"path": "README.md"}' in client.renderer.tokenizer.decode(result["probe"]["generated"])


def test_api_cache_reuses_original_source_bound_response(api):
    client, state = api
    client.identity["generator_sha256"] = "old-source"
    first = client(body())
    path = next(client.cache.directory.glob("*.json"))
    record = json.loads(path.read_bytes())
    old_path = path.with_name(hashlib.sha256(canonical(record["binding"])).hexdigest() + ".json")
    path.rename(old_path)
    original = old_path.read_bytes()
    client.identity["generator_sha256"] = "new-source"
    replay = client(body())
    assert replay["probe"] == first["probe"]
    assert replay["generation_provenance"]["cached_generator_sha256"] == "old-source"
    assert replay["generation_provenance"]["current_generator_sha256"] == "new-source"
    assert replay["generation_provenance"]["date"] == first["generation_provenance"]["date"]
    assert old_path.read_bytes() == original
    assert len(state["calls"]) == 1
    client.extra_body = {"provider": {"only": ["different"]}}
    client(body())
    assert len(state["calls"]) == 2


def test_openrouter_reasoning_budget_only_on_uncached_requests(api, monkeypatch):
    client, state = api
    client(body())
    record = json.loads(next(client.cache.directory.glob("*.json")).read_bytes())
    client.endpoint = "https://openrouter.ai/api/v1/chat/completions"
    client.identity["endpoint"] = client.endpoint
    client.cache.get(body(), lambda: record["response"])

    def request(payload):
        state["calls"].append(copy.deepcopy(payload))
        return copy.deepcopy(record["response"])

    monkeypatch.setattr(client, "request", request)
    original = client(body())
    assert client(body()) == original
    assert len(state["calls"]) == 1
    changed = body()
    changed["seed"] = 1
    client(changed)
    assert state["calls"][-1]["reasoning"] == {"max_tokens": 1024}
    assert "reasoning_effort" not in state["calls"][-1]
    client(changed)
    assert len(state["calls"]) == 2


def test_api_retry_and_request_identity(api, monkeypatch):
    client, state = api
    monkeypatch.setattr("fidelity_generation.time.sleep", lambda _: None)
    state["failures"] = 1
    client(body())
    assert len(state["calls"]) == 2
    changed = body()
    changed["seed"] = 1
    client(changed)
    assert len(state["calls"]) == 3


def test_api_rejects_unroundtrippable_span(api, monkeypatch):
    client, _ = api
    monkeypatch.setattr(client.renderer.tokenizer, "decode", lambda *a, **k: "changed")
    with pytest.raises(ValueError, match="losslessly"):
        client(body())


def test_hf_cache_and_tool_parser(tmp_path):
    renderer = ChatRenderer(Tokenizer())
    client = HFClient(renderer, tmp_path, checkpoint="root", family="qwen3_5", snapshot=tmp_path, identity={})
    generated = "<think>reason</think>\n<tool_call>\n<function=read_file>\n<parameter=path>\nREADME.md\n</parameter>\n</function>\n</tool_call><|im_end|>"
    client.generate = lambda _: {"generated": list(generated.encode())}
    request = body()
    request["tools"] = [{"type": "function", "function": {"name": "read_file", "parameters": {"properties": {"path": {"type": "string"}}}}}]
    result = client(request)
    assert result["reasoning"] == "reason"
    assert json.loads(result["tool_calls"][0]["function"]["arguments"]) == {"path": "README.md"}
    client.generate = lambda _: pytest.fail("cache miss")
    assert client(request) == result


class EffortTokenizer(Tokenizer):
    def get_chat_template(self):
        return "thinking_effort"

    def apply_chat_template(self, messages, *, thinking_effort="max", **kwargs):
        return "effort=" + thinking_effort + ":" + super().apply_chat_template(messages, **kwargs)


def test_renderer_binds_effort_for_prompt_and_completion():
    renderer = ChatRenderer(EffortTokenizer())
    request = body()
    assert renderer.render(request).startswith("effort=high:")
    assert renderer.prompt(request) == list(renderer.render(request).encode())
    assistant = {"role": "assistant", "content": "answer", "reasoning_content": "reason"}
    generated = renderer.completion(request, assistant)
    assert renderer.tokenizer.decode(renderer.prompt(request) + generated) == renderer.render(request, assistant)
    request["reasoning"] = {"effort": "low"}
    with pytest.raises(ValueError, match="conflicting reasoning efforts"):
        renderer.prompt(request)


def test_api_renders_cached_wire_effort_and_default_without_calls(api, monkeypatch):
    client, state = api
    client.family = "kimi_k3"
    client.renderer = ChatRenderer(EffortTokenizer())
    client.endpoint = "https://openrouter.ai/api/v1/chat/completions"
    client.identity["endpoint"] = client.endpoint
    request = body()
    raw = {"choices": [{"message": {"content": "answer", "reasoning_content": "reason"}}]}
    client.cache.get(request, lambda: raw)
    original = next(client.cache.directory.glob("*.json")).read_bytes()
    monkeypatch.setattr(client, "request", lambda _: pytest.fail("new API request"))
    high = client(request)
    assert client.renderer.tokenizer.decode(high["probe"]["prompt_ids"]).startswith("effort=high:")
    assert high["generation_provenance"]["rendered_reasoning"]["template_options"] == {"thinking_effort": "high"}
    changed = body()
    changed["seed"] = 1
    budget = copy.deepcopy(changed)
    budget.pop("reasoning_effort")
    budget["reasoning"] = {"max_tokens": 1024}
    client.cache.get(budget, lambda: raw)
    cached = client(changed)
    sizing = {**changed, "max_tokens": 1}
    assert client.prompt(sizing)["probe"]["prompt_ids"] == cached["probe"]["prompt_ids"]
    assert client.prompt(changed)["probe"]["prompt_ids"] == cached["probe"]["prompt_ids"]
    assert client.renderer.tokenizer.decode(cached["probe"]["prompt_ids"]).startswith("effort=max:")
    assert cached["generation_provenance"]["rendered_reasoning"]["source"] == "documented Kimi K3 provider default max"
    assert original in [p.read_bytes() for p in client.cache.directory.glob("*.json")]
    assert state["calls"] == []


def test_truncated_thinking_is_not_mistaken_for_final_content():
    message = parse_qwen_response("still thinking", [], thinking_prefix=True)
    assert message["reasoning_content"] == "still thinking" and message["content"] == ""


class ToolOrderTokenizer(Tokenizer):
    """A template that serializes tools in their given key order (Mistral tojson)."""

    def apply_chat_template(self, messages, tools=None, tokenize=False, add_generation_prompt=False):
        return json.dumps(tools) + super().apply_chat_template(messages, tokenize=tokenize,
                                                               add_generation_prompt=add_generation_prompt)


def test_cached_rerender_keeps_original_tool_key_order(api):
    client, state = api
    client.renderer = ChatRenderer(ToolOrderTokenizer())
    request = body()
    request["tools"] = [{"type": "function", "function": {"name": "read_file", "description": "d",
                                                          "parameters": {"type": "object"}}}]
    first = client(copy.deepcopy(request))
    # A second identical request is a cache hit read back from canonical JSON.
    second = client(copy.deepcopy(request))
    assert len(state["calls"]) == 1
    assert second["probe"]["prompt_ids"] == first["probe"]["prompt_ids"] == client.prompt(request)["probe"]["prompt_ids"]
    assert bytes(second["probe"]["prompt_ids"]).decode().startswith('[{"type": "function"')


class DateTokenizer(Tokenizer):
    def get_chat_template(self):
        return "{{ current_date }} template"

    def apply_chat_template(self, messages, tools=None, tokenize=False, add_generation_prompt=False, current_date=None):
        return f"date={current_date}:" + super().apply_chat_template(messages, tokenize=tokenize,
                                                                       add_generation_prompt=add_generation_prompt)


def test_template_date_is_pinned_and_checked_against_cached_requests(api):
    import datetime
    client, state = api
    today = datetime.datetime.now(datetime.timezone.utc).date().isoformat()
    client.renderer = ChatRenderer(DateTokenizer(), template_date=today)
    result = client(body())
    assert bytes(result["probe"]["prompt_ids"]).decode().startswith(f"date={today}:")
    client.renderer = ChatRenderer(DateTokenizer(), template_date="2000-01-01")
    with pytest.raises(ValueError, match="differs from template date"):
        client(body())
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        ChatRenderer(DateTokenizer(), template_date="yesterday")


def test_length_cut_tool_call_is_rendered_verbatim_and_never_executed(api):
    client, state = api
    state["message"] = {"role": "assistant", "content": "", "reasoning_content": "r",
                        "tool_calls": [{"id": "c", "type": "function",
                                        "function": {"name": "read_file", "arguments": "{\"path\": \"a"}}]}
    result = client(body())
    assert result["tool_calls"] == []
    text = bytes(result["probe"]["generated"]).decode()
    # Only the text both bounding renders share: never the partial arguments.
    assert text.startswith("<reasoning>r") and "path" not in text


def test_wire_body_keeps_request_key_order_and_keys_its_cache(api, tmp_path):
    client, state = api
    request = body()
    request["tools"] = [{"type": "function", "function": {"name": "read_file", "parameters": {"type": "object"}}}]
    client(copy.deepcopy(request))
    sent = state["calls"][-1]
    assert list(sent["tools"][0]) == ["type", "function"] and list(sent["tools"][0]["function"]) == ["name", "parameters"]
    assert client.identity["wire_order"] == "request"
    # A sorted-wire record never satisfies a request-order client, and vice versa.
    sorted_client = APIClient(client.renderer, client.cache.directory, checkpoint="official-model", family="qwen3_5",
                              base=client.endpoint.removesuffix("/chat/completions"), model="official-model",
                              key_file=client.key_file, wire_order="sorted")
    assert sorted_client.cache.find({**copy.deepcopy(request), "model": "official-model"}) is None


def test_engine_client_sends_local_prompt_ids_and_keeps_engine_ids(tmp_path):
    from fidelity_generation import EngineClient
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(payload)
            ids = payload["spec"]["prompt_ids"]
            text = "<reasoning>r" + "answer"
            out = {"probe": {"engine": "fake", "cold": True, "no_speculation": True, "cached_tokens": 0,
                             "prompt_ids": ids, "generated": list(text.encode())}}
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(out).encode())

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    tokenizer = Tokenizer()
    tokenizer._fidelity_parse_completion = lambda text, mode: {"role": "assistant", "reasoning_content": "r", "content": "answer", "tool_calls": []}
    try:
        client = EngineClient(ChatRenderer(tokenizer), tmp_path / "cache", checkpoint="m", family="deepseek_v41",
                              base=f"http://127.0.0.1:{server.server_port}", token=None, identity={})
        request = {**body(), "response_format": {"type": "json_object"}}
        result = client(request)
        assert seen[0]["spec"]["prompt_ids"] == client.renderer.prompt(request)
        assert seen[0]["spec"]["cold"] and seen[0]["spec"]["no_speculation"]
        assert "response_format" not in seen[0]["body"] and seen[0]["body"]["temperature"] == 0
        assert result["probe"]["generated"] == list("<reasoning>ranswer".encode())
        client(request)
        assert len(seen) == 1
    finally:
        server.shutdown()
        thread.join()
        server.server_close()
