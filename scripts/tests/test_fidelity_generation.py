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


def test_truncated_thinking_is_not_mistaken_for_final_content():
    message = parse_qwen_response("still thinking", [], thinking_prefix=True)
    assert message["reasoning_content"] == "still thinking" and message["content"] == ""
