import json
import os
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ernie55_flash import prepare_corpus as corpus


def demo(function="lookup", calls=None, description=""):
    schema = [{"name": function, "description": description, "parameters": {}}]
    if calls is None:
        calls = [{"name": function, "arguments": {"k": "x"}}]
    return ("Tool schemas:\n" + json.dumps(json.dumps(schema)) + "\n\nUser query:\nuse lookup"
            + "\n\nExpected tool calls:\n" + json.dumps(json.dumps(calls)))


def test_decode_all_published_tool_demos_in_order_and_keep_empty_calls():
    messages, tools = corpus.tool_conversation(demo() + "\n\n---\n\n" + demo(calls=[], description="second"))
    assert [m["role"] for m in messages] == ["user", "assistant", "user", "assistant"]
    assert messages[1]["tool_calls"][0]["function"] == {"name": "lookup", "arguments": {"k": "x"}}
    assert messages[3]["tool_calls"] == []
    assert [t["function"]["description"] for t in tools] == ["", "second"]
    with pytest.raises(ValueError, match="declared schema"):
        corpus.tool_conversation(demo(calls=[{"name": "missing", "arguments": {}}]))


def test_pinned_segment_encoder_preserves_marker_text_and_emits_native_calls():
    cache = Path(os.environ.get("HF_HUB_CACHE", Path(os.environ.get("HF_HOME", "~/.cache/huggingface"))
                                .expanduser() / "hub"))
    source = cache / "models--baidu--ERNIE-5.5-Flash-Preview-BF16/snapshots" / corpus.SOURCE_REVISION
    if not source.is_dir():
        pytest.skip("pinned ERNIE tokenizer snapshot absent")
    tokenizer = corpus.load_tokenizer(source)
    _, ids, rendered = corpus.encode_row({"prompt": "<ernie:tool_call>", "id": "literal"}, "short", tokenizer)
    call_id = tokenizer.convert_tokens_to_ids("<ernie:tool_call>")
    assert call_id not in ids
    assert rendered.endswith("<ernie:think>") and "reasoning_effort=high" in rendered
    _, ids, rendered = corpus.encode_row({"prompt": demo(), "id": "tool"}, "tool_call", tokenizer)
    assert ids.count(call_id) == 1
    assert '<invoke name="lookup">' in rendered
    assert "<ernie:tool_list>" in rendered


def test_eight_long_witnesses_have_separate_verified_inventory(tmp_path):
    class Tokenizer:
        @staticmethod
        def encode_text(text):
            return [1] * len(text)

        @staticmethod
        def render_chat(messages, **chat):
            assert chat == corpus.CHAT
            from types import SimpleNamespace
            return SimpleNamespace(segments=(messages[0]["content"],))

        @staticmethod
        def encode_segments(segments):
            return [1] * len(segments[0])

    witness = corpus.write_witnesses(tmp_path, Tokenizer(), [dict(id="wiki", prompt="a" * 512)],
                                     [dict(domain="wiki")])
    manifest = dict(schema=corpus.SCHEMA, status="passed", contract=corpus.CONTRACT,
                    dataset=corpus.DATASET, long_context_witnesses=witness)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    sizes = [ids.shape[1] for ids in corpus.load_witnesses(tmp_path)]
    assert len(set(sizes)) == 8 and 16384 <= min(sizes) and max(sizes) <= 65536
    with pytest.raises(ValueError, match="unknown split"):
        list(corpus.load_token_batches(tmp_path, "witness"))
    witness["records"][1]["input_ids_sha256"] = "0" * 64
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="IDs differ"):
        list(corpus.load_witnesses(tmp_path))
