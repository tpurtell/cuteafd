#!/usr/bin/env python3
"""Render Mistral Large 4 chat goldens with two official renderers.

Each case is an OpenAI chat request. It is rendered by the checkpoint's own
``chat_template.jinja`` under Transformers' ``apply_chat_template`` (the
fidelity recipe's ChatRenderer path: JSON-string tool arguments parsed first)
and by ``mistral_common``'s instruct tokenizer from ``tekken.json``; the two
must agree byte for byte and token for token, or the script fails. The Rust
renderer (cuteafd-api ``chat::mistral``) must reproduce the strings.

Needs transformers and mistral_common (CPU only):

  python3 -I scripts/qualify/mistral_large4/mistral-template-goldens.py SNAPSHOT
"""
import argparse
import copy
import json
import pathlib
import shutil

ROOT = pathlib.Path(__file__).resolve().parents[3]
FIXTURES = ROOT / "rust/crates/cuteafd-api/src/openai/chat/mistral/fixtures"

WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "Current weather for a city (°C).",
    "parameters": {"type": "object", "properties": {
        "city": {"type": "string", "description": "City name, e.g. 台北"},
        "days": {"type": "integer", "minimum": 1, "maximum": 7}}, "required": ["city"]}}}
RUN = {"type": "function", "function": {"name": "run_tests", "description": "Run the suite.",
                                        "parameters": {"type": "object", "properties": {}}}}

CASES = {
    "single_user": {"messages": [{"role": "user", "content": "What is 2 + 2?"}]},
    "system_low_effort": {"messages": [{"role": "system", "content": "You are terse."},
                                       {"role": "user", "content": "Hi"}], "reasoning_effort": "low"},
    "medium_effort_multi_turn_reasoning": {"messages": [
        {"role": "user", "content": "Pick a number."},
        {"role": "assistant", "content": "7", "reasoning_content": "Any number works."},
        {"role": "user", "content": "Another?"}], "reasoning_effort": "medium"},
    "tools_and_calls": {"tools": [WEATHER, RUN], "messages": [
        {"role": "system", "content": "Use tools."},
        {"role": "user", "content": "Weather in 台北 for 2 days, then run tests."},
        {"role": "assistant", "content": None, "reasoning_content": "Two calls.", "tool_calls": [
            {"id": "a1b2c3d4e", "type": "function",
             "function": {"name": "get_weather", "arguments": "{\"city\":\"台北\",\"days\":2}"}},
            {"id": "f5g6h7i8j", "type": "function", "function": {"name": "run_tests", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "a1b2c3d4e", "name": "get_weather", "content": "22 °C, rain"},
        {"role": "tool", "tool_call_id": "f5g6h7i8j", "name": "run_tests", "content": "ok: 12 passed"},
        {"role": "user", "content": "Summarize."}]},
    "json_schema": {"messages": [{"role": "user", "content": "Give me a point."}], "response_format": {
        "type": "json_schema", "json_schema": {"name": "point", "strict": True, "schema": {
            "type": "object", "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
            "required": ["x", "y"], "additionalProperties": False}}}},
    "json_object": {"messages": [{"role": "user", "content": "Reply in JSON."}],
                    "response_format": {"type": "json_object"}},
    "text_blocks_and_marker_literals": {"messages": [
        {"role": "user", "content": [{"type": "text", "text": "Quote: "},
                                     {"type": "text", "text": "a [b] {c}\n\n  d"}]}]},
    "consecutive_users": {"messages": [{"role": "user", "content": "First."},
                                       {"role": "user", "content": "Second."}]},
    "assistant_prefix_history": {"messages": [
        {"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi! How can I help?"},
        {"role": "user", "content": "Tell me a joke."}], "reasoning_effort": "high"},
}


def hf_render(tokenizer, body):
    messages = copy.deepcopy(body["messages"])
    for message in messages:
        for call in message.get("tool_calls") or []:
            arguments = call["function"].get("arguments")
            if isinstance(arguments, str):
                try:
                    call["function"]["arguments"] = json.loads(arguments)
                except json.JSONDecodeError:
                    pass
    options = {key: body[key] for key in ("reasoning_effort", "response_format") if key in body}
    return tokenizer.apply_chat_template(messages, tools=body.get("tools"), tokenize=False,
                                         add_generation_prompt=True, **options)


def mistral_render(tokenizer, body):
    from mistral_common.protocol.instruct.request import ChatCompletionRequest
    request = {key: body[key] for key in ("messages", "tools", "reasoning_effort", "response_format") if key in body}
    messages = copy.deepcopy(request["messages"])
    for message in messages:
        if message.get("reasoning_content") is not None:
            thinking = {"type": "thinking", "thinking": message.pop("reasoning_content")}
            text = message.get("content")
            message["content"] = [thinking] + ([{"type": "text", "text": text}] if text else [])
        if message["role"] == "tool":
            message.pop("name", None)
    request["messages"] = messages
    encoded = tokenizer.encode_chat_completion(ChatCompletionRequest(**request))
    return encoded.text, encoded.tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot", type=pathlib.Path)
    args = parser.parse_args()
    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
    from tokenizers import Tokenizer
    from transformers import PreTrainedTokenizerFast
    hf = PreTrainedTokenizerFast.from_pretrained(args.snapshot, local_files_only=True)
    hf.chat_template = (args.snapshot / "chat_template.jinja").read_text()
    raw = Tokenizer.from_file(str(args.snapshot / "tokenizer.json"))
    mistral = MistralTokenizer.from_file(str(args.snapshot / "tekken.json"))
    goldens = []
    for name, body in CASES.items():
        expected = hf_render(hf, body)
        text, tokens = mistral_render(mistral, body)
        assert text == expected, f"{name}: template and mistral_common differ\n{expected!r}\n{text!r}"
        ids = raw.encode(expected, add_special_tokens=False).ids
        assert ids == tokens, f"{name}: tokenizer.json and tekken token ids differ"
        goldens.append({"name": name, "request": body, "expected": expected, "token_ids": ids})
    FIXTURES.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.snapshot / "chat_template.jinja", FIXTURES / "mistral_large4.jinja")
    (FIXTURES / "mistral_template_goldens.json").write_text(json.dumps(goldens, indent=1, ensure_ascii=False) + "\n")
    print(f"{len(goldens)} goldens agree across the checkpoint template and mistral_common")


if __name__ == "__main__":
    main()
