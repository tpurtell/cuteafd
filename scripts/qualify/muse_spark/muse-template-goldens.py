#!/usr/bin/env python3
"""Render Muse Spark chat-template goldens with Hugging Face's Jinja environment.

The Rust renderer (cuteafd-api `chat::muse`) must reproduce these strings byte
for byte. Each case is a template context; `current_date` is fixed so the
goldens do not depend on the day they were rendered.

Run inside the coordinator dev image (CPU only):

  docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp \\
    -v "$PWD:/work" cuteafd-coordinator-dev:latest \\
    python3 /work/scripts/qualify/muse_spark/muse-template-goldens.py
"""
import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
FIXTURES = ROOT / "rust/crates/cuteafd-api/src/openai/chat/muse/fixtures"
TEMPLATE = "muse_spark.jinja"
DATE = "2026-11-01"

WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current weather for a city (°C).",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name, e.g. 台北"},
                "days": {"type": "integer", "minimum": 1, "maximum": 7},
                "units": {"type": "string", "enum": ["c", "f"]},
            },
            "required": ["city"],
        },
    },
}
REFRESH = {"type": "function", "function": {"name": "refresh"}}
WRITE = {
    "type": "function",
    "function": {
        "name": "write_file",
        "description": "Write \"text\" to a path.\nOverwrites.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "text": {"type": "string"},
                "mode": {"type": "object", "properties": {"append": {"type": "boolean"}}},
                "tags": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["path", "text"],
            "additionalProperties": False,
        },
    },
}


def gen(**context):
    context.setdefault("add_generation_prompt", True)
    context.setdefault("current_date", DATE)
    return context


CALL = {"id": "call_1", "type": "function",
        "function": {"name": "get_weather", "arguments": {"city": "台北", "days": 3}}}
WRITE_CALL = {"id": "call_2", "type": "function",
              "function": {"name": "write_file", "arguments": {
                  "path": "/tmp/a<b>.txt", "text": "line 1\nline \"2\" <|eot|>", "mode": {"append": True},
                  "tags": ["x", "y"]}}}

CASES = {
    "plain": gen(messages=[{"role": "user", "content": "Hello"}]),
    "system_and_effort": gen(
        messages=[{"role": "system", "content": "Be terse. Reasoning effort is your call."},
                  {"role": "user", "content": "2+2?"}],
        reasoning_effort="low"),
    "effort_in_system": gen(
        messages=[{"role": "developer", "content": "Reasoning strength: 3."},
                  {"role": "user", "content": "Hi"}]),
    "escaping": gen(messages=[{"role": "user", "content": "say <|eot|> and <image> literally"}]),
    "multi_turn_history": gen(
        reasoning_effort="high",
        messages=[{"role": "user", "content": "First"},
                  {"role": "assistant", "content": "One.", "reasoning_content": "count"},
                  {"role": "user", "content": "Second"},
                  {"role": "assistant", "content": "Two."},
                  {"role": "user", "content": "Third"}]),
    "tools_declared": gen(tools=[WEATHER, REFRESH, WRITE],
                          messages=[{"role": "user", "content": "Weather in Taipei?"}]),
    "tool_round_trip": gen(
        tools=[WEATHER, WRITE],
        messages=[{"role": "user", "content": "Weather, then save it."},
                  {"role": "assistant", "content": "Checking.", "reasoning_content": "need weather",
                   "tool_calls": [CALL]},
                  {"role": "tool", "tool_call_id": "call_1", "content": "{\"temp\": 21}"},
                  {"role": "assistant", "content": None, "tool_calls": [WRITE_CALL]},
                  {"role": "tool", "tool_call_id": "call_2", "content": "ok"}]),
    "parallel_calls": gen(
        tools=[WEATHER],
        messages=[{"role": "user", "content": "Two cities"},
                  {"role": "assistant", "content": "", "tool_calls": [
                      CALL, {"id": "call_3", "type": "function",
                             "function": {"name": "get_weather", "arguments": {"city": "Oslo"}}}]},
                  {"role": "tool", "tool_call_id": "call_1", "content": "21"},
                  {"role": "tool", "tool_call_id": "call_3", "content": "4"}]),
    "content_parts": gen(messages=[{"role": "user", "content": [
        {"type": "text", "text": "Part one. "}, {"type": "text", "text": "Part two."}]}]),
    "no_generation_prompt": gen(add_generation_prompt=False,
                                messages=[{"role": "user", "content": "Hi"},
                                          {"role": "assistant", "content": "Hello!"}]),
}


def main():
    # Render with the checkout's pinned source: the pinned revision's
    # chat-template helpers are what the Rust renderer is checked against.
    pinned = (ROOT / "third_party/transformers/src/transformers").resolve()
    sys.path.insert(0, str(pinned.parent))
    import transformers
    from transformers.utils.chat_template_utils import render_jinja_template
    if pathlib.Path(transformers.__file__).resolve().parent != pinned:
        raise SystemExit(f"import transformers from the pinned checkout: {pinned}")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(FIXTURES / "muse_template_goldens.json"))
    args = parser.parse_args()
    template = (FIXTURES / TEMPLATE).read_text()
    goldens = []
    for name, context in CASES.items():
        kwargs = {k: v for k, v in context.items() if k not in ("messages", "tools", "add_generation_prompt")}
        rendered, _ = render_jinja_template(
            conversations=[context["messages"]], tools=context.get("tools"), chat_template=template,
            add_generation_prompt=context["add_generation_prompt"], bos_token="<|begin_of_text|>", **kwargs)
        goldens.append({"name": name, "context": dict(context, bos_token="<|begin_of_text|>"),
                        "expected": rendered[0]})
    pathlib.Path(args.out).write_text(json.dumps(goldens, ensure_ascii=False, indent=1) + "\n")
    print(f"wrote {len(goldens)} cases to {args.out}")


if __name__ == "__main__":
    main()
