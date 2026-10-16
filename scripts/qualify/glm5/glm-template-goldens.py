#!/usr/bin/env python3
"""Render GLM 5.3 chat-template goldens with Hugging Face's Jinja environment.

The Rust renderer (cuteafd-api `native_v41::glm`) must reproduce these strings
byte for byte. Each case is a template context: the request conversion under
test produces the same context, so the goldens stay independent of it.

Run inside the coordinator dev image (CPU only):

  docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp \\
    -v "$PWD:/work" cuteafd-coordinator-dev:latest \\
    python3 /work/scripts/qualify/glm5/glm-template-goldens.py
"""
import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
FIXTURES = ROOT / "rust/crates/cuteafd-api/src/openai/chat/glm5/fixtures"
TEMPLATES = {"exl3_k4": "glm53_exl3_k4.jinja", "zai": "glm53_zai.jinja"}

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
                "threshold": {"type": "number", "default": 0.5, "exclusiveMinimum": 1e-07},
                "units": {"type": "string", "enum": ["c", "f"]},
            },
            "required": ["city"],
        },
        "strict": True,
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
DEFERRED = {"type": "function", "function": {"name": "search_docs", "defer_loading": True,
                                             "parameters": {"type": "object", "properties": {}}}}


def gen(**context):
    return dict(add_generation_prompt=True, **context)


CASES = {
    "single_user": gen(messages=[{"role": "user", "content": "What is 2 + 2?"}]),
    "system_high_effort": gen(
        messages=[{"role": "system", "content": "You are terse."}, {"role": "user", "content": "Hi"}],
        reasoning_effort="high",
    ),
    "low_effort_multi_turn_reasoning": gen(
        messages=[
            {"role": "user", "content": "Pick a number."},
            {"role": "assistant", "content": " 7 ", "reasoning_content": "Any number works."},
            {"role": "user", "content": "Another?"},
        ],
        reasoning_effort="low",
    ),
    "clear_thinking_drops_old_reasoning": gen(
        messages=[
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "One", "reasoning_content": "old thought"},
            {"role": "user", "content": "Second"},
            {"role": "assistant", "content": "Two", "reasoning_content": "new thought"},
        ],
        clear_thinking=True,
    ),
    "inline_think_in_content": gen(
        messages=[
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "<think>hidden</think>\n\nvisible answer\n"},
            {"role": "user", "content": "q2"},
        ],
    ),
    "reasoning_scope_across_assistants": gen(
        messages=[
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "A", "reasoning_content": "thinking about a"},
            {"role": "user", "content": "b"},
            {"role": "assistant", "content": "B"},
            {"role": "user", "content": "c"},
        ],
    ),
    "text_parts_and_image_reminder": gen(
        messages=[
            {"role": "system", "content": [{"type": "text", "text": "sys "}, {"type": "text", "text": "parts"}]},
            {"role": "user", "content": [
                {"type": "text", "text": "Look: "},
                {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
                {"type": "text", "text": " then answer."},
            ]},
        ],
    ),
    "tools_declared": gen(
        messages=[{"role": "user", "content": "Weather in Taipei?"}],
        tools=[WEATHER, REFRESH, DEFERRED],
    ),
    "tool_round_trip": gen(
        messages=[
            {"role": "system", "content": "Use tools when helpful."},
            {"role": "user", "content": "Weather in Taipei and Oslo, then save it."},
            {"role": "assistant", "content": "", "reasoning_content": "Two lookups.", "tool_calls": [
                {"id": "call_a", "type": "function", "function": {"name": "get_weather", "arguments": {
                    "city": "Taipei", "days": 2, "threshold": 1.0, "units": "c"}}},
                {"id": "call_b", "type": "function", "function": {"name": "get_weather", "arguments": {
                    "city": "Oslo", "days": 1, "threshold": 1e-07}}},
            ]},
            {"role": "tool", "tool_call_id": "call_b", "content": "{\"temp\": -3}"},
            {"role": "tool", "tool_call_id": "call_a", "content": "{\"temp\": 31}"},
            {"role": "assistant", "content": "Saving.", "tool_calls": [
                {"id": "call_c", "type": "function", "function": {"name": "write_file", "arguments": {
                    "path": "/tmp/w.txt", "text": "Taipei 31 \"hot\"\nOslo -3 ❄",
                    "mode": {"append": False}, "tags": ["天气", "weather"]}}},
            ]},
            {"role": "tool", "tool_call_id": "call_c", "content": [{"type": "text", "text": "ok"}]},
            {"role": "assistant", "content": "Done: Taipei 31°C, Oslo -3°C.", "reasoning_content": "Report."},
            {"role": "user", "content": "Thanks, and now Paris?"},
        ],
        tools=[WEATHER, WRITE],
        reasoning_effort="high",
    ),
    "tool_results_unsortable": gen(
        messages=[
            {"role": "user", "content": "Refresh twice."},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "r1", "type": "function", "function": {"name": "refresh", "arguments": {}}},
                {"id": "r2", "type": "function", "function": {"name": "refresh", "arguments": {}}},
            ]},
            {"role": "tool", "tool_call_id": "r2", "content": "second"},
            {"role": "tool", "content": "no id"},
        ],
        tools=[REFRESH],
    ),
    "tool_reference_result": gen(
        messages=[
            {"role": "user", "content": "Find the docs tool."},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "t1", "type": "function", "function": {"name": "refresh", "arguments": {}}}]},
            {"role": "tool", "tool_call_id": "t1", "content": [{"type": "tool_reference", "name": "search_docs"}]},
        ],
        tools=[REFRESH, DEFERRED],
    ),
    "numbers_and_escapes": gen(
        messages=[
            {"role": "user", "content": "Numbers"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "n1", "type": "function", "function": {"name": "calc", "arguments": {
                    "a": 1e16, "b": 1.5e-05, "c": 0.0001, "d": -0.0, "e": 123456789.125, "f": 12345678901234567890,
                    "g": None, "h": [1, 2.5, "x\ty\u0001 "], "i": {"k": "💡"}}}}]},
            {"role": "tool", "tool_call_id": "n1", "content": "ok"},
        ],
    ),
}


def main():
    # Render with the checkout's pinned source, not an installed distribution:
    # the pinned revision's chat-template helpers are what the Rust renderer is
    # checked against, and no transformers wheel ships in the dev image.
    pinned = (ROOT / "third_party/transformers/src/transformers").resolve()
    sys.path.insert(0, str(pinned.parent))
    import transformers
    from transformers.utils.chat_template_utils import render_jinja_template
    if pathlib.Path(transformers.__file__).resolve().parent != pinned:
        raise SystemExit(f"import transformers from the pinned checkout: {pinned}")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(FIXTURES / "glm53_template_goldens.json"))
    args = parser.parse_args()
    templates = {name: (FIXTURES / path).read_text() for name, path in TEMPLATES.items()}
    goldens = []
    for name, context in CASES.items():
        expected = {}
        for template_name, template in templates.items():
            kwargs = {k: v for k, v in context.items() if k not in ("messages", "tools", "add_generation_prompt")}
            rendered, _ = render_jinja_template(
                conversations=[context["messages"]], tools=context.get("tools"), chat_template=template,
                add_generation_prompt=context["add_generation_prompt"], **kwargs)
            expected[template_name] = rendered[0]
        goldens.append({"name": name, "context": context, "expected": expected})
    pathlib.Path(args.out).write_text(json.dumps(goldens, ensure_ascii=False, indent=1) + "\n")
    print(f"wrote {len(goldens)} cases to {args.out}")


if __name__ == "__main__":
    main()
