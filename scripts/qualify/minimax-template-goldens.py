#!/usr/bin/env python3
"""Render MiniMax's unmodified checkpoint template with reference Jinja2.

Uses the same sandbox, whitespace settings and Python-compatible tojson as
Transformers (the Mistral template-golden reference). CPU only:
  python3 -I scripts/qualify/minimax-template-goldens.py SNAPSHOT
"""
import argparse
import copy
import json
import pathlib
import shutil

from transformers.utils.chat_template_utils import _compile_jinja_template

ROOT = pathlib.Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "rust/crates/cuteafd-api/src/openai/chat/minimax/fixtures"
TOOL = {"type": "function", "function": {"name": "lookup", "description": "Look up a record.",
    "parameters": {"type": "object", "properties": {"key": {"type": "string"},
        "count": {"type": "integer"}, "options": {"type": "object"}, "items": {"type": "array"}}}}}
CALL = {"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": json.dumps({
    "key": "001", "count": 2, "options": {"enabled": True, "skip": None, "nested": {"key": "v"}},
    "items": [{"name": "a"}, {"name": "b"}], "omit": None})}}
CASES = {
    "user": {"messages": [{"role": "user", "content": "Hello"}]},
    "system_user": {"messages": [{"role": "system", "content": "Be terse."}, {"role": "user", "content": "Hi"}]},
    "developer": {"messages": [{"role": "developer", "content": "Be useful."}, {"role": "user", "content": "Hi"}]},
    "root_developer": {"messages": [{"role": "root", "content": "Root policy", "current_date": "2026-11-14",
        "current_location": "Here"}, {"role": "developer", "content": "Developer policy"}, {"role": "user", "content": "Hi"}]},
    "multi_turn_reasoning": {"messages": [{"role": "user", "content": "Pick one."},
        {"role": "assistant", "reasoning_content": "I choose carefully.", "content": "1"},
        {"role": "user", "content": "Again?"}]},
    "inline_reasoning": {"messages": [{"role": "assistant", "content": "<mm:think>\nThink\n</mm:think>\nAnswer\n"},
        {"role": "user", "content": "Next"}]},
    "adaptive_without_reasoning": {"messages": [{"role": "assistant", "content": "Direct answer"},
        {"role": "user", "content": "Next"}]},
    "tools_declared": {"tools": [TOOL], "messages": [{"role": "user", "content": "Look up 001"}]},
    "recursive_tool_call_and_result": {"tools": [TOOL], "messages": [{"role": "user", "content": "Look up"},
        {"role": "assistant", "content": None, "tool_calls": [CALL]},
        {"role": "tool", "tool_call_id": "call_1", "content": "Found"}, {"role": "user", "content": "Summarize"}]},
    "parallel_calls_and_results": {"tools": [TOOL], "messages": [{"role": "assistant", "content": "Checking", "tool_calls": [CALL, CALL]},
        {"role": "tool", "tool_call_id": "call_1", "content": "First"}, {"role": "tool", "tool_call_id": "call_2", "content": "Second"}]},
    "empty_content": {"messages": [{"role": "system", "content": ""}, {"role": "user", "content": ""},
        {"role": "assistant", "content": ""}, {"role": "user", "content": None}]},
    "text_blocks": {"messages": [{"role": "user", "content": [{"type": "text", "text": "First "},
        {"type": "text", "text": "second"}]}]},
    "mid_conversation_developer": {"messages": [{"role": "user", "content": "First"}, {"role": "assistant", "content": "OK"},
        {"role": "system", "content": "New policy"}, {"role": "user", "content": "Next"}]},
    "effort_low": {"reasoning_effort": "low", "messages": [{"role": "user", "content": "Hi"}]},
    "effort_high": {"reasoning_effort": "high", "messages": [{"role": "user", "content": "Hi"}]},
    "unicode_tool_schema": {"tools": [{"type": "function", "function": {"name": "weather",
        "description": "Weather in 台北 (°C)", "parameters": {"type": "object", "properties": {}}}}],
        "messages": [{"role": "user", "content": "台北?"}]},
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=pathlib.Path)
    args = parser.parse_args()
    source = args.snapshot / "chat_template.jinja"
    template = _compile_jinja_template(source.read_text())
    goldens = []
    for name, request in CASES.items():
        context = copy.deepcopy(request)
        context["add_generation_prompt"] = True
        for message in context["messages"]:
            for call in message.get("tool_calls", []):
                arguments = call["function"]["arguments"]
                if isinstance(arguments, str):
                    call["function"]["arguments"] = json.loads(arguments)
        goldens.append({"name": name, "request": request, "expected": template.render(**context)})
    FIXTURES.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, FIXTURES / "minimax_m3.jinja")
    (FIXTURES / "minimax_template_goldens.json").write_text(json.dumps(goldens, indent=1, ensure_ascii=False) + "\n")
    print(f"{len(goldens)} Jinja2 goldens from {args.snapshot.name}")


if __name__ == "__main__":
    main()
