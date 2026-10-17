#!/usr/bin/env python3
"""V4.1 prompt-encoding conformance cases from a checkpoint's own encoding.py.

Imports SNAPSHOT/encoding/encoding.py (the checkpoint's reference encoder) and
renders a fixed set of OpenAI chat cases through `encode_messages`: the render
cases of DeepSeek-V4.1-Flash-1017's encoding/test_encoding.py that the
OpenAI chat API can express, plus the checkpoint's encoding/tests fixtures
truncated to their last user or tool turn (an API request always ends there).
Writes JSON {checkpoint, encoding_sha256, cases: [{id, request, prompt}]}
for cuteafd-api's conformance test (rust/crates/cuteafd-api/tests/fixtures/).

    encoding-conformance.py --snapshot SNAP --checkpoint ORG/NAME --out FILE.json

Each request carries an explicit numeric reasoning_effort in thinking mode, so
the rendered effort line does not depend on a server's named-effort policy.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

LOOKUP = {"type": "function", "function": {"name": "lookup", "description": "Look up a value",
          "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}}}}}


def call(call_id: str, name: str = "lookup", arguments: str = "{}") -> dict:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


def unit_cases() -> list[dict]:
    """Render cases mirrored from 1017's encoding/test_encoding.py."""
    drop = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1", "reasoning_content": "r1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "", "reasoning_content": "r2", "tool_calls": [call("c1")]},
        {"role": "tool", "tool_call_id": "c1", "content": "value"},
    ]
    return [
        # test_v41_drop_thinking_without_tools, both assertions.
        dict(id="drop_thinking_tool_result_continues_turn", messages=drop, thinking=True, effort=50),
        dict(id="drop_thinking_next_user_turn", thinking=True, effort=50,
             messages=drop + [{"role": "assistant", "content": "a3", "reasoning_content": "r3"},
                              {"role": "user", "content": "q3"}]),
        # The same histories with tools declared keep every turn's reasoning.
        dict(id="tools_keep_reasoning_across_turns", thinking=True, effort=50, tools=[LOOKUP],
             messages=drop + [{"role": "assistant", "content": "a3", "reasoning_content": "r3"},
                              {"role": "user", "content": "q3"}]),
        # Several tool results in one turn, then a user follow-up in the same block.
        dict(id="drop_thinking_tool_results_then_user_text", thinking=True, effort=50,
             messages=[{"role": "user", "content": "q1"},
                       {"role": "assistant", "content": "", "reasoning_content": "r1", "tool_calls": [call("a"), call("b")]},
                       {"role": "tool", "tool_call_id": "b", "content": "rb"},
                       {"role": "tool", "tool_call_id": "a", "content": "ra"},
                       {"role": "user", "content": "q2"}]),
        # test_v41_mid_conversation_system_message.
        dict(id="mid_conversation_system_message", thinking=True, effort=88,
             messages=[{"role": "system", "content": "sys"}, {"role": "user", "content": "q1"},
                       {"role": "assistant", "content": "a1", "reasoning_content": "r1"},
                       {"role": "system", "content": "mid sys"}]),
        # test_v41_leading_system_message_uses_system_token.
        dict(id="leading_system_message", thinking=False,
             messages=[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "hello"}]),
        # test_v41_chat_mode_has_no_reasoning_effort_or_system_token.
        dict(id="chat_mode_no_effort", thinking=False, messages=[{"role": "user", "content": "hello"}]),
        # test_v41_maps_reasoning_effort_to_1_100_budget (numeric budgets).
        *[dict(id=f"reasoning_effort_{b}", thinking=True, effort=b, messages=[{"role": "user", "content": "question"}])
          for b in (1, 42, 75, 100)],
        # test_v41_tool_instructions_use_spaced_dsml_tags_in_chat_mode.
        dict(id="tool_instructions_chat_mode", thinking=False, tools=[LOOKUP],
             messages=[{"role": "system", "content": "system"}, {"role": "user", "content": "question"}]),
        # test_v41_renders_images.
        dict(id="renders_images", thinking=False,
             messages=[{"role": "user", "content": [{"type": "text", "text": "inspect"},
                                                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}]}]),
        # test_v41_tool_result_images_follow_prompt_order ("tool_messages").
        dict(id="tool_result_images_follow_call_order", thinking=False,
             messages=[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}]},
                       {"role": "assistant", "tool_calls": [call("a", "a"), call("b", "b")]},
                       *[{"role": "tool", "tool_call_id": n, "content": [
                           {"type": "text", "text": f"result {n}"},
                           {"type": "image_url", "image_url": {"url": "data:image/png;base64,AQ=="}}]} for n in ("b", "a")]]),
    ]


PIXEL = "data:image/png;base64,AA=="


def data_urls(value):
    """Image sources render as placeholders; the API accepts only http(s) or data URLs."""
    if isinstance(value, dict):
        if value.get("type") == "image_url":
            return {**value, "image_url": {**value["image_url"], "url": PIXEL}}
        return {k: data_urls(v) for k, v in value.items()}
    if isinstance(value, list):
        return [data_urls(v) for v in value]
    return value


def fixture_cases(encoding_dir: Path) -> list[dict]:
    """encoding/tests inputs the OpenAI chat API expresses, cut after the last user/tool turn."""
    cases = []
    for path in sorted((encoding_dir / "tests").glob("test_input_*.json")):
        data = json.loads(path.read_text())
        case = data if isinstance(data, dict) else {"messages": data}
        messages = data_urls(copy.deepcopy(case["messages"]))
        # latest_reminder, task tokens and system-message tools have no OpenAI chat form.
        if any(m["role"] == "latest_reminder" or "task" in m or (m["role"] == "system" and "tools" in m) for m in messages):
            continue
        for m in messages:
            for i, c in enumerate(m.get("tool_calls") or []):
                c.setdefault("id", f"call_{i}")
        last = max(i for i, m in enumerate(messages) if m["role"] in ("user", "tool"))
        effort = case.get("reasoning_effort")
        effort = {"max": 100, "high": 75, "low": 50}.get(effort, effort)
        thinking = case.get("thinking_mode", "chat") == "thinking"
        cases.append(dict(id=path.stem, messages=messages[:last + 1], thinking=thinking,
                          effort=(effort or 75) if thinking else None, tools=case.get("tools")))
    return cases


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    encoding_dir = a.snapshot / "encoding"
    spec = importlib.util.spec_from_file_location("checkpoint_encoding", encoding_dir / "encoding.py")
    enc = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = enc
    spec.loader.exec_module(enc)
    out = []
    for case in unit_cases() + fixture_cases(encoding_dir):
        messages = copy.deepcopy(case["messages"])
        if case.get("tools"):
            # encoding.py renders tools on a leading system message; an API
            # request's tools render there too (an empty one when absent).
            if messages[0]["role"] != "system":
                messages.insert(0, {"role": "system", "content": ""})
            messages[0] = {**messages[0], "tools": case["tools"]}
        prompt = enc.encode_messages(messages, thinking_mode="thinking" if case["thinking"] else "chat",
                                     reasoning_effort=case.get("effort") if case["thinking"] else None)
        request = {"messages": case["messages"], "thinking": {"type": "enabled" if case["thinking"] else "disabled"}}
        if case["thinking"]:
            request["reasoning_effort"] = case["effort"]
        if case.get("tools"):
            request["tools"] = case["tools"]
        out.append({"id": case["id"], "request": request, "prompt": prompt})
    a.out.write_text(json.dumps({"checkpoint": a.checkpoint, "snapshot_revision": a.snapshot.name,
        "encoding_sha256": hashlib.sha256((encoding_dir / "encoding.py").read_bytes()).hexdigest(),
        "cases": out}, ensure_ascii=False, indent=1) + "\n")
    print(f"{a.out}: {len(out)} cases")


if __name__ == "__main__":
    main()
