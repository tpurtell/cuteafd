#!/usr/bin/env python3
"""Engine backend byte-exact gate: one conversation as Chat Completions,
Anthropic Messages and OpenAI Responses against a served cuteafd model.

Greedy (temperature 0), thinking off and on, a plain turn, a tool-call turn
(the reply must call the tool) and an image turn when the model has vision.
Each case first sends an untimed one-token warm-up, so the three protocols
decode from the same prefix-cache restore. Per case it compares, across the
three protocols:
- the prompt token hash and length the engine admitted (the server's
  `matched benchmark prompt` log lines; needs CUTEAFD_BENCH_NONCE_SEED set
  in the server and `--log` pointing at its log);
- the generated reasoning, text and tool call (name and parsed arguments);
- completion token counts.

Usage: engine-byte-exact.py --url http://127.0.0.1:PORT --key-file KEY
       [--log SERVER_LOG] [--image PNG] [--out results.json]
No keys are printed. Exit 0 only when every case matches.
"""
import argparse
import base64
import json
import re
import sys
import time
import urllib.request


def post(url, key, path, body, headers=None):
    request = urllib.request.Request(url.rstrip("/") + path, data=json.dumps(body).encode(), method="POST")
    request.add_header("content-type", "application/json")
    request.add_header("authorization", "Bearer " + key)
    for name, value in (headers or {}).items():
        request.add_header(name, value)
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.load(response)


def get(url, key, path):
    request = urllib.request.Request(url.rstrip("/") + path)
    request.add_header("authorization", "Bearer " + key)
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


TOOL = {"name": "read_file", "description": "Read a text file and return its contents.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "File path"}},
                       "required": ["path"]}}


def cases(image_b64):
    yield "plain", [("user", "What is the capital of France? Answer in one short sentence.")], False, None
    yield "tool", [("user", "Show me what is in notes.txt. Use the tool.")], True, None
    yield "tool-result", [("user", "Show me what is in notes.txt. Use the tool."),
                          ("call", {"path": "notes.txt"}), ("result", "buy milk\nwalk the dog"),
                          ("user", "How many tasks are listed?")], True, None
    if image_b64:
        yield "image", [("user", "Describe this image in one short sentence.")], False, image_b64


def chat_body(model, turns, tools, image, thinking, max_tokens):
    messages = [{"role": "system", "content": "You are a concise assistant."}]
    for kind, value in turns:
        if kind == "user":
            content = value
            if image and len(messages) == 1:
                content = [{"type": "text", "text": value},
                           {"type": "image_url", "image_url": {"url": "data:image/png;base64," + image}}]
            messages.append({"role": "user", "content": content})
        elif kind == "call":
            messages.append({"role": "assistant", "content": "", "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "read_file", "arguments": json.dumps(value)}}]})
        elif kind == "result":
            messages.append({"role": "tool", "tool_call_id": "call_1", "content": value})
    body = {"model": model, "messages": messages, "temperature": 0, "max_tokens": max_tokens, "stream": False,
            "thinking": {"type": "enabled" if thinking else "disabled"}}
    if thinking:
        body["reasoning_effort"] = "high"
    if tools:
        body["tools"] = [{"type": "function", "function": TOOL}]
    return body


def messages_body(turns, tools, image, thinking, max_tokens):
    messages = []
    for kind, value in turns:
        if kind == "user":
            content = value
            if image and not messages:
                content = [{"type": "text", "text": value},
                           {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image}}]
            messages.append({"role": "user", "content": content})
        elif kind == "call":
            messages.append({"role": "assistant", "content": [
                {"type": "tool_use", "id": "call_1", "name": "read_file", "input": value}]})
        elif kind == "result":
            messages.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": value}]})
    body = {"model": "claude-sonnet-5", "system": "You are a concise assistant.", "messages": messages,
            "max_tokens": max_tokens, "temperature": 0, "stream": False,
            "thinking": {"type": "enabled" if thinking else "disabled"}}
    if thinking:
        body["output_config"] = {"effort": "high"}
    if tools:
        body["tools"] = [{"name": TOOL["name"], "description": TOOL["description"], "input_schema": TOOL["parameters"]}]
    return body


def responses_body(turns, tools, image, thinking, max_tokens):
    items = []
    for kind, value in turns:
        if kind == "user":
            content = [{"type": "input_text", "text": value}]
            if image and not items:
                content.append({"type": "input_image", "image_url": "data:image/png;base64," + image})
            items.append({"type": "message", "role": "user", "content": content})
        elif kind == "call":
            items.append({"type": "function_call", "call_id": "call_1", "name": "read_file", "arguments": json.dumps(value)})
        elif kind == "result":
            items.append({"type": "function_call_output", "call_id": "call_1", "output": value})
    body = {"model": "gpt-6.1-sol", "instructions": "You are a concise assistant.", "input": items,
            "max_output_tokens": max_tokens, "temperature": 0, "stream": False, "store": False,
            "reasoning": {"effort": "high" if thinking else "none", "summary": "auto"}}
    if tools:
        body["tools"] = [{"type": "function", **TOOL}]
    return body


def normalize_chat(response):
    message = response["choices"][0]["message"]
    calls = [(c["function"]["name"], json.loads(c["function"]["arguments"] or "{}")) for c in message.get("tool_calls") or []]
    return {"reasoning": message.get("reasoning_content") or "", "text": message.get("content") or "", "calls": calls,
            "output_tokens": response["usage"]["completion_tokens"], "input_tokens": response["usage"]["prompt_tokens"]}


def normalize_messages(response):
    text = "".join(b["text"] for b in response["content"] if b["type"] == "text")
    reasoning = "".join(b["thinking"] for b in response["content"] if b["type"] == "thinking")
    calls = [(b["name"], b["input"]) for b in response["content"] if b["type"] == "tool_use"]
    usage = response["usage"]
    return {"reasoning": reasoning, "text": text, "calls": calls, "output_tokens": usage["output_tokens"],
            "input_tokens": usage["input_tokens"] + usage.get("cache_read_input_tokens", 0) + usage.get("cache_creation_input_tokens", 0)}


def normalize_responses(response):
    text, reasoning, calls = "", "", []
    for item in response["output"]:
        if item["type"] == "message":
            text += "".join(p.get("text", "") for p in item["content"])
        elif item["type"] == "reasoning":
            reasoning += "".join(p.get("text", "") for p in item.get("content") or item.get("summary") or [])
        elif item["type"] == "function_call":
            calls.append((item["name"], json.loads(item["arguments"] or "{}")))
    usage = response["usage"]
    return {"reasoning": reasoning, "text": text, "calls": calls, "output_tokens": usage["output_tokens"],
            "input_tokens": usage["input_tokens"]}


HASH = re.compile(r"matched benchmark prompt.*?prompt_tokens=(\d+) prompt_token_hash=([0-9a-f]{16})")
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def admitted(log, since):
    if not log:
        return []
    with open(log, errors="replace") as f:
        f.seek(since)
        return HASH.findall(ANSI.sub("", f.read()))


def log_size(log):
    if not log:
        return 0
    with open(log, "rb") as f:
        return f.seek(0, 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--key-file", required=True)
    parser.add_argument("--log")
    parser.add_argument("--image")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--out")
    args = parser.parse_args()
    key = open(args.key_file).read().strip()
    record = get(args.url, key, "/v1/models")["data"][0]
    model = record["id"]
    vision = bool((record.get("capabilities") or {}).get("vision"))
    image = base64.b64encode(open(args.image, "rb").read()).decode() if args.image and vision else None
    results, failures = [], []
    for name, turns, tools, case_image in cases(image):
        for thinking in (False, True):
            row = {"case": name, "thinking": thinking}
            outputs, hashes = {}, {}
            # Untimed warm-up: every protocol then restores the same prompt
            # from the prefix cache, so all three decode under one engine
            # condition (a cold prefill is not bit-identical to a restore).
            warm = chat_body(model, turns, tools, case_image, thinking, 1)
            post(args.url, key, "/v1/chat/completions", warm)
            for protocol, path, build, norm, headers in [
                ("chat", "/v1/chat/completions", lambda: chat_body(model, turns, tools, case_image, thinking, args.max_tokens), normalize_chat, None),
                ("messages", "/v1/messages", lambda: messages_body(turns, tools, case_image, thinking, args.max_tokens), normalize_messages,
                 {"anthropic-version": "2023-06-01"}),
                ("responses", "/v1/responses", lambda: responses_body(turns, tools, case_image, thinking, args.max_tokens), normalize_responses, None),
            ]:
                before = log_size(args.log)
                started = time.time()
                outputs[protocol] = norm(post(args.url, key, path, build(), headers))
                outputs[protocol]["seconds"] = round(time.time() - started, 2)
                time.sleep(0.3)
                seen = admitted(args.log, before)
                hashes[protocol] = seen[-1] if seen else None
            row["outputs"] = outputs
            row["prompt"] = hashes
            reference = outputs["chat"]
            same = all(outputs[p][k] == reference[k] for p in ("messages", "responses")
                       for k in ("reasoning", "text", "calls", "output_tokens", "input_tokens"))
            same_prompt = None if not args.log else len({hashes[p] for p in hashes}) == 1 and hashes["chat"] is not None
            row["match"] = same and same_prompt is not False
            row["prompt_match"] = same_prompt
            if name == "tool" and not reference["calls"]:
                row["note"] = "model answered without calling the tool"
            if not row["match"]:
                failures.append(f"{name} thinking={thinking}")
            results.append(row)
            print(f"{name:12s} thinking={str(thinking):5s} match={row['match']} prompt={same_prompt} "
                  f"in={reference['input_tokens']} out={reference['output_tokens']} calls={reference['calls']} "
                  f"text={reference['text'][:60]!r}", flush=True)
    summary = {"model": model, "vision": vision, "cases": results, "failures": failures}
    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=1)
    print("PASS" if not failures else "FAIL: " + ", ".join(failures))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
