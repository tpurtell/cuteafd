#!/usr/bin/env python3
"""README: Headless official openai-python GA/beta Realtime gate.

Install in scratch: uv venv --python 3.12 "$SCRATCH/venv"; uv pip install
--python "$SCRATCH/venv/bin/python" openai==3.26.1 websockets==17.2
Run: CUTEAFD_GATEWAY_KEY=<local-key> "$SCRATCH/venv/bin/python" openai-python.py
--url ws://127.0.0.1:8080/v1/realtime --model default [--beta] [--skip-audio]
--key/--key-env names an environment variable, NEVER a literal credential.
Requires no audio device; sends text, a forced get_time call/result, and 250ms
PCM16/24kHz silence. Audio commit is a protocol probe, not speech recognition.
No default URL; OpenAI hosts are rejected. TLS uses the system/SSL_CERT_FILE CA.
"""
import argparse
import asyncio
import base64
import json
import os
import sys
from urllib.parse import urlsplit, urlunsplit

from openai import AsyncOpenAI

TOOL = {"type": "function", "name": "get_time", "description": "Return a fixed test time.",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}


def options():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", required=True)
    p.add_argument("--model", default="default")
    p.add_argument("--key", "--key-env", dest="key_env", default="CUTEAFD_GATEWAY_KEY")
    p.add_argument("--timeout", type=float, default=30)
    p.add_argument("--skip-audio", action="store_true")
    p.add_argument("--beta", action="store_true")
    a = p.parse_args()
    u = urlsplit(a.url)
    if (u.scheme not in ("ws", "wss") or not u.hostname or u.username or u.password
            or u.query or u.fragment or u.path.rstrip("/") != "/v1/realtime"
            or u.hostname.lower() == "openai.com" or u.hostname.lower().endswith(".openai.com")):
        p.error("use an explicit non-OpenAI ws(s)://HOST/v1/realtime URL without credentials/query")
    if not os.environ.get(a.key_env):
        p.error(f"set the credential environment variable {a.key_env}")
    return a, u


async def run(a, u):
    base = urlunsplit((u.scheme, u.netloc, "/v1", "", ""))
    http_base = urlunsplit(("https" if u.scheme == "wss" else "http", u.netloc, "/v1", "", ""))
    async with AsyncOpenAI(api_key=os.environ[a.key_env], base_url=http_base, websocket_base_url=base) as client:
        api = client.beta.realtime if a.beta else client.realtime
        async with api.connect(model=a.model) as c:
            async def receive_until(kind):
                while True:
                    event = await c.recv()
                    e = event.model_dump(exclude_none=True)
                    typ = e.get("type")
                    print(typ, flush=True)
                    if typ in ("response.output_text.delta", "response.text.delta"):
                        print(e.get("delta", "").replace(os.environ[a.key_env], "<redacted>"), flush=True)
                    if typ == "error":
                        raise RuntimeError("server error event")
                    if typ == "response.done" and e.get("response", {}).get("status") != "completed":
                        raise RuntimeError("response did not complete successfully")
                    if typ == kind:
                        return e

            await receive_until("session.created")
            session = ({"modalities": ["text"], "input_audio_format": "pcm16", "turn_detection": None}
                       if a.beta else {"type": "realtime", "output_modalities": ["text"],
                                       "audio": {"input": {"format": {"type": "audio/pcm", "rate": 24000},
                                                           "turn_detection": None}}})
            session.update({"tools": [TOOL], "tool_choice": "auto"})
            await c.session.update(session=session)
            await receive_until("session.updated")
            await c.conversation.item.create(item={"type": "message", "role": "user",
                "content": [{"type": "input_text", "text": "Say hello in one short sentence."}]})
            await c.response.create(response={"tool_choice": "none"})
            result = await receive_until("response.done")
            if not any(i.get("type") == "message" for i in result["response"].get("output", [])):
                raise RuntimeError("text response has no message output")
            await c.conversation.item.create(item={"type": "message", "role": "user",
                "content": [{"type": "input_text", "text": "Call get_time now."}]})
            await c.response.create(response={"tool_choice": {"type": "function", "name": "get_time"}})
            result = await receive_until("response.done")
            calls = [i for i in result["response"].get("output", []) if i.get("type") == "function_call"]
            if len(calls) != 1 or calls[0].get("name") != "get_time" or not calls[0].get("call_id"):
                raise RuntimeError("missing or unexpected get_time call")
            if json.loads(calls[0]["arguments"]) != {}:
                raise RuntimeError("unexpected get_time arguments")
            await c.conversation.item.create(item={"type": "function_call_output", "call_id": calls[0]["call_id"],
                "output": '{"time":"2000-01-01T00:00:00Z"}'})
            await c.response.create(response={"tool_choice": "none"})
            await receive_until("response.done")
            if not a.skip_audio:
                await c.input_audio_buffer.append(audio=base64.b64encode(bytes(12000)).decode())
                await c.input_audio_buffer.commit()
                await receive_until("input_audio_buffer.committed")
            print("PASS openai-python", flush=True)


if __name__ == "__main__":
    args, url = options()
    try:
        asyncio.run(asyncio.wait_for(run(args, url), args.timeout))
    except Exception as error:
        # SDK exceptions can include request credentials; print only the class.
        print(f"FAIL openai-python {type(error).__name__}", file=sys.stderr)
        sys.exit(1)
