"""Shared headless scenario for real Python Realtime frameworks."""
import argparse
import asyncio
import json
import os
import sys
from urllib.parse import urlsplit

TOOL = {"type": "function", "name": "get_time", "description": "Return a fixed test time.",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}
TIME = {"time": "2000-01-01T00:00:00Z"}


def options(configure=None):
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True)
    p.add_argument("--model", default="default")
    p.add_argument("--key", "--key-env", dest="key_env", default="CUTEAFD_GATEWAY_KEY")
    p.add_argument("--timeout", type=float, default=30)
    p.add_argument("--skip-audio", action="store_true")
    if configure:
        configure(p)
    a = p.parse_args()
    u = urlsplit(a.url)
    if (u.scheme not in ("ws", "wss") or not u.hostname or u.username or u.password
            or u.query or u.fragment or u.path != "/v1/realtime"
            or u.hostname.lower() == "openai.com" or u.hostname.lower().endswith(".openai.com")):
        p.error("use an explicit non-OpenAI ws(s)://HOST/v1/realtime URL without credentials/query")
    if not os.environ.get(a.key_env):
        p.error("set the credential environment variable")
    return a


class Inbox:
    def __init__(self, key):
        self.key = key
        self.events = asyncio.Queue()

    def fail(self):
        self.events.put_nowait(None)

    def push(self, event):
        e = event.model_dump(exclude_none=True) if hasattr(event, "model_dump") else event
        if not isinstance(e, dict) or not isinstance(e.get("type"), str):
            self.fail()
            return
        print(e["type"], flush=True)
        if e["type"] in ("response.output_text.delta", "response.text.delta"):
            print(e.get("delta", "").replace(self.key, "<redacted>"), flush=True)
        if e["type"] == "error" or (e["type"] == "response.done" and e["response"].get("status") != "completed"):
            self.fail()
        else:
            self.events.put_nowait(e)

    async def until(self, typ):
        while True:
            e = await self.events.get()
            if e is None:
                raise RuntimeError("framework/server protocol error")
            if e["type"] == typ:
                return e


async def scenario(send, inbox, a, executed, audio, *, audio_first=False):
    async def user(text):
        await send({"type": "conversation.item.create", "item": {"type": "message", "role": "user",
                    "content": [{"type": "input_text", "text": text}]}})

    if audio_first and not a.skip_audio:
        await audio(bytes(12000))
        await inbox.until("input_audio_buffer.committed")
    await user("Say hello in one short sentence.")
    await send({"type": "response.create", "response": {"tool_choice": "none"}})
    text = await inbox.until("response.done")
    if not any(i.get("type") == "message" for i in text["response"].get("output", [])):
        raise RuntimeError("missing text output")
    await user("Call get_time now.")
    await send({"type": "response.create", "response": {"tool_choice": {"type": "function", "name": "get_time"}}})
    response = await inbox.until("response.done")
    calls = [i for i in response["response"].get("output", []) if i.get("type") == "function_call"]
    if len(calls) != 1 or calls[0].get("name") != "get_time" or not calls[0].get("call_id"):
        raise RuntimeError("missing or unexpected get_time call")
    if json.loads(calls[0]["arguments"]) != {}:
        raise RuntimeError("invalid get_time arguments")
    final = await inbox.until("response.done")
    if executed() != 1 or not any(i.get("type") == "message" for i in final["response"].get("output", [])):
        raise RuntimeError("framework did not execute tool and return text")
    if not audio_first and not a.skip_audio:
        await audio(bytes(12000))
        await inbox.until("input_audio_buffer.committed")


def main(label, run, configure=None):
    a = options(configure)
    try:
        asyncio.run(asyncio.wait_for(run(a), a.timeout))
        print(f"PASS {label}", flush=True)
    except Exception as error:
        print(f"FAIL {label} {type(error).__name__}", file=sys.stderr)
        sys.exit(1)
