#!/usr/bin/env python3
"""Offline-shaped Realtime WebSocket conformance probe for a running gateway.

Requires websockets. Credentials come only from CUTEAFD_API_KEY; no frame or
credential dumps. With --generate this makes ONE backend turn (paid upstream
if configured); the default exercises session/item/audio/error paths only.
"""
import argparse
import asyncio
import base64
import json
import os
from urllib.parse import urlencode


async def run(args):
    import websockets

    query = urlencode({"model": args.model})
    headers = {}
    key = os.environ.get("CUTEAFD_API_KEY")
    if key:
        headers["Authorization"] = "Bearer " + key
    if args.beta:
        headers["OpenAI-Beta"] = "realtime=v1"
    url = args.url.rstrip("/") + "/v1/realtime?" + query
    # websockets 14+ renamed extra_headers; choose without retrying a handshake.
    import inspect

    header_name = (
        "additional_headers"
        if "additional_headers" in inspect.signature(websockets.connect).parameters
        else "extra_headers"
    )
    async with websockets.connect(url, subprotocols=["realtime"], **{header_name: headers}) as ws:
        async def recv():
            event = json.loads(await asyncio.wait_for(ws.recv(), 30))
            assert isinstance(event.get("event_id"), str), "missing server event_id"
            return event

        async def expect(kind):
            event = await recv()
            assert event["type"] == kind, "expected " + kind + ", got " + event["type"]
            return event

        async def send(event):
            await ws.send(json.dumps(event))

        assert ws.subprotocol == "realtime"
        await expect("session.created")
        if args.beta:
            await expect("conversation.created")
        config = {"instructions": "Reply with a single short greeting."}
        if args.beta:
            config.update(modalities=["text"], turn_detection=None)
        else:
            config.update(output_modalities=["text"], audio={"input": {"turn_detection": None}})
        await send({"type": "session.update", "session": config})
        await expect("session.updated")
        await send({"type": "conversation.item.create", "item": {"type": "message", "id": "probe_user", "role": "user", "content": [{"type": "input_text", "text": "Hello"}]}})
        await expect("conversation.item.created" if args.beta else "conversation.item.added")
        if not args.beta:
            await expect("conversation.item.done")
        await send({"type": "conversation.item.retrieve", "item_id": "probe_user"})
        item = await expect("conversation.item.retrieved")
        assert item["item"]["id"] == "probe_user"
        if args.generate:
            await send({"type": "response.create"})
            await expect("response.created")
            delta = False
            while True:
                event = await recv()
                if event["type"] == ("response.text.delta" if args.beta else "response.output_text.delta"):
                    delta = True
                if event["type"] == "error":
                    raise AssertionError("generation returned an error")
                if event["type"] == "response.done":
                    assert event["response"]["status"] == "completed", "generation did not complete"
                    assert delta, "no text delta"
                    break
        await ws.send("{")
        await expect("error")
        await send({"type": "unknown", "event_id": "probe_error"})
        assert (await expect("error"))["error"]["event_id"] == "probe_error"
        await send({"type": "input_audio_buffer.append", "audio": base64.b64encode(bytes(4800)).decode()})
        await send({"type": "input_audio_buffer.commit"})
        await expect("input_audio_buffer.committed")
        await expect("conversation.item.created" if args.beta else "conversation.item.added")
        if not args.beta:
            await expect("conversation.item.done")
        await send({"type": "input_audio_buffer.clear"})
        await expect("input_audio_buffer.cleared")
        await send({"type": "conversation.item.delete", "item_id": "probe_user"})
        await expect("conversation.item.deleted")
    print("Realtime conformance PASS; backend turns requested:", int(args.generate))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="ws://127.0.0.1:8000")
    parser.add_argument("--model", default="gpt-realtime")
    parser.add_argument("--beta", action="store_true")
    parser.add_argument("--generate", action="store_true", help="request one backend turn (may incur upstream cost)")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
