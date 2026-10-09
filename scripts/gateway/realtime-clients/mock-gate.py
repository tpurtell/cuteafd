#!/usr/bin/env python3
"""README: local-only mock gate and sanitized real-client handshake captures.

Use the scratch venv (openai==3.26.1, websockets==16.1.1,
openai-agents[voice]==0.23.1, pipecat-ai[openai]==1.12.0).
CUTEAFD_RT_NODE_ROOT="$SCRATCH/node" "$SCRATCH/venv/bin/python" mock-gate.py
--capture-dir "$HOME/.cache/cuteafd/builds/api-gateway/rtclients/captures"
Runs Python GA+beta, Node WS/native, Agents JS/Python and Pipecat on loopback.
Use --clients NAME... / --modes NAME... to select cases. Generates an
isolated TLS CA with openssl for Node's wss-only SDK; never disables verification.
Exercises success, explicit error, failed response, malformed event, disconnect,
missing tool and timeout. This is a client/transport gate, not server conformance.
Captures contain exact first wire frames and headers with credentials redacted.
"""
import argparse
import asyncio
import base64
import json
import os
from pathlib import Path
import secrets
import ssl
import subprocess
import sys
from urllib.parse import parse_qs, urlsplit

from websockets.asyncio.server import serve

HERE = Path(__file__).resolve().parent
SENSITIVE = {"authorization", "x-api-key", "api-key", "cookie", "set-cookie", "sec-websocket-protocol"}


def sanitize(value, key):
    if isinstance(value, dict):
        return {k: "<redacted>" if k.lower() in SENSITIVE else sanitize(v, key) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize(v, key) for v in value]
    return value.replace(key, "<redacted>") if isinstance(value, str) else value


def certificate(directory):
    cert, key = directory / "loopback-cert.pem", directory / "loopback-key.pem"
    subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1",
                    "-keyout", str(key), "-out", str(cert), "-subj", "/CN=localhost",
                    "-addext", "subjectAltName=DNS:localhost,IP:127.0.0.1"],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    key.chmod(0o600)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert, key)
    return context, cert


async def exercise(client, mode, capture_dir, context, cert):
    credential = secrets.token_urlsafe(32)
    capture = {"client": client, "mode": mode, "headers": {}, "first_frames": [], "first_frames_raw": [], "frame_types": [],
               "tool_result_received": False, "audio_bytes": 0}
    event_id = 0
    response_id = 0
    session = {"id": "sess_mock", "object": "realtime.session", "type": "realtime", "model": "default"}

    async def handler(ws):
        nonlocal event_id, response_id
        path = urlsplit(ws.request.path)
        query = parse_qs(path.query)
        capture["path"] = path.path
        capture["query"] = sanitize(query, credential)
        capture["headers"] = {k: "<redacted>" if k.lower() in SENSITIVE else sanitize(v, credential)
                              for k, v in ws.request.headers.raw_items()}
        capture["subprotocol"] = sanitize(ws.subprotocol, credential)
        capture["subprotocols_offered"] = ["<redacted>" if p.startswith("openai-insecure-api-key.") else p
                                          for p in ws.request.headers.get("sec-websocket-protocol", "").split(", ") if p]
        assert path.path == "/v1/realtime", path.path
        assert query == {"model": ["default"]}, query
        if client == "openai-node-native":
            assert f"openai-insecure-api-key.{credential}" in ws.request.headers.get("sec-websocket-protocol", "")
        else:
            assert ws.request.headers.get("authorization") == f"Bearer {credential}"
        if mode == "disconnect":
            await ws.close()
            return
        if mode == "malformed":
            await ws.send('{"type":')
            return
        if mode == "timeout":
            await ws.wait_closed()
            return

        async def emit(typ, **fields):
            nonlocal event_id
            event_id += 1
            await ws.send(json.dumps({"type": typ, "event_id": f"evt_{event_id}", **fields}))

        await emit("session.created", session=session)
        async for raw in ws:
            frame = json.loads(raw)
            capture["frame_types"].append(frame["type"])
            if len(capture["first_frames"]) < 6:
                capture["first_frames"].append(sanitize(frame, credential))
                capture["first_frames_raw"].append(sanitize(raw, credential))
            if frame["type"] == "session.update":
                session.update(frame["session"])
                await emit("session.updated", session=session)
            elif frame["type"] == "conversation.item.create":
                item = frame["item"]
                item.setdefault("id", f"user_{event_id}")
                item.setdefault("object", "realtime.item")
                if item["type"] == "function_call_output":
                    assert item["call_id"] == "call_mock"
                    assert json.loads(item["output"]) == {"time": "2000-01-01T00:00:00Z"}
                    capture["tool_result_received"] = True
                await emit("conversation.item.added" if client == "pipecat" else "conversation.item.created",
                           item=item, previous_item_id=None)
            elif frame["type"] == "response.create":
                if mode == "error":
                    await emit("error", error={"type": "invalid_request_error", "code": "mock_error",
                                              "message": f"mock error {credential}", "param": None})
                    continue
                response_id += 1
                rid, iid = f"resp_{response_id}", f"item_{response_id}"
                response = {"id": rid, "object": "realtime.response", "status": "in_progress", "status_details": None, "output": []}
                await emit("response.created", response=response)
                choice = frame.get("response", {}).get("tool_choice")
                call = (isinstance(choice, dict) or choice == "required") and mode != "missing-tool"
                common = {"response_id": rid, "item_id": iid, "output_index": 0}
                if call:
                    item = {"id": iid, "object": "realtime.item", "type": "function_call", "status": "in_progress",
                            "name": "get_time", "call_id": "call_mock", "arguments": ""}
                    await emit("response.output_item.added", response_id=rid, output_index=0, item=item)
                    await emit("conversation.item.added", item=item, previous_item_id=None)
                    await emit("response.function_call_arguments.delta", **common, call_id="call_mock", delta="{}")
                    await emit("response.function_call_arguments.done", **common, call_id="call_mock", name="get_time", arguments="{}")
                    item.update(status="completed", arguments="{}")
                else:
                    item = {"id": iid, "object": "realtime.item", "type": "message", "status": "in_progress",
                            "role": "assistant", "content": []}
                    await emit("response.output_item.added", response_id=rid, output_index=0, item=item)
                    await emit("conversation.item.added", item=item, previous_item_id=None)
                    beta = "OpenAI-Beta" in ws.request.headers
                    prefix = "response.text" if beta else "response.output_text"
                    await emit(prefix + ".delta", **common, content_index=0, delta="Hello from the local mock.")
                    await emit(prefix + ".done", **common, content_index=0, text="Hello from the local mock.")
                    item.update(status="completed", content=[{"type": "text" if beta else "output_text", "text": "Hello from the local mock."}])
                await emit("response.output_item.done", response_id=rid, output_index=0, item=item)
                await emit("conversation.item.done", item=item, previous_item_id=None)
                response.update(status="failed" if mode == "failed-response" else "completed", output=[item],
                                usage={"total_tokens": 2, "input_tokens": 1, "output_tokens": 1,
                                       "input_token_details": {"text_tokens": 1, "audio_tokens": 0, "cached_tokens": 0},
                                       "output_token_details": {"text_tokens": 1, "audio_tokens": 0}})
                await emit("response.done", response=response)
            elif frame["type"] == "input_audio_buffer.append":
                pcm = base64.b64decode(frame["audio"], validate=True)
                assert pcm == bytes(12000)
                capture["audio_bytes"] += len(pcm)
            elif frame["type"] == "input_audio_buffer.commit":
                assert capture["audio_bytes"] == 12000
                await emit("input_audio_buffer.committed", item_id="audio_mock", previous_item_id=None)
            else:
                raise AssertionError(f"Unexpected client frame {frame['type']}")

    tls = client.startswith("openai-node")
    async with serve(handler, "127.0.0.1", 0, ssl=context if tls else None,
                     subprotocols=["realtime"] if client == "openai-node-native" else None) as server:
        port = server.sockets[0].getsockname()[1]
        url = f"{'wss' if tls else 'ws'}://127.0.0.1:{port}/v1/realtime"
        if client in ("agents-python", "pipecat"):
            script = "pipecat-client" if client == "pipecat" else client
            cmd = [sys.executable, str(HERE / f"{script}.py")]
        elif client.startswith("openai-python"):
            cmd = [sys.executable, str(HERE / "openai-python.py")]
            if client.endswith("beta"):
                cmd += ["--beta"]
        else:
            script = "openai-node" if client == "openai-node-native" else client
            cmd = ["node", str(HERE / f"{script}.mjs")]
            if client == "openai-node-native":
                cmd += ["--native"]
        cmd += ["--url", url, "--timeout", "3" if mode == "timeout" else "15"]
        env = {**os.environ, "CUTEAFD_GATEWAY_KEY": credential, "NODE_EXTRA_CA_CERTS": str(cert),
               "OPENAI_API_KEY": "", "OPENAI_AGENTS_DISABLE_TRACING": "1"}
        proc = await asyncio.create_subprocess_exec(*cmd, env=env, stdout=asyncio.subprocess.PIPE,
                                                   stderr=asyncio.subprocess.PIPE)
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), 25)
        except TimeoutError:
            proc.kill()
            await proc.wait()
            raise AssertionError(f"{client}/{mode} hung")
        capture["exit_code"] = proc.returncode
        capture["stdout"] = sanitize(stdout.decode(), credential)
        capture["stderr"] = sanitize(stderr.decode(), credential)
        assert credential not in json.dumps(capture)
        (capture_dir / f"{client}-{mode}.json").write_text(json.dumps(capture, indent=2) + "\n")
        if mode == "success":
            assert proc.returncode == 0, capture["stderr"]
            assert capture["tool_result_received"]
            assert capture["audio_bytes"] == 12000
        else:
            assert proc.returncode != 0, f"{client} accepted {mode}"
        print(f"PASS {client}/{mode}", flush=True)


async def main(a):
    directory = Path(a.capture_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    context, cert = certificate(directory)
    try:
        for client in a.clients:
            for mode in a.modes:
                await exercise(client, mode, directory, context, cert)
    finally:
        (directory / "loopback-key.pem").unlink(missing_ok=True)
        cert.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", required=True)
    parser.add_argument("--clients", nargs="+", default=["openai-python", "openai-python-beta", "openai-node",
                        "openai-node-native", "agents-js", "agents-python", "pipecat"])
    parser.add_argument("--modes", nargs="+", default=["success", "error", "failed-response", "malformed",
                        "disconnect", "missing-tool", "timeout"])
    asyncio.run(main(parser.parse_args()))
