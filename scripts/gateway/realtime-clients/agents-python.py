#!/usr/bin/env python3
"""README: headless openai-agents-python 0.23.1 RealtimeRunner gate (MIT).

uv pip install --python "$SCRATCH/venv/bin/python" 'openai-agents[voice]==0.23.1'
CUTEAFD_GATEWAY_KEY=<local-key> "$SCRATCH/venv/bin/python" agents-python.py
--url ws://127.0.0.1:8080/v1/realtime --model default [--skip-audio]
--key names an environment variable, never a literal credential. No audio devices.
Runner executes get_time; its model sends tool results. Tracing disabled.
"""
import asyncio
import json
import os

from agents import function_tool, set_tracing_disabled
from agents.realtime import RealtimeAgent, RealtimeRunner
from agents.realtime.model_inputs import RealtimeModelSendRawMessage

from framework_common import Inbox, TIME, main, scenario


async def run(a):
    set_tracing_disabled(True)
    executed = 0

    @function_tool
    async def get_time() -> str:
        """Return a fixed test time."""
        nonlocal executed
        executed += 1
        return json.dumps(TIME)

    agent = RealtimeAgent(name="Local gate", instructions="Answer briefly. Call get_time when requested.", tools=[get_time])
    runner = RealtimeRunner(agent, config={"tracing_disabled": True, "model_settings": {
        "model_name": a.model, "output_modalities": ["text"], "turn_detection": None,
        "input_audio_transcription": None}})
    session = await runner.run(model_config={"url": f"{a.url}?model={a.model}", "api_key": os.environ[a.key_env]})
    inbox = Inbox(os.environ[a.key_env])

    async def receive():
        try:
            async for event in session:
                if event.type == "raw_model_event" and event.data.type == "raw_server_event":
                    inbox.push(event.data.data)
                elif event.type == "error":
                    inbox.fail()
        except Exception:
            inbox.fail()
        finally:
            inbox.fail()

    async with session:
        reader = asyncio.create_task(receive())
        try:
            await inbox.until("session.updated")
            async def send(e):
                await session.model.send_event(RealtimeModelSendRawMessage(message={
                    "type": e["type"], "other_data": {k: v for k, v in e.items() if k != "type"}}))
            async def audio(pcm):
                await session.send_audio(pcm, commit=True)
            await scenario(send, inbox, a, lambda: executed, audio)
        finally:
            reader.cancel()
            await asyncio.gather(reader, return_exceptions=True)


if __name__ == "__main__":
    main("agents-python", run)
