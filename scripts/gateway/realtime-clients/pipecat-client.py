#!/usr/bin/env python3
"""README: headless Pipecat 1.12.0 OpenAIRealtimeLLMService gate (BSD-2-Clause).

uv pip install --python "$SCRATCH/venv/bin/python" 'pipecat-ai[openai]==1.12.0'
CUTEAFD_GATEWAY_KEY=<local-key> "$SCRATCH/venv/bin/python" pipecat-client.py
--url ws://127.0.0.1:8080/v1/realtime --model default [--skip-audio]
--key names an environment variable. No audio devices or hosted bootstrap.
A real pipeline/assistant context aggregator executes get_time and feeds its
result back to the service. Synthetic audio enters as InputAudioRawFrame.
Default mode inherits stock Pipecat serialization: VAD/transcription/tracing
None fields are omitted. --preserve-nulls selects the separate manual-input
subclass that retains these nulls. Both send silent audio and an explicit commit
before the text/tool scenario; silence does not trigger server VAD auto-commit.
Stock mode checks that session.updated retains the server_vad default.
Its typed tool_choice accepts required, not a named function; only get_time is
exposed, so required forces that tool. The initial empty context adds one warmup
response before the scenario. No SDK transport/parser/tool executor is replaced.
"""
import asyncio
import os

os.environ.setdefault("PIPECAT_LOG_LEVEL", "ERROR")
from loguru import logger
logger.remove()  # Framework exception logs can contain endpoint credentials.

from pipecat.frames.frames import InputAudioRawFrame, LLMContextFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.workers.runner import WorkerRunner
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.openai.realtime import events
from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService, OpenAIRealtimeLLMSettings

from framework_common import Inbox, TIME, TOOL, main, scenario


async def run(a):
    inbox = Inbox(os.environ[a.key_env])
    executed = 0
    original_parse = events.parse_server_event

    def observe(message):
        event = original_parse(message)
        inbox.push(event)
        return event

    class ObservedService(OpenAIRealtimeLLMService):
        async def _receive_task_handler(self):
            try:
                await super()._receive_task_handler()
            except Exception:
                inbox.fail()
            finally:
                inbox.fail()

        async def push_error(self, *args, **kwargs):
            inbox.fail()

    class NullPreservingService(ObservedService):
        async def send_client_event(self, event):
            if isinstance(event, events.SessionUpdateEvent):
                # Opt-in manual-input mode; stock mode inherits the serializer.
                payload = event.model_dump(exclude_none=True)
                payload["session"]["audio"]["input"].update(turn_detection=None, transcription=None)
                payload["session"]["tracing"] = None
                await self._ws_send(payload)
            else:
                await super().send_client_event(event)

    service_type = NullPreservingService if a.preserve_nulls else ObservedService
    service = service_type(base_url=a.url, api_key=os.environ[a.key_env], start_audio_paused=False,
        settings=OpenAIRealtimeLLMSettings(model=a.model, session_properties=events.SessionProperties(
            output_modalities=["text"], tools=[TOOL], tool_choice="auto",
            audio=events.AudioConfiguration(input=events.AudioInput(
                format=events.PCMAudioFormat(type="audio/pcm", rate=24000), turn_detection=None)))))

    async def get_time(params):
        nonlocal executed
        executed += 1
        await params.result_callback(TIME)

    service.register_function("get_time", get_time)
    context = LLMContext(messages=[])
    aggregators = LLMContextAggregatorPair(context)
    pipeline = Pipeline([aggregators.user(), service, aggregators.assistant()])
    task = PipelineWorker(pipeline, params=PipelineParams(audio_in_sample_rate=24000, audio_out_sample_rate=24000,
                                                       enable_metrics=False, enable_usage_metrics=False))
    driver = WorkerRunner(handle_sigint=False)
    events.parse_server_event = observe
    running = asyncio.create_task(driver.run(task))
    try:
        updated = await inbox.until("session.updated")
        input_config = updated["session"].get("audio", {}).get("input", {})
        if not a.preserve_nulls and input_config.get("turn_detection", {}).get("type") != "server_vad":
            raise RuntimeError("stock serializer did not retain server VAD default")
        # Empty initial context triggers a response; wait for it before the scenario.
        await task.queue_frame(LLMContextFrame(context))
        await inbox.until("response.done")
        async def send(e):
            classes = {"conversation.item.create": events.ConversationItemCreateEvent,
                       "response.create": events.ResponseCreateEvent}
            if e["type"] == "response.create":
                response = e["response"]
                # Pipecat's typed tool_choice excludes a named function. With
                # only get_time exposed, required forces the same single tool.
                if isinstance(response.get("tool_choice"), dict):
                    response = {**response, "tool_choice": "required"}
                e = {**e, "response": {**response, "output_modalities": ["text"]}}
            await service.send_client_event(classes[e["type"]].model_validate(e))
        async def audio(pcm):
            await service.process_frame(InputAudioRawFrame(audio=pcm, sample_rate=24000, num_channels=1),
                                        FrameDirection.DOWNSTREAM)
            await service.send_client_event(events.InputAudioBufferCommitEvent())
        await scenario(send, inbox, a, lambda: executed, audio, audio_first=True)
    finally:
        await task.cancel()
        await asyncio.gather(running, return_exceptions=True)
        events.parse_server_event = original_parse


if __name__ == "__main__":
    main("pipecat", run, lambda parser: parser.add_argument("--preserve-nulls", action="store_true"))
