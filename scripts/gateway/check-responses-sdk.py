#!/usr/bin/env python3
"""Offline official-SDK accumulator check against the Rust Scripted fixture.

Run with the server URL as the only argument. The fixture is the ignored Rust
`gateway::responses::tests::serve_sdk_fixture` test; it serves text, reasoning,
function, custom and max-token responses in that order. No upstream calls.
"""
import sys

from openai import OpenAI


def check(base_url: str) -> None:
    client = OpenAI(base_url=base_url, api_key="offline-fixture", max_retries=0)
    tools = [
        {"type": "function", "name": "f", "parameters": {"type": "object"}},
        {"type": "custom", "name": "patch", "format": {"type": "text"}},
    ]
    for expected in ("text", "reasoning", "function_call", "custom_tool_call", "incomplete"):
        with client.responses.stream(
            model="gpt-6.1-sol", input="fixture", tools=tools,
            reasoning={"summary": "auto"}, include=["reasoning.encrypted_content"],
        ) as stream:
            events = list(stream)
            # The SDK accumulator only finalizes completed responses; incomplete
            # responses carry the final snapshot on their terminal event instead.
            response = (events[-1].response if expected == "incomplete"
                        else stream.get_final_response())
        assert [event.sequence_number for event in events] == list(range(len(events)))
        if expected == "text":
            assert response.output_text == "hello world"
        elif expected == "reasoning":
            assert response.output[0].summary[0].text == "trace"
            assert response.output[0].encrypted_content.startswith("cuteafd.v1.")
            assert response.output_text == "answer"
        elif expected == "function_call":
            assert response.output[0].arguments == '{"x":1}'
            assert response.output[0].call_id == "call_function"
        elif expected == "custom_tool_call":
            assert response.output[0].input == "patch\ntext"
        else:
            assert response.status == "incomplete"
            assert response.incomplete_details.reason == "max_output_tokens"
        print(f"SDK accumulator: {expected} OK")


if __name__ == "__main__":
    if len(sys.argv) != 2 or not sys.argv[1].startswith("http://127.0.0.1:"):
        raise SystemExit("usage: check-responses-sdk.py http://127.0.0.1:PORT/v1")
    check(sys.argv[1])
