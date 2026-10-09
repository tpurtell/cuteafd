#!/usr/bin/env python3
"""Offline SDK wire gate; run with the Rust scripted_sdk_smoke test server."""
import argparse
from urllib.parse import urlsplit

import anthropic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    args = parser.parse_args()
    parsed = urlsplit(args.base_url)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        parser.error("this offline fixture gate only accepts loopback HTTP servers")
    client = anthropic.Anthropic(
        base_url=args.base_url, api_key="offline-fixture", max_retries=0, timeout=20.0
    )
    params = dict(
        model="claude-offline-fixture", max_tokens=1024,
        messages=[{"role": "user", "content": "Offline fixture prompt"}],
    )
    with client.messages.stream(**params) as stream:
        message = stream.get_final_message()
    assert [b.type for b in message.content] == ["thinking", "text", "tool_use", "tool_use", "text"]
    assert message.content[0].thinking == "offline reasoning"
    assert message.content[0].signature
    assert message.content[1].text == "before"
    assert message.content[2].input == {"a": 1}
    assert message.content[3].input == {"b": 2}
    assert message.content[4].text == "after"
    assert message.stop_reason == "tool_use"
    assert message.usage.input_tokens == 17
    assert message.usage.output_tokens == 9
    nonstream = client.messages.create(**params)
    lhs, rhs = message.model_dump(), nonstream.model_dump()
    lhs.pop("id"); rhs.pop("id")
    assert lhs == rhs, (lhs, rhs)
    with client.messages.stream(**params) as stream:
        partial = stream.get_final_message()
    assert partial.stop_reason == "max_tokens"
    assert partial.content[0].input == {"nested": {"x": 1}}
    nonstream = client.messages.create(**params)
    lhs, rhs = partial.model_dump(), nonstream.model_dump()
    lhs.pop("id"); rhs.pop("id")
    assert lhs == rhs, (lhs, rhs)
    assert client.messages.count_tokens(model=params["model"], messages=params["messages"]).input_tokens > 0
    models = client.models.list(limit=1)
    assert models.data[0].type == "model"
    assert client.models.retrieve(models.data[0].id).id == models.data[0].id
    print("Anthropic SDK stream accumulation, JSON equivalence, counting and models: PASS")


if __name__ == "__main__":
    main()
