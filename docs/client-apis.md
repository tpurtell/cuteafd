# Client APIs: Claude Code, Codex and Realtime

Besides OpenAI Chat Completions (`/v1/chat/completions`), cuteafd serves the
APIs that the Claude Code and Codex CLIs and Realtime voice clients speak, so
they can run against your own model with no proxy in between.

- **Anthropic Messages:**
  - `POST /v1/messages`, streaming and non-streaming, with tools, thinking,
    images and stop reasons;
  - `POST /v1/messages/count_tokens`;
  - the Anthropic model listing.
- **OpenAI Responses:**
  - `POST /v1/responses`, over SSE or WebSocket;
  - `GET`/`DELETE /v1/responses/{id}` and `input_items`;
  - `previous_response_id`;
  - function, custom (freeform) and `local_shell` tools;
  - `/v1/responses/compact` and `input_tokens`;
  - a Codex model catalog at `/v1/codex/models.json`.
- **OpenAI Realtime:** a `GET /v1/realtime` WebSocket, with GA and beta event
  names.
  - Text and function calling work. Audio input works on audio-capable models.
  - Speech output and transcription are not available yet. Requests for them
    get an explicit error.

Every serving family mounts these routes beside `/v1/chat/completions`, over
the same engine: a Messages or Responses turn renders through the model's own
chat template, tool syntax and grammar, exactly as the chat route would. Turn
them off with `--gateway off` (launchers: `GATEWAY=off`). Structured output
(`json_schema`, strict tools) answers unsupported until a per-model probe has
passed; `json_object` works. `count_tokens` is exact (the checkpoint's
tokenizer) for text turns.

**Browser sockets.** A Responses or Realtime WebSocket opened from a web
page of another origin is refused unless it carries the key or the origin is
listed with `--gateway-allow-origin URL`; CLIs and SDKs send no Origin and
are unaffected.

**Keys.** Start the server with `--api-key-file FILE`. Clients send that key
the way they would to the real service:
- `x-api-key` or `Authorization: Bearer` for Messages;
- Bearer for Responses;
- Bearer or the `openai-insecure-api-key.<key>` WebSocket subprotocol for
  Realtime.

**Model names.** `--official-model-names` (on by default when serving)
accepts any requested model id and runs the served model. It also advertises the ids the CLIs look for: Claude
Code's model discovery lists only `claude-*` ids, and Codex has a fixed set of
slugs. To refresh those lists without a rebuild, generate a file with
`scripts/gateway/official-model-names.py` and pass it as
`--official-model-names-file`.

**Web search.** Claude Code's WebSearch and Codex's web search run on the
server: `--gateway-search exa` (needs `EXA_API_KEY`; `--search` on the
`cuteafd gateway` test harness) or `--gateway-search searxng=URL` (no
key; `scripts/gateway/searxng/` runs a local SearXNG).

**Claude Code:**

```sh
export ANTHROPIC_BASE_URL=http://HOST:PORT
export ANTHROPIC_API_KEY=$(cat FILE)
export CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1   # optional: /model picker
claude
```

All of its model slots (`ANTHROPIC_MODEL`,
`ANTHROPIC_DEFAULT_{OPUS,SONNET,HAIKU}_MODEL`, `CLAUDE_CODE_SUBAGENT_MODEL`)
map to the served model.

**Codex CLI,** in `~/.codex/config.toml`:

```toml
model = "gpt-6.1-sol"          # a slug Codex knows, so it keeps its full tool set
model_provider = "cuteafd"
web_search = "live"            # optional: server-side web search

[model_providers.cuteafd]
name = "cuteafd"
base_url = "http://HOST:PORT/v1"
wire_api = "responses"
env_key = "CUTEAFD_API_KEY"    # export CUTEAFD_API_KEY=$(cat FILE)
supports_standalone_web_search = true
# Optional: the served model's real context window and output limit, so
# Codex compacts at the right point instead of using its built-in numbers.
model_catalog_url = "http://HOST:PORT/v1/codex/models.json"
```

Codex's responses-lite mode uses the client-executed `web.run` extension,
not a hosted Responses tool. `supports_standalone_web_search = true` enables
its authenticated `POST /v1/alpha/search` calls. Queries work with either search
provider; Exa also supports page `open`, and `find` searches opened text cached
for the session. `time` works locally; image search, click, screenshot, finance,
weather and sports return explicit unsupported tool output. Cached-mode searches
use the provider's index; page opens only reuse already-opened session pages,
never fetching uncached pages. Reference/page caches
are bounded and expire after an hour of inactivity; reopen URLs if refs expire.

**Realtime:** any client that accepts a custom URL can connect to
`ws://HOST:PORT/v1/realtime?model=...`. These run headless against it:
- the openai-python and openai-node SDKs (Node requires `wss://`);
- Agents JS/Python;
- Pipecat.

Runners are in `scripts/gateway/realtime-clients/`.
