# Usage history and console access

**With the full log on (the default), user prompts and model outputs, and
with media on (the default) their images and audio, are stored in plain
text on the serving host for 24 hours.** Turn it off on the `/usage`
settings panel, set its retention or cap to zero, or start with
`--usage off` (`USAGE=off` for `run.sh`), which also stops the metadata
tier. The coordinator logs one line at startup saying whether the full log
is on and for how long.

## What is stored, where, how long

Both tiers live in `~/.cache/cuteafd/<instance>/usage/` on the launch host
(`run.sh` uses `default`), mounted into the coordinator, so restarts, WIP
slots and new images keep the history.

| Tier | File | Holds | Default retention | Default cap |
|---|---|---|---|---|
| Metadata | `usage.sqlite` | one row per request: protocol, route, client kind, key label (`k:` + 8 hex of the key's SHA-256), models, session id and source, token counts, timings, outcome, byte sizes. Never payloads. | 7 days | 256 MiB |
| Daily rollups | `usage.sqlite` (`daily`) | per day × protocol × client × model counts, token sums and latency histograms; no ids | 90 days | — |
| Full log | `usage-log.sqlite` | request and response payloads as delta entries (below), credentials removed | 24 hours | 1 GiB, media included |
| Media | `media/<sha256[0:2]>/<sha256>.<ext>` | each distinct image or audio blob once, by content hash | while referenced | counted in the full-log cap |

Credentials never reach either file: the log captures bodies only, never
headers; redaction removes credential fields a client may embed in a body
(`authorization`, `x-api-key`, `api_key`, `cookie`, `password`, `secret`,
`access_token`, and any `*_token`, `*_secret`, `*_api_key` name, in any
case); and the server's own API key and console secret (as read at startup)
are replaced by `[REDACTED]` wherever they appear in a payload. `data:` URIs
and base64 image/audio parts (and long base64 strings that decode to a known
image, audio or PDF format) become `{"$media":{sha256,mime,bytes}}`
references; with media on the bytes go to the media directory, otherwise
only the reference is kept. Requests that fail authentication store no
payload. Deleted rows are zeroed in place (`secure_delete`) and the WAL is
truncated after every clear and prune, so cleared or expired payloads leave
no plaintext behind. The usage directory is 0700 and its files 0600.
Benchmark requests (those carrying a running benchmark's verified token) are
skipped by both tiers unless the benchmark page's "Record benchmark requests
in usage history" box is ticked (the `record_bench` setting).

## Virtual sessions and delta entries

Agent clients resend the whole conversation every turn. The full log
stores each request as an entry in a virtual session:

- `base`: the first request of a conversation, with its full item list.
- `append`: the request extends an earlier entry's items exactly; the entry
  stores the parent's id, the parent's item count and only the new items.
- `chained`: a Responses `previous_response_id` continuation; the parent's
  output is part of its history.
- `edit`: the history was edited, truncated or rewound; the entry stores the
  full list and the index where it diverged from the session's latest entry.
- `rebased`: the entry's parent expired; it was rewritten with its full
  history and is now the base of what remains of its chain.

Matching runs on the log writer thread over hashes of the redacted items
(prompt-caching markers and single-text-block spelling ignored), only
within one protocol, client kind, API key and session (`x-session-id`,
`previous_response_id`, `prompt_cache_key`, `metadata.user_id`, the
Realtime connection); requests without a session match only others without
one. A request too large to store whole (a string over 256 KiB, or more than
8 MiB of items) is stored as its own truncated base and never becomes a
parent, so later turns always rebuild exactly; the viewer marks it. The system prompt, tools and request settings are
stored once per distinct value. `/console/usage/log/<rid>` rebuilds the
full request by walking parents; the conversation view shows each entry's
new items and response, with repeated assistant turns collapsed and edits
marked.

## The dashboard and console access

`/usage` (USAGE in the page header) shows request analytics: KPI tiles,
token flow, latency, prefix cache, speculation, clients → protocols →
models, sessions, the full-log conversation view, errors, the request list
and settings. Every chart click is a server-side filter over the whole
range. The page is public; its data needs the console cookie.

The launcher prints the console unlock link with its token only to an
interactive terminal (or with `CUTEAFD_PRINT_CONSOLE_LINK=1`); redirected
logs get the secret's path instead. The persistent secret lives at
`~/.cache/cuteafd/console/secret` (0600 in an owned 0700 directory).
Console cookies do not authorize API calls, and an API key does not unlock
console data. Rotate with `scripts/launch/console-secret.sh rotate` or
`run.sh --rotate-console-secret`; running coordinators reload within 10
seconds.

The live console at `/` streams generated token text (`CONSOLE_TEXT=on`,
the default) only to viewers holding the console cookie; others see
`TEXT (unlock)`. While a benchmark run holds the server its synthetic
prompts are shown to everyone. `CONSOLE_TEXT=off` disables text for all
viewers.

## Clearing and exporting

- On `/usage` → Settings: "Clear full log" (payloads and media) or "Clear
  everything" (both tiers), or change retention, caps, media and the
  benchmark switch. `GET/PUT /console/usage/settings` take the same object.
  Changes (`PUT`, `POST`) must come from the page itself: same-origin
  `Sec-Fetch-Site`, a matching `Origin`, or the `x-cuteafd-console: 1` header.
- Without a browser: `cuteafd usage clear-log`, `cuteafd usage clear` and
  `cuteafd usage export-csv [--range 7d] [--bench] [--out FILE]` (metadata
  only), each with `--dir ~/.cache/cuteafd/<instance>/usage`. Stop the
  coordinator first or the running writer may recreate rows.
- With the coordinator stopped, deleting `usage-log.sqlite*` and `media/`
  clears only payloads.
- `cuteafd usage demo --dir DIR [--console-secret-file FILE]` serves
  `/usage` over a synthetic week of history for page work (CPU only).
- `cuteafd gateway … --usage-dir DIR --console-secret-file FILE` records the
  CPU gateway's traffic the same way.
