# Dashboard

Every coordinator serves two pages next to its API, sharing one header and
navigation. Both are compiled into the binary and need no CDN.

## Live console (`/`)

A family-neutral view of the running engine, fed by every serve loop:

- **Execution lanes:** a time or text view of every lane's prefill, decode and
  verify steps, zoomable from 150 to 1,200 px/s, with pause.
- **Pipeline micro-steps:** the family's own stage breakdown (attention,
  routing, Spark exchange, experts, sampling): last round, 60 s median and max,
  plus a per-layer profile.
- **Draft acceptance** by draft position for the family's speculator (DFlash2,
  dSpark or MTP), rolling 60 s.
- **Prefix cache:** device snapshots, or pinned host KV offload where the family
  has it, with restore latency.
- **Mapped tables** and **recent requests**.

Token text is off by default. `CONSOLE_TEXT=on` in the launch config
(`--console-text`) streams generated text to anyone who can open the page.
`/v1/console/snapshot` returns the same state as JSON. Producers skip all
per-round console work while nobody is watching.

## Benchmark (`/bench`)

Runs `cuteafd bench` profiles against the server it is served from.
`ENABLE_BENCH=on` turns it on and requires an API key; controls need that key,
even on private networks. While a run is active, other clients' inference
requests get 503 with `Retry-After`, so measurements are not shared.

Panels:

- **Speed:** decode by content, concurrency sweep, prefill matrix, decode vs
  retained context, prefix cache.
- **Quality:** fidelity (quick and standard tiers, plus full), structured output,
  long-context needle, math, instruction following, code pass@1, tool eval,
  reasoning effort.
- **Agentic session:** a replayed coding-agent session.
- **Context:** hardware, configuration and startup.

Save custom profiles from the run dialog. Each panel exports as SVG or PNG;
the whole run exports as a report (SVG/PNG), a share card (SVG/PNG) or JSON,
and saved JSON reports load back into the page. A report with a failed check is watermarked
UNVERIFIED in every export. `cuteafd bench publish` turns saved reports into
the cards in this repository; see the [fidelity design](fidelity-design.md)
for how quality is scored.
