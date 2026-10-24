# Using agents on cuteafd

How cuteafd work is split between an orchestrating Claude session and its
agents, and the rules that keep many agents productive on one shared
cluster. `AGENTS.md` holds the engineering rules every agent follows; this
file is about running the agents themselves. It records what we learned
building v0 and v1 (2026-09-28 onward).

The judgment (who does what, how to brief, how to review) is the same in
every environment. Only the launch mechanics differ:

- **Enhanced Claude Code** (the hybrid launcher): the Agent tool takes a
  `model` (Codex, Astra, DeepSeek, MiMo or Claude) and an `effort`, so every
  agent is launched, resumed and stopped the same way. See
  [Launching: enhanced](#launching-enhanced-claude-code).
- **Plain Claude Code**: the Agent tool runs Claude models only; Codex runs
  through `scripts/agents/codex-launch.sh`. See
  [Launching: plain](#launching-plain-claude-code).

Check which you have at the start of a session: the Agent tool's `model`
list names `claude/chatgpt/…` and `claude/deepseek/…` entries only in the
enhanced build.

## Roles

| Who | Does | Doesn't |
|---|---|---|
| **Orchestrator** (one Claude Opus session) | Plans, writes briefs, picks the model per task, makes default and policy calls, reviews every branch, resolves merge conflicts, merges into `work/p0`, cuts releases, talks to TJ | Long hardware runs, bulk implementation |
| **Codex Sol 6.1** | The workhorse: kernels, ports, measurements, A/B campaigns, release builds and smoke matrices, investigations | Merging, tagging, pushing images, changing defaults without passing gates |
| **Astra 6** (`high`) | Hard kernel and optimization work: new kernel structures, tiling and fusion ideas, roofline gaps, and coding those kernels | Simple kernels, investigation, profiling, probes, harnesses, integration, gates (Sol does these) |
| **DeepSeek 4.1 Flash** | Fast structural work: renames, mechanical refactors, doc and table edits, sweeps across many files, log/report digests, simple scripts | Numerics, kernels, judgment calls |
| **Claude Opus subagents** | Judgment-heavy cross-cutting work: engine design changes (e.g. the device-driven exchange), hard merges, investigations needing many decisions, independent reviews | Work Sol can do from a clear brief |
| **Claude Fable** | Extremely rarely: important design or planning, front-end design, a special kernel insight | Anything else |
| **MiMo** | Only when every other option is exhausted | — |

## Choosing a model

**Balance (TJ, 2026-10-10):** with Claude budget getting tighter, bias bounded,
less judgment-heavy work toward Sol 6.1 high/xhigh on the primary
subscription: parity and A/B runs, per-family ports that follow an existing
template, card definitions, fixtures and test gaps, docs. Opus stays on design,
integration leads and hard calls.

Budget drives the split. Claude is the scarcest: eight parallel Opus agents
once used ~60% of a week's capacity in under a day. Codex Sol has had ample
budget (under 1% of a weekly subscription after a full day of work). With a
larger Claude budget now, Opus can take somewhat more of the judgment work,
but bounded engineering still goes to Sol.

| Task | Model | Effort |
|---|---|---|
| Default engineering, measurement, A/B, release smoke | Sol 6.1 | `high` |
| Subtle numerics, kernels, root-cause investigations | Sol 6.1 | `xhigh` |
| Hard kernel and optimization work: a new kernel structure, a tiling or fusion idea, why a kernel misses its roofline, and writing that kernel | Astra 6 | `high`; it designs and codes the kernel; Sol does the harnesses, gates and integration around it |
| Structural or simple work, fast turnaround | DeepSeek Flash | `high`, `max` for larger sweeps |
| Design changes, hard merges, independent review, decisions | Claude Opus | default |
| Shared build/run tooling answering an agent's tooling request (wip.sh, wip-cards, hwlock, smoke, launchers, cpu-gate) | Claude Opus, or Sonnet for a narrow, well-specified fix | default |
| Important design/planning, front-end design | Claude Fable | default; extremely rarely |
| Everything else exhausted | MiMo | default |

- **Judgment calls and shared tooling go to Claude (TJ, 2026-10-13).** Use Opus agents for important judgment calls, and Opus or Sonnet for shared build-system work done in response to another agent's tooling request: every agent depends on that tooling, so a mistake there costs everyone. Sol stays the default for everything else.
- **Astra for hard kernels (TJ, 2026-10-13):** Astra `high` is good for kernel and optimization concepts and can code those kernels itself. Don't spend it on simple kernels or on probes, harnesses and integration, which Sol does. Astra once burned a subscription week mostly on probe plumbing in one investigation.
- **Rotate models on repeated kernel ideas (TJ, 2026-10-13):** when a kernel or optimization idea needs several iterations, don't give every round to the same model. Rotate the strong models (Astra `high`, Opus, Fable, Sol `xhigh`), each taking a turn with the previous rounds' results and measurements, and pick who goes first at random. Different smart friends see different things. One round per model per turn, then the measured result goes to the next.
- **Fable is rare.** It costs far more per task; use it at a critical minimum (TJ, 2026-10-05). Opus 5.5 and Sol 6.1 are strong, and in some ways better, so default to them for hard problems.
- **Codex subscription:** there is one Codex subscription (TJ cancelled the
  backup one, 2026-10-22). Start agents on the plain `claude/chatgpt/...`
  models; the `-backup` model names no longer authenticate. Astra burns
  budget faster than Sol but less than Fable.
- **Read the error body, not the headline.** The gateway prints "Server is
  temporarily limiting requests (not your usage limit)" for both cases. A
  body with `usage_limit_reached` and `resets_in_seconds` in days is the
  subscription's weekly limit: every Codex agent stops at its next call.
  Relaunch each on another model (Opus, DeepSeek) as a fresh agent
  with a resume note pointing at its PR (`pr-sync pull` restores STATUS.md;
  its background jobs keep running and must not be duplicated). Keep a shared
  `codex-runs/resume-note.md` for this.
- **Stop the original before relaunching.** An agent hit by `usage_limit_reached` can keep running
  for a while (some of its calls still complete). Stop it explicitly and confirm its worktree is
  quiet (`git status` stable) before a replacement takes over that worktree; otherwise two agents
  write the same tree (2026-10-07, MiMo audio).
- **Transient 429s:** a stream cut off after `response.created` with HTTP 429 (no `usage_limit_reached`) is OpenAI-side capacity, not a subscription limit, so switching subscription doesn't help (TJ, 2026-10-05). Resume the agent after a few minutes from its STATUS.md; keep briefs and STATUS current so a mid-task death costs little.
- **Capacity vs limit:** "Selected model is at capacity" (or a similar
  overload error) is the provider being busy, not our quota. Retry after a
  few minutes; pushed commits survive, so resume with a note. A usage-limit
  error means switch model.
- **Budget pacing:** a Codex subscription's week lasted a few hours with six Sol `xhigh`/`high` agents plus Astra `high` in parallel. Keep about three `xhigh` agents at once, default to `high`, and don't let an agent spawn sub-agents at `xhigh` without reason.
- **Parallelism:** run several Sol agents at once on independent tasks
  (separate branches and worktrees, disjoint hardware); serialize only what
  shares a GPU or build cache. Queue Codex work early, it is slower per task
  than Claude.
- **DeepSeek Flash track record (2026-10-05):** four tasks, all correct on
  the first pass: a 7-host image audit and cleanup that followed the keep
  rules exactly; a bench-console fix with unit tests and an unprompted
  headless-browser render check; and a `max`-effort change threading the
  console hub through six serve paths with a drop guard, where it flagged
  its own edge case. Strong at bounded, well-specified, checkable work;
  not yet tried on numerics, kernels, or noisy hardware measurement.
  Stretching it next to multi-file platform items (build hygiene,
  readiness/health).
- Don't drop a numerics or kernel task to DeepSeek to save time; review cost
  outweighs it. Give DeepSeek work whose result is easy to check.

## Writing a brief

Hardware-card briefs use `scripts/bench/wip-cards.py`; see
[WIP hardware cards](docs/wip-cards.md) for the shared kit, arm, probe,
build and cleanup contract. Two commands cover most briefs; paste them:

```sh
# A/B two commits on cards: builds both arms, preflights on the CPU, queues fairly,
# interleaves, and writes summary.md (before -> after per card and metric, first real error).
python3 scripts/bench/wip-cards.py --kit gate --interleave --repeats 1 --cards qwen38-exl3-min \
  --arm base=<task>-b:<slot> --arm cand=<task>-c:<slot> --build base=origin/work/p0 --build cand=HEAD \
  --panels baseline --levels 1,16 --post-load-c1 --trace --binary <cuteafd built from HEAD> --task <task>
# CPU gate: cargo + script suite, verdict from parsed summaries, optional new-failure diff against a base.
scripts/dev/cpu-gate --task <task> [--cargo] [--scripts] [--skip-live-planners] [--base origin/work/p0]
```

Same build, two configurations: `--arm-wip` both arms to one sealed WIP
instance and `--arm-set ARM:CARD:KEY=VALUE`. `--set CARD:PROFILE=...` turns
on profiling everywhere, `--preflight-only` checks without queuing, and
`--resume` reruns only failed jobs. Do not copy an RC driver or take outer serving
locks: `cuteafd bench smoke` takes its own locks. CPU builds/tests run at
nice 19 with 16 jobs (`CARGO_BUILD_JOBS=16`, `RUST_TEST_THREADS=16`).

**A/B rules learned in v3 (2026-10-10):**
- **Same launcher in both arms.** A candidate whose `run.sh` changed runs
  both arms on the candidate's launcher. A baseline-launcher candidate arm
  produced a false V4.1 max C8 loss (0.88-0.95) that reversed (1.09) once
  both arms matched.
- **Adaptive draft C8 swings ±10-18% between launches** on V4.1 max. One
  matched pair cannot decide C8 there. Use 3 interleaved pairs with
  alternating order, plus a fixed-policy control (`DSPARK_DRAFT_POLICY=full`).
  That control exists **only on V4.1**. V4's dSpark proposes a fixed block, but
  which of its rows are verified goes through the shared draft policy, as in
  every family. On V4, isolate speculation with a `SPECULATOR=off` pair, and
  check prompt dependence with a second nonce seed.
  A shared-build toggle (`--arm-wip`) isolates a flag from the build.
- **Judge on the paired median**, not on the ratio of arm medians, and report
  the raw pairs and their order.
- **Builds:** `--build` arms seed on the less-loaded LAN-only seed (kakapo or
  cassowary) by default (`--seed kakapo|cassowary|rhea|moa` places it). LAN-only
  seeds only build (10 GbE; their slots reach the Sparks through sparknest),
  so their CPU load never skews a serving card; rhea and
  moa serve cards too, so wait for an idle one without holding locks; never
  wrap `wip.sh` in a Spark lock yourself.
- **Page cache:** GB10 CUDA allocations reclaim Linux page cache. Keep caches
  warm for serving; SM121 admission uses `MemAvailable` bounded by cgroup
  hard-limit headroom, not raw `cudaMemGetInfo` (which under-reports reclaimable
  cache). Discrete RTX uses CUDA free. Cache eviction belongs only to deliberate
  cold-read benchmarks.
- **Briefs:** point at `scripts/agents/codex-preamble.md` in an up-to-date
  checkout. The main checkout can lag; a stale copy cost one agent a v1-era
  rule set.

The same brief works for every model. One bounded task with everything the
agent needs to finish without asking:

1. **Branch and base:** `work/<task>` off `origin/work/p0`, in a worktree
   outside the repo (the preamble covers this). Name the PR/issue number and
   the preflight command (`hwlock.py --preflight`) for any hardware run.
2. **Context:** the measured numbers and file paths that motivate the task,
   the `PLAN.md` item, what was already tried. Point at earlier reports
   rather than restating them.
3. **Gates:** golden NLL/KL/top-1 bounds, byte-exactness or lossless-spec
   checks, the hardware configs (natural minimum / maximum) and how many
   launches per arm.
4. **Decision rule, in full.** Codex applies thresholds literally. It kept a
   BF16 default because dual-RTX C1 gained 1.35% against a 2% bar, despite
   C4 +15% and C16 +10%. Say which metrics matter most (C1 first for TJ, then
   C4 and the reasoning-on agent session; 32-token micro cases don't matter),
   or ask it to recommend and let the orchestrator decide.
5. **Defaults:** "don't change defaults unless the gates pass", plus what to
   leave opt-in.
6. **Report:** before → after tables with conditions, gate results, commits,
   open issues.

Start every cluster brief with `scripts/agents/codex-preamble.md` (worktree,
build, lock, sudo, frugality and no-merge rules). `codex-launch.sh` prepends
it automatically; for Agent-tool launches, tell the agent to read and follow
it first or paste it in. Restating TJ's standing rules where they apply
avoids rework: single residency (never two formats of one tensor resident),
honor checkpoint numerics, never read code whose licence covers
re-implementations (e.g. b12x PR #342).

DeepSeek briefs can be shorter but must be exact: the files, the
transformation, and how to check it (a grep, a test, a diff shape).

## Launching: enhanced Claude Code

Every agent is an Agent-tool call with `run_in_background` semantics (the
tool returns at once and a completion notification wakes the orchestrator):

```text
Agent(subagent_type="general-purpose",
      model="claude/chatgpt/gpt-6.1-sol",   # see table below
      effort="high",                               # per-launch reasoning effort
      description="<3-5 words>",
      prompt="Read and follow scripts/agents/codex-preamble.md. <brief>")
```

| Model id | Use |
|---|---|
| `claude/chatgpt/gpt-6.1-sol` | Sol 6.1; effort `high`/`xhigh` |
| `claude/chatgpt/gpt-6-astra` | Astra 6; effort `medium`/`high` |
| `claude/deepseek/deepseek-flash` | DeepSeek 4.1 Flash; effort `high`/`max` |
| `opus`, `fable` | Claude subagents |
| `claude/xiaomi/mimo-v2.6-pro` | last resort; no graded effort |

- Keep briefs as files, as in the plain path: write
  `~/.cache/cuteafd/builds/codex-runs/<name>.md` and make the prompt
  "read the preamble, the wave's hardware allocation file and `<name>.md`,
  keep the PR body current with `scripts/dev/pr-sync push`". The brief survives
  a restart, a resume note can be appended to it, and either launch path can
  run it. For a parallel wave, one shared allocation file says which locks
  and hosts each agent owns (e.g. one agent on `gpu1.lock` + a single
  Spark, the rest queued on the shared pool).
- Agents may spawn their own component agents. A component's
  `SendMessage` reaches the orchestrator, not its parent (the parent sees
  only the component's completion). Expect to relay mid-task notes between
  them, and tell components to put anything the parent needs in their final
  report.
- These agents run inside Claude Code's tool harness with this session's
  tools and permissions (Bash, Read, Edit, SSH, `agent-sudo`), not the Codex
  CLI, so `codex-launch.sh`/`codex-stop.sh` don't apply: stop one with
  `TaskStop`, resume one with `SendMessage` (keeps its context), start fresh
  with a new Agent call.
- Long runs: the agent starts one blocking background command and waits on
  it, as in [Waiting](#waiting-and-restarts).
- The launcher must not sandbox the client. The hybrid launcher once wrapped
  it in `bwrap`, a user namespace that broke SSH ("Bad owner or permissions
  on /etc/ssh/…"), `agent-sudo` ("no new privileges") and writes under
  `/mnt`. It now selects the second login with
  `CLAUDE_SECURESTORAGE_CONFIG_DIR` instead. If those errors reappear,
  check `grep NoNewPrivs /proc/self/status` (must be 0) before anything else.

## Launching: plain Claude Code

Claude subagents use the Agent tool (`opus`, `fable`, …) as above. Codex runs
through the CLI wrapper, not the Claude Code Codex plugin: the plugin runs
Codex in a read-only or workspace-write sandbox (no writes to `~/.cache`, no
GPU, no SSH to the Sparks, no `git push`), and every cluster task sent
through it came back blocked with drafts only.

```sh
# 1. Write the brief:    ~/.cache/cuteafd/builds/codex-runs/<name>.md
# 2. Start it as a background command; its exit wakes the orchestrator:
scripts/agents/codex-launch.sh <name> [high|xhigh]
# 3. Read the result:    ~/.cache/cuteafd/builds/codex-runs/<name>.report.md
# Stop a run (whole process tree):
scripts/agents/codex-stop.sh <name>
```

`codex-launch.sh` prepends the preamble and runs `codex exec -s
danger-full-access` under `setsid`, recording its PID. The model is the
Codex CLI default from `~/.codex/config.toml` (`gpt-6.1-sol`); pass
`CODEX_MODEL` for Astra. Override the runs directory with `CODEX_RUNS`.
DeepSeek and MiMo are not available in this environment.

- **Stopping:** always `codex-stop.sh <name>`. Killing only the launcher
  leaves Codex alive as an orphan that keeps editing and collides with any
  relaunch.
- **Relaunching:** append a `RESUME NOTE` to the brief saying what the earlier
  run left (worktree, uncommitted edits, known bugs) and that no other run is
  active. Codex picks up from the worktree state.

## External agents: Grok and Gemini (TJ, 2026-10-14)

Two more subscriptions are available. They can't go through the Agent tool or
the gateway (no official API, and their backends would notice rewritten
requests), so they run as CLIs. Their capacity is entirely the
orchestrator's to use; TJ isn't using these subscriptions.

| CLI | Model | Launch flags | Expected strengths |
|---|---|---|---|
| `grok` (X Premium+) | `grok-4.7` (default), `grok-4.7-build-fast` | `--always-approve`, `--prompt-file`, `--reasoning-effort` | mechanical work; a fallback when Codex is down or out of quota |
| `agy` (Google Antigravity) | `gemini-4-argon-high`/`-medium`/`-low`, `gemini-3.8-flash-*` | `--dangerously-skip-permissions`, `-p`, `--effort`, `--add-dir` | second opinions on hard kernel and technical problems, vision, deep research and search |

Use the interactive stdlib driver for a worker that needs steering or answers:

```sh
python3 scripts/agents/ext-agent.py start TASK --tool grok --effort high \
  --cwd /absolute/worktree --preamble /absolute/preamble --brief /absolute/brief --auto-approve
python3 scripts/agents/ext-agent.py wait TASK --until idle --timeout 600
python3 scripts/agents/ext-agent.py transcript TASK --since-last
python3 scripts/agents/ext-agent.py send TASK 'One concise correction.'
python3 scripts/agents/ext-agent.py stop TASK
```

`--auto-approve` is explicit: without it the CLI keeps its own approval
behavior and a directory-trust prompt surfaces as `asking` for the driver to
answer. Our drivers pass it in a trusted worktree; it authorizes approvals
inside that worktree, not merges, pushes or scope changes.

Give the DeepSeek driver agent
`scripts/agents/skills/ext-agent-driver/SKILL.md` and the worker's brief.
The worker cannot SendMessage; its driver reports milestones, unresolved
questions and validated results to the orchestrator. `wait` blocks instead
of agent-side polling. `idle` is an available input prompt, not proof the
task is done. Busy sends are queued; inspect the answer to verify steering.
`interrupt` uses each CLI's own key: Ctrl+C for Grok, Escape for Antigravity.
Runs and transcripts live in `EXT_AGENT_RUNS` (default `~/.cache/ext-agent`).
Antigravity 503s have bounded automatic backoff; state and retry evidence
are exposed in JSON. Python 3, tmux and authenticated CLIs are required.

The old `ext-launch.sh grok|agy NAME [model] [effort]` remains a single-turn
batch launcher with logs under `AGENT_RUNS`; use `ext-stop.sh` only for those
batch runs. Use `ext-agent.py stop` for interactive runs so its supervisor
and process descendants are stopped too. Never give both launchers the same
run name: both use the `ext-NAME` tmux namespace.

**They are on trial.** Every use is scored in `docs/agent-scorecard.md`:
correctness, judgement, quality, creativity, wall time, plus notes. Trial
patterns:
- **Duels:** give the same task as a Sol or Opus agent to Gemini or Grok in a
  separate worktree, and compare the results. Kernel design and optimization
  are the best duels.
- **Low-priority tasks** on their own.
- **Reviews** of another agent's PR, compared with the code reviewer's.
- **Research and vision** (Gemini): compare with our web search (Exa).

The same gates and rules apply as for every agent: no tooling extensions, no
sudo, no power changes, the fair queue for hardware, public-text hygiene. An
external agent never merges, and never pushes to work/p0 or the fork's master.
Compact the scorecard log into its summary table every ~10 entries, and
update this section's table as the picture firms up.

## Gate execution

The agent Bash tool can evaluate commands in a non-final `&&` operand, where
Bash ignores inline `set -e`. Never run a multi-step gate as an inline
`set -e; build; tests; echo PASS` command. Invoke the standalone entry point
(`scripts/dev/cpu-gate`, or `bash path/to/gate.sh`), or explicitly chain every
required step with `&&` / `|| exit 1`. Do not source a gate or call its shell
function from an `if`/`&&` condition: that also suppresses errexit.

The canonical CPU gate requires Cargo's JSON `build-finished.success` before
running tests, then parses test-result failure counts (or nextest's JUnit
report). Missing/empty reports, timeouts and moved HEADs fail closed, including
with `--base`. Report failing test IDs, not merely exit codes or counts. Commit
before starting the gate; never move HEAD during it.

## Running agents

- Workers update current state with `scripts/dev/pr-sync push` at checkpoints
  and post verdicts/seals with `pr-sync comment`. The PR body is authoritative;
  `pr-sync pull` restores a local STATUS.md. Only message the coordinator for
  a merge-ready SHA, verdict, decision or blocker. Stacked PRs target their
  parent branch; the coordinator comments rather than rewriting worker bodies.
- **One agent per task.** Never start a second agent on a task that is still
  running (`codex-launch.sh` refuses to). A duplicate finds the first one's
  worktree, backs off, and reports an "ownership" question instead of
  working.
- **Resume, don't restart:** continue a stopped or finished agent with its
  context (`SendMessage`, or a resume note for the CLI) and tell it to check
  its own jobs before relaunching anything.

### Waiting and restarts

- Agents wait on long runs with one blocking background command and wake on
  its completion. The orchestrator likewise waits for completion
  notifications; don't poll logs in between, peek only when TJ asks.
- **Before a Claude Code restart or a usage-limit reset,** have every agent
  checkpoint: commit and push WIP, then `scripts/dev/pr-sync push` the PR
  body (branch, head, gates, exact result-collection commands, evidence paths,
  next steps). After the restart, `pr-sync pull` and resume from the PR.
  Background shells and their completion watchers die with the session, so
  finished jobs must be collected by hand.
- A session limit stops every agent at once; resume each with a short note.
- **Launch every hardware run, golden and long build as a systemd user
  unit:** `systemd-run --user --unit=<task-name> --collect
  --setenv=PATH="$PATH" --setenv=HOME="$HOME" --working-directory=<dir>
  bash -c '<cmd> > <log> 2>&1; echo exit $? >> <log>'`. `nohup setsid`
  is not enough: the job stays in the Claude session's `tmux-spawn-*.scope`
  cgroup and dies with it, queued hwlock tickets included (twice on
  2026-10-22). Check `/proc/<pid>/cgroup` reads `app.slice/<name>.service`.
  Pass PATH and HOME explicitly: a user unit otherwise gets systemd's
  minimal PATH, and host-side Cargo exits 127. Transient units live only in
  `/run` and unload when the command exits, so they leave nothing behind.
  systemd expands `$VAR`/`${...}` in the unit command line itself, which
  mangles bash arrays, `$?` and `$(...)` meant for the inner shell: pass
  `--expand-environment=no`, or put the commands in a script file and run
  `bash /path/script.sh`.
- Keep systemd's stop defaults on those units (KillMode=control-group,
  TimeoutStopSec=90 s): never pass `-p KillMode=mixed|process` or a shorter
  `TimeoutStopSec`. On stop, `cuteafd bench smoke` removes its coordinator
  and Spark workers (bounded, 60 s) and `hwlock.py` / `wip-cards.py` forward
  the signal and keep the locks until it has, so the next lock holder finds
  clean hardware.
- Stop your own jobs by PID or `systemctl --user stop <unit>`, never with
  `pkill -f` (the pattern can match your own Bash command line and kill it),
  `kill -9` or `systemctl kill -s KILL` (no teardown runs; the containers
  outlive the released locks).

### Known Codex behavior

- Follows gates and AGENTS.md rules carefully and reports missed gates
  honestly, including its own mistakes.
- Conservative on judgment calls; rarely pushes a borderline result over the
  line. That's acceptable because the orchestrator decides.
- Applies thresholds literally (see "Decision rule").
- Writes reusable tooling along the way (gate scripts, probes, profilers).
- Has made questionable policy calls when left to decide defaults (the
  codex/v1 "honor checkpoint precision" refusals); always review defaults.

## Shared cluster discipline

These apply to every agent and are also in `AGENTS.md`.

- **Locks:** take `sparks.lock`, then `gpu0.lock` when using GPU0 (TJ may hold it for an interactive server), then `gpu1.lock`, only around actual runs,
  each run one blocking command with a timeout. Reversed lock order
  deadlocked the cluster with both GPUs idle and ~30 jobs queued.
- **Teardown:** stop servers, containers and Spark workers before releasing
  the locks (lock scripts should trap exit). A stray server held 30 GB of GPU1
  for two hours and broke another agent's measurements.
- **Watchdogs:** device-side waits and long runs get timeouts. A device-mode
  hang held both locks for ~4 hours before a watchdog existed.
- **Disk:** builds fill raptor's root NVMe fast (`builds/` reached 1.4 TB and
  crashed runs with "No space left on device"). Delete Cargo `target*` and
  release staging when a task finishes; check `df -h /` before large builds.
- **AOT exports outside the locks:** CuTe/Triton exporters query the device
  at compile time, so the export container needs a GPU, but not a lock.
  Pin it to an idle RTX (≤512 MiB used, bounded wait), watchdog its
  memory, and on a Spark export only when no serving container runs and
  ≥100 GiB CUDA memory is free. Log host, GPU and time so a concurrent
  measurement can be explained.
- **Root:** run privileged commands directly as
  `agent-sudo -n --agent-context "<why>" <command>` with the whole command
  visible. A wrapper script under sudo (`sudo python profile.py`) is flagged by
  the approval engine; TJ pre-approved `ncu` itself.
- **Releases:** only the orchestrator tags, moves `main`, or pushes images, and
  only with TJ's approval. A release run fail-stops if any spot-smoke fails;
  check that no other agent is using the GPUs before release measurements
  (an overlapping kernel build once made V4.1 look 15% slower).

## Reviewing and merging

1. Read the report, then check the branch: `git merge-base --is-ancestor
   origin/work/p0 origin/<branch>` and `git diff --stat origin/work/p0...origin/<branch>`.
2. Merge in a temporary worktree. On conflicts, keep both sides' intent; if
   an agent wrote the branch and the conflict is substantial, send it back to
   that agent to merge `origin/work/p0` and re-verify.
3. Check launcher keys: every key `run-family.sh` reads must be in
   `release_known_key` (`scripts/lib/release-common.sh`).
4. Tests: `cargo test --workspace`, and compare failing script-test ids
   against `origin/work/p0` (add none).
5. Push to `work/p0` (fast-forward when possible).
6. Override a conservative call when the measurements justify it, and record
   the reason in the commit (e.g. the V4.1 FP8 vocabulary head default).

For a second opinion on a risky branch, have a different model review it
than the one that wrote it (Opus or Astra reviewing Sol), given the diff and
the gates, not the author's conclusion.

## Keeping this guide current

This file is the source of truth for agent usage. When the orchestrator
learns something about using agents (a model's strengths, a launch problem,
a budget change, a briefing technique) it updates this file in the same
step as any memory note, and the memory note points here instead of
duplicating it. Record the lesson in the table below.

## Lessons in brief

| What happened | Rule now |
|---|---|
| Containers run as host uid 1000/1001 with no passwd row; torch/Inductor `getpass` crashed (3 times) | Every torch-importing container gets `USER`/`LOGNAME`/`HOME`/`TORCHINDUCTOR_CACHE_DIR`/`TRITON_CACHE_DIR` set (in codex-preamble) |
| Codex plugin tasks all blocked (sandbox) | Plain: `scripts/agents/codex-launch.sh`; enhanced: Agent tool with a Codex model |
| `bwrap` launcher broke SSH, sudo and `/mnt` writes for every agent | Launcher selects credentials with `CLAUDE_SECURESTORAGE_CONFIG_DIR`, no namespace |
| Killed launcher, orphaned Codex kept editing | Stop with `codex-stop.sh` / `TaskStop` |
| Duplicate runs deferred to each other | One agent per task; resume notes |
| Literal 2% threshold kept a slower default | State the full decision rule |
| Lock-order deadlock | `sparks.lock`, then `gpu1.lock`, with timeouts |
| Stray 30 GB server | Teardown before releasing locks |
| A build downloading wheels held sparks.lock ~45 min, 5 agents queued | Build outside hardware locks; locks only around GPU/Spark use |
| 4-hour device-mode hang | Watchdogs and per-step timeouts |
| Disk full at 1.4 TB of builds | Clean build output at task end |
| `sudo python …` flagged | Sudo the real command directly |
| Restart lost agents' watchers | PR-body checkpoints before restarts; `pr-sync pull` on resume |
| Five hardware slots lost to harness setup mistakes (2026-10-11/12) | `hwlock.py --preflight` re-checks at enqueue and at grant |
| 8 Opus agents, 60% weekly in a day | Sol for bounded work; Claude orchestrates and judges |
| Shared Cargo target across two worktrees reused base metadata for the candidate (false-fresh build) | A/B builds use a separate `CARGO_TARGET_DIR` per arm |
| Orchestrator's `git commit -a` in its own worktree swept a sub-agent's uncommitted edits (the harness had placed the agent there) into a pushed commit | Orchestrator merges and commits in a dedicated throwaway worktree, stages files by name, and checks `git status` first; agent briefs name an explicit worktree outside `.claude/` |
| "Model at capacity" ended a run | Retry after a few minutes; resume with a note |
| ~2,700 hand-written driver scripts; most of one day's failures came from them (wrong container prefix, stale markers, dropped trace, missing --interleave) | The preamble's two commands; wip-cards preflights before queuing, `--resume` replaces retries, and agents ask for tooling instead of writing drivers |
| Codex subscription hit its weekly limit; 6 agents stopped at once | Read `usage_limit_reached`; relaunch on another model from STATUS.md |
