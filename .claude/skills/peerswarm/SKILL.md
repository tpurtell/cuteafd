---
name: peerswarm
description: Activate and coordinate with the CuteAFD multi-vendor PeerSwarm (Google Antigravity master, Anthropic Claude Code, OpenAI Codex) using local SQLite agent mail, PM items, merge queue, fair resource leases, and worktree/GC tooling.
---

# PeerSwarm (CuteAFD)

When the operator says **"activate peerswarm"** or asks to check/coordinate swarm work:

1. Identify your peer ID from `$PEERSWARM_PEER` (`agy`, `claude`, or `codex`).
2. Use `peerswarm` on `PATH` or `<repo>/peerswarm/target/release/peerswarm`.
3. Run:
   ```bash
   peerswarm activate --peer "$PEERSWARM_PEER"
   ```
4. Follow `peerswarm/PEERSWARM.md` for swarm coordination and `AGENTS.md` for CuteAFD's standing ML, engineering, build, and cluster rules.
5. **On restart / crash recovery**: immediately check `peerswarm job list --all --peer "$PEERSWARM_PEER"` and `peerswarm peer subagents --peer "$PEERSWARM_PEER"`, reattach to surviving durable systemd jobs, and resume or respawn any interrupted subagents from their task worktrees and briefs.
6. **Master auto-scheduling**: consult `peerswarm schedule --kind <kind> --tier <tier>` and `peerswarm budget` before assigning new work — honoring `peerswarm peer drain|pause`, keeping GPUs utilized, draining merges before rebase conflicts pile up, and moderating a vendor only when its last available account drops ≤ 20% (Claude `tpurtell`+`vimoin` failover is automatic).
7. **Durable jobs & containment**: keep all build caches, task worktrees, and artifacts inside `peerswarm.state/`; launch long gates/builds/goldens/benchmarks via `peerswarm job run` (local or `--host maxq`), acquire locks via `peerswarm resource exec`, manage lifecycle via `peerswarm restart|stop|account`, and wait reactively via `peerswarm wait --peer "$PEERSWARM_PEER"` (add `--merges` if master). The watchdog and read-only LAN dashboard (`http://raptor:7788`) run automatically whenever the swarm is launched.
