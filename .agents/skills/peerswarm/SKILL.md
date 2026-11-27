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
5. Keep all build artifacts, task worktrees, and intermediate files inside `peerswarm.state/`.
6. Use `peerswarm resource exec` for cluster GPU/Spark/build leases, `peerswarm worktree create` for task worktrees, and `peerswarm wait --peer "$PEERSWARM_PEER"` (add `--merges` if master) when waiting reactively.
