# CuteAFD Coordinator Entrypoint

Read `AGENTS.md` for standing engineering, ML, build, and cluster rules, and `peerswarm/PEERSWARM.md` for multi-coordinator workflow rules.

When the operator says **"activate peerswarm"**, run:

```bash
peerswarm/target/release/peerswarm activate --peer "${PEERSWARM_PEER:-claude}"
```

Delegate bulk implementation, test sweeps, benchmarks, and multi-file audits to native subagents inside task worktrees created via `peerswarm worktree create`. Never poll in a loop—use `peerswarm wait --peer "${PEERSWARM_PEER:-claude}"` when waiting on peers.
