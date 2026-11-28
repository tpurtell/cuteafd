"""Unit tests for the long-context calibration bank harness, privacy/secret scans,
quality acceptance gates, and bwrap sandbox isolation."""
from __future__ import annotations

import json
import os
from pathlib import Path
import pwd
import subprocess
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.corpus.long_context_bank import (
    sanitize_reproduction_script,
    scan_text_for_privacy_and_secrets,
    scan_text_for_secrets,
)
from scripts.corpus.long_context_harness import (
    WorkspaceSandbox,
    _extract_sse_reasoning_delta,
    build_cli_bwrap_argv,
    check_trace_quality,
    cut_at_message_boundary,
    ephemeral_cli_cred_dir,
)
from scripts.corpus.long_context_specs import (
    CASES,
    CASES_BY_ID,
    TRAINING_DATA_ROOT,
    V2_PANEL_SNAPSHOT,
)


class _FakeEstimator:
    def estimate_messages(self, messages, tools=None):
        total = sum(len(json.dumps(m)) // 4 for m in messages)
        return total, {"fake": total}


def test_specs_invariants_and_local_sources():
    ids = [c.id for c in CASES]
    assert len(ids) == len(set(ids)) == 16
    c16 = [c for c in CASES if c.length_class == "16k"]
    c32 = [c for c in CASES if c.length_class == "32k"]
    assert len(c16) == 10
    assert len(c32) == 6
    assert sum(1 for c in c16 if c.split == "calibration") == 8
    assert sum(1 for c in c16 if c.split == "heldout") == 2
    assert sum(1 for c in c32 if c.split == "calibration") == 4
    assert sum(1 for c in c32 if c.split == "heldout") == 2

    cal_groups = {c.group for c in CASES if c.split == "calibration"}
    hold_groups = {c.group for c in CASES if c.split == "heldout"}
    assert not (cal_groups & hold_groups)

    for c in CASES:
        if c.source_kind in ("openrouter_agentic", "wiki_qa", "oss_repo_qa"):
            assert c.provider_only, f"{c.id} missing provider_only"
            assert c.expected_provider_names, f"{c.id} missing expected_provider_names"
        if c.source_kind in ("wiki_qa", "oss_repo_qa"):
            assert len(c.extra.get("qa_turns", ())) >= 3, f"{c.id} must have >=3 QA turns"

    if TRAINING_DATA_ROOT.is_dir():
        for c in CASES:
            for rel in c.local_sources:
                assert (TRAINING_DATA_ROOT / rel).is_file(), f"Missing {rel}"
            if c.oss_repo_dir:
                repo = TRAINING_DATA_ROOT / c.oss_repo_dir
                assert repo.is_dir(), f"Missing {repo}"
                for rel in c.oss_include_globs:
                    assert (repo / rel).is_file(), f"Missing {repo / rel}"

    if V2_PANEL_SNAPSHOT.is_dir():
        v2_m = json.loads((V2_PANEL_SNAPSHOT / "manifest.json").read_text())
        v2_cal = {r["group"] for r in v2_m["splits"]["calibration"]["records"]}
        v2_hold = {r["group"] for r in v2_m["splits"]["heldout"]["records"]}
        assert not (cal_groups & v2_hold)
        assert not (hold_groups & v2_cal)


def test_sandbox_containment(tmp_path):
    sb = WorkspaceSandbox(tmp_path / "sb")
    assert "Wrote" in sb.execute_tool("write_file", {"path": "a/b.txt", "content": "hello\nworld\n"})
    assert "1: hello" in sb.execute_tool("read_file", {"path": "a/b.txt"})
    assert "escapes sandbox" in sb.execute_tool("read_file", {"path": "../outside.txt"})
    assert "absolute paths" in sb.execute_tool("write_file", {"path": "/tmp/evil.txt", "content": "x"})
    assert "rejected by sandbox policy" in sb.execute_tool("run_shell", {"command": "cat ~/.config/cuteafd/secrets/openrouter-key"})


def test_sse_reasoning_delta_no_duplication():
    delta_both = {
        "reasoning": "hello ",
        "reasoning_details": [{"type": "reasoning.text", "text": "hello "}],
    }
    assert _extract_sse_reasoning_delta(delta_both) == "hello "
    assert _extract_sse_reasoning_delta({"reasoning_details": [{"type": "reasoning.summary", "summary": "sum"}]}) == "sum"


def test_cut_at_message_boundary_never_orphans_tool_calls():
    est = _FakeEstimator()
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u" * 400},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": {"path": "a"}}}],
        },
        {"role": "tool", "tool_call_id": "c1", "name": "read_file", "content": "x" * 1200},
        {"role": "assistant", "content": "done " * 200},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c2", "type": "function", "function": {"name": "run_shell", "arguments": {"command": "ls"}}}],
        },
        {"role": "tool", "tool_call_id": "c2", "name": "run_shell", "content": "y" * 2000},
    ]
    trimmed, _, _ = cut_at_message_boundary(msgs, None, est, target_tokens=450, min_tokens=300, max_tokens=600)
    assert trimmed[-1]["role"] in ("tool", "assistant")
    if trimmed[-1]["role"] == "assistant":
        assert not trimmed[-1].get("tool_calls")


def test_reproduction_script_sanitizer_has_zero_privacy_hits():
    import py_compile
    import tempfile

    corpus_dir = REPO_ROOT / "scripts/corpus"
    for fname in ("long_context_specs.py", "long_context_harness.py", "long_context_bank.py"):
        raw = (corpus_dir / fname).read_text(encoding="utf-8")
        sanitized = sanitize_reproduction_script(fname, raw)
        hits = scan_text_for_privacy_and_secrets(sanitized)
        assert hits == [], f"Sanitized {fname} still has privacy hits: {hits}"
        # Confirm sanitized script is still syntactically valid Python
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=True) as tf:
            tf.write(sanitized)
            tf.flush()
            py_compile.compile(tf.name, doraise=True)


def test_privacy_and_secret_scanner():
    assert scan_text_for_privacy_and_secrets("clean text with code x = 1") == []
    # Secrets
    assert "secret:openrouter_key" in scan_text_for_secrets("key = sk-or-v1-abcdef1234567890abcdef")
    assert "secret:hf_token" in scan_text_for_secrets("tok = hf_abcdefghijklmnopqrstuvwxyz")
    assert "secret:github_pat" in scan_text_for_secrets("ghp_1234567890abcdefghij1234567890")
    assert "secret:aws_access_key" in scan_text_for_secrets("AKIAIOSFODNN7EXAMPLE")
    assert "secret:bearer_secret" in scan_text_for_secrets("Authorization: Bearer abcdef1234567890abcdef1234")
    assert "literal:custom" in scan_text_for_secrets("oops my_super_secret_val_99", [("custom", "my_super_secret_val_99")])
    # TJ-private hostnames, username, paths, handles, internals, emails
    import socket

    real_host = socket.gethostname()
    for bad, prefix in [
        ("running on raptor today", "privacy:private_hostname"),
        (f"host is {real_host} here", "privacy:private_hostname"),
        ("copied to maxq and sparknest", "privacy:private_hostname"),
        ("nodes ostrich dodo emu kiwi rhea moa kakapo cassowary", "privacy:private_hostname"),
        ("file at /home/tj/foo", "privacy:home_tj_path"),
        ("drwxrwxr-x 1 tj tj 120 Nov 28 work", "privacy:host_username_tj"),
        ("owner is tj in output", "privacy:host_username_tj"),
        ("cd ~/Developer/project", "privacy:dev_dir_path"),
        ("see Developer/cuteafd/README", "privacy:dev_dir_path"),
        ("repo wrldsuksgo2mars/panel", "privacy:private_handle"),
        ("github.com/tpurtell/x", "privacy:private_handle"),
        ("run hwlock and wip.sh on sparkinfer", "privacy:cuteafd_internal"),
        ("contact alice@example.com for details", "privacy:email_address"),
    ]:
        hits = scan_text_for_privacy_and_secrets(bad)
        assert any(h.startswith(prefix) for h in hits), f"Expected {prefix} for {bad!r}, got {hits}"


def test_quality_gate_rejects_degenerate_and_accepts_valid():
    spec_ag = CASES_BY_ID["lc-16k-cal-01"]
    # Degenerate trace: empty content + huge reasoning + 4 tool calls on empty workspace
    degen = {
        "token_count_estimate": 16500,
        "has_reasoning": True,
        "reasoning_chars": 30000,
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "build"},
            {"role": "assistant", "content": "", "reasoning_content": "x" * 30000},
        ],
    }
    defects = check_trace_quality(degen, spec_ag)
    assert any("degenerate" in d for d in defects)
    assert any("tool calls" in d for d in defects)
    assert any("failing-then-passing" in d for d in defects)

    # Valid agentic trace (fail->pass cycle)
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "p1"}]
    for i in range(10):
        msgs.append({
            "role": "assistant", "content": "", "reasoning_content": "plan " * 20,
            "tool_calls": [{"id": f"w{i}", "type": "function", "function": {"name": "write_file", "arguments": {"path": f"f{i}.cpp", "content": "int x;"}}}],
        })
        msgs.append({"role": "tool", "tool_call_id": f"w{i}", "name": "write_file", "content": "Wrote file"})
    for i, out in enumerate(["exit_code=1\nFAILED test_ring", "exit_code=0\nAll tests OK", "exit_code=0\nAll 20 tests OK"]):
        msgs.append({
            "role": "assistant", "content": "", "reasoning_content": "run tests " * 10,
            "tool_calls": [{"id": f"s{i}", "type": "function", "function": {"name": "run_shell", "arguments": {"command": "./test"}}}],
        })
        msgs.append({"role": "tool", "tool_call_id": f"s{i}", "name": "run_shell", "content": out})
    msgs.append({"role": "assistant", "content": "All tests pass.", "reasoning_content": "done " * 10})
    good_ag = {"token_count_estimate": 16200, "has_reasoning": True, "reasoning_chars": 1500, "messages": msgs}
    assert check_trace_quality(good_ag, spec_ag) == []

    # Valid agentic trace (first-try green code with >=4 real build/test runs with output)
    msgs_4pass = [{"role": "system", "content": "sys"}, {"role": "user", "content": "p1"}]
    for i in range(10):
        msgs_4pass.append({
            "role": "assistant", "content": "", "reasoning_content": "plan " * 20,
            "tool_calls": [{"id": f"w{i}", "type": "function", "function": {"name": "write_file", "arguments": {"path": f"f{i}.cpp", "content": "int x;"}}}],
        })
        msgs_4pass.append({"role": "tool", "tool_call_id": f"w{i}", "name": "write_file", "content": "Wrote file"})
    for i in range(4):
        msgs_4pass.append({
            "role": "assistant", "content": "", "reasoning_content": "run tests " * 10,
            "tool_calls": [{"id": f"s{i}", "type": "function", "function": {"name": "run_shell", "arguments": {"command": "./test"}}}],
        })
        msgs_4pass.append({"role": "tool", "tool_call_id": f"s{i}", "name": "run_shell", "content": f"exit_code=0\nAll {10 + i} tests OK"})
    msgs_4pass.append({"role": "assistant", "content": "All tests pass.", "reasoning_content": "done " * 10})
    good_4pass = {"token_count_estimate": 16200, "has_reasoning": True, "reasoning_chars": 1500, "messages": msgs_4pass}
    assert check_trace_quality(good_4pass, spec_ag) == []

    # Wiki QA gate rejects 1-turn QA, accepts 3-turn QA
    spec_wk = CASES_BY_ID["lc-16k-cal-06"]
    one_turn = {
        "token_count_estimate": 16000, "has_reasoning": True, "reasoning_chars": 1000,
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "doc + q1"},
            {"role": "assistant", "content": "a1", "reasoning_content": "r" * 300},
        ],
    }
    assert any(">=3" in d for d in check_trace_quality(one_turn, spec_wk))
    three_turn = {
        "token_count_estimate": 16000, "has_reasoning": True, "reasoning_chars": 1000,
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "doc + q1"},
            {"role": "assistant", "content": "a1", "reasoning_content": "r" * 300},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2", "reasoning_content": "r" * 300},
            {"role": "user", "content": "q3"},
            {"role": "assistant", "content": "a3", "reasoning_content": "r" * 300},
        ],
    }
    assert check_trace_quality(three_turn, spec_wk) == []


def test_bwrap_run_shell_isolation(tmp_path):
    """Verify OS-level bwrap --unshare-all blocks host secrets, /mnt/sparknest, and network."""
    sb = WorkspaceSandbox(tmp_path / "bwrap_sb")
    home = pwd.getpwuid(os.getuid()).pw_dir

    out_direct = sb.execute_tool("run_shell", {"command": "cat ~/.config/cuteafd/secrets/openrouter-key"})
    assert "rejected" in out_direct or "exit_code=1" in out_direct
    out_glob = sb.execute_tool("run_shell", {"command": f"cat {home}/.config/cute*/secrets/*"})
    assert "exit_code=1" in out_glob
    assert "No such file or directory" in out_glob

    out_nest = sb.execute_tool("run_shell", {"command": "ls /mnt/sparknest"})
    assert "exit_code=2" in out_nest or "exit_code=1" in out_nest
    assert "No such file or directory" in out_nest

    out_net = sb.execute_tool(
        "run_shell",
        {"command": "python3 -c \"import urllib.request; urllib.request.urlopen('https://example.com', timeout=5)\""},
    )
    assert "exit_code=1" in out_net
    assert "URLError" in out_net or "gaierror" in out_net or "OSError" in out_net

    out_tools = sb.execute_tool(
        "run_shell",
        {"command": "python3 -c 'print(2+2)' && node -e 'console.log(3+3)' && go version && rustc --version && g++ --version | head -n1"},
    )
    assert "exit_code=0" in out_tools
    assert "4" in out_tools and "6" in out_tools

    # 5. ls -la, id, and hostname must show neutral 'user' / 'sandbox', never host username or hostname
    import socket

    real_user = pwd.getpwuid(os.getuid()).pw_name
    real_host = socket.gethostname()
    sb.execute_tool("write_file", {"path": "sample.txt", "content": "ok\n"})
    out_id = sb.execute_tool("run_shell", {"command": "id && hostname && ls -la"})
    assert "exit_code=0" in out_id
    assert "(user)" in out_id
    assert "sandbox" in out_id
    assert real_user not in out_id
    assert real_host not in out_id


def test_cli_bwrap_ephemeral_cred_isolation(tmp_path):
    """Verify Codex & Claude bwrap wrappers never expose host /home/tj, real ~/.claude, real ~/.codex,
    ~/.config/cuteafd, ~/.ssh, HF token, host username/hostname, or /mnt/sparknest, and delete temp cred copies."""
    import socket

    sb = WorkspaceSandbox(tmp_path / "cli_sb")
    pw = pwd.getpwuid(os.getuid())
    home = pw.pw_dir
    real_user = pw.pw_name
    real_host = socket.gethostname()
    probe = (
        f"test ! -e {home} && "
        f"test ! -e {home}/.claude && "
        f"test ! -e {home}/.codex && "
        f"test ! -e {home}/.config/cuteafd && "
        f"test ! -e {home}/.ssh && "
        f"test ! -e {home}/.cache/huggingface && "
        f"test ! -e /mnt/sparknest && "
        "id && hostname && ls -la && echo EPHEMERAL_ISOLATED_OK"
    )
    for cli_kind in ("codex", "claude"):
        with ephemeral_cli_cred_dir(sb.root, cli_kind) as cred_dir:
            assert cred_dir.is_dir()
            names = {p.name for p in cred_dir.iterdir()}
            if cli_kind == "codex":
                assert names <= {"auth.json", "config.toml"}
            else:
                assert names <= {".credentials.json", ".claude.json"}
            cmd = build_cli_bwrap_argv(
                sandbox_root=sb.root, cred_dir=cred_dir, cli_kind=cli_kind, inner_cmd=["/bin/bash", "-c", probe]
            )
            proc = subprocess.run(cmd, env={}, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=15)
            assert proc.returncode == 0, f"{cli_kind} bwrap probe failed: {proc.stderr}"
            assert "EPHEMERAL_ISOLATED_OK" in proc.stdout
            assert "(user)" in proc.stdout
            assert real_user not in proc.stdout
            assert real_host not in proc.stdout

            ver_cmd = build_cli_bwrap_argv(
                sandbox_root=sb.root, cred_dir=cred_dir, cli_kind=cli_kind, inner_cmd=[cli_kind, "--version"]
            )
            vproc = subprocess.run(ver_cmd, env={}, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=15)
            assert vproc.returncode == 0, f"{cli_kind} --version inside bwrap failed: {vproc.stderr}"
        # After context exit, ephemeral credential directory must be completely deleted
        assert not cred_dir.exists()
