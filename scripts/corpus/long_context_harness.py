"""Sandboxed agentic loop, CLI trace drivers, and local long-document QA drivers.

Security, privacy, and provenance guarantees:
- OpenRouter API key is read directly from file in memory, never logged, printed,
  written to disk, or passed in argv/env of any child process.
- First-party OpenRouter routing is enforced via ``provider: {only: [...], allow_fallbacks: false}``
  and verified against ``response["provider"]``.
- Model-generated code runs ONLY inside a dedicated per-case sandbox directory under
  ``~/.cache/cuteafd/builds/long-context-bank/sandboxes/<case_id>`` executed via
  ``bwrap --unshare-all`` (no network, no host $HOME, no /mnt/sparknest).
- Codex and Claude CLI drivers run inside ``bwrap`` with ephemeral per-case credential
  directories (``CODEX_HOME`` / ``CLAUDE_CONFIG_DIR``) containing ONLY the auth file
  and minimal non-personal config — never the real ``~/.codex`` or ``~/.claude``/``~/.claude.json``.
  Host ``/home/tj`` is not mounted at all, and temp credentials are deleted on exit.
- Output messages use model-neutral HF/OpenAI chat format:
  - assistant messages store visible thinking in ``reasoning_content`` (never raw token IDs)
  - ``tool_calls[].function.arguments`` are stored as parsed JSON dicts so Hugging Face
    ``apply_chat_template`` works natively across GLM-5.3, Qwen3.8, Hy4, etc.
"""
from __future__ import annotations

from contextlib import contextmanager
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import pwd
import re
import resource
import shutil
import signal
import subprocess
import time
import urllib.error
import urllib.request
from typing import Any, Iterator

from scripts.corpus.long_context_specs import (
    CaseSpec,
    MODEL_MAX_OUTPUT_TOKENS,
    OPENROUTER_KEY_PATH,
    REFERENCE_TOKENIZERS,
    TRAINING_DATA_ROOT,
)


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MAX_TOOL_OUTPUT_CHARS = 4500
MAX_READ_FILE_CHARS = 7500


AGENTIC_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files and directories inside the sandbox workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative directory path inside workspace (default '.').",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a UTF-8 text file inside the sandbox workspace with line numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative file path."},
                    "start_line": {"type": "integer", "description": "1-based start line (optional)."},
                    "end_line": {"type": "integer", "description": "1-based end line inclusive (optional)."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create or overwrite a UTF-8 text file inside the sandbox workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative file path."},
                    "content": {"type": "string", "description": "Full file contents."},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace an exact substring in an existing workspace file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative file path."},
                    "target": {"type": "string", "description": "Exact existing text to replace."},
                    "replacement": {"type": "string", "description": "Replacement text."},
                },
                "required": ["path", "target", "replacement"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": "Run a shell command inside the sandbox workspace (e.g. compile, run tests, check files).",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Bash command to run in workspace root."},
                },
                "required": ["command"],
            },
        },
    },
]


class TokenEstimator:
    """Fast local multi-tokenizer length estimator (never persists token IDs)."""

    def __init__(self, tokenizer_paths: dict[str, str] | None = None):
        from tokenizers import Tokenizer

        paths = tokenizer_paths or REFERENCE_TOKENIZERS
        self._tokenizers: dict[str, Any] = {}
        for name, snap in paths.items():
            tok_file = Path(snap) / "tokenizer.json"
            if tok_file.exists():
                self._tokenizers[name] = Tokenizer.from_file(str(tok_file))
        if not self._tokenizers:
            raise RuntimeError("No local reference tokenizers found under /mnt/sparknest/hf-home/hub")

    def render_neutral_text(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> str:
        parts: list[str] = []
        if tools:
            parts.append(f"<tools>\n{json.dumps(tools, ensure_ascii=False, sort_keys=True)}\n</tools>")
        for msg in messages:
            role = msg.get("role", "")
            parts.append(f"<role:{role}>")
            if msg.get("reasoning_content"):
                parts.append(f"<think>\n{msg['reasoning_content']}\n</think>")
            if msg.get("content"):
                parts.append(str(msg["content"]))
            if msg.get("tool_calls"):
                parts.append(json.dumps(msg["tool_calls"], ensure_ascii=False, sort_keys=True))
            if role == "tool" and msg.get("name"):
                parts.append(f"[tool:{msg['name']}]")
        return "\n".join(parts)

    def estimate_text(self, text: str) -> dict[str, int]:
        return {
            name: len(tok.encode(text, add_special_tokens=False).ids)
            for name, tok in self._tokenizers.items()
        }

    def estimate_messages(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> tuple[int, dict[str, int]]:
        rendered = self.render_neutral_text(messages, tools)
        by_tok = self.estimate_text(rendered)
        overhead = 6 * len(messages)
        by_tok = {k: v + overhead for k, v in by_tok.items()}
        median = sorted(by_tok.values())[len(by_tok) // 2]
        return median, by_tok


class WorkspaceSandbox:
    """Strictly contained bwrap sandbox workspace for executing untrusted model tool calls."""

    def __init__(self, root: Path):
        self.root = root.resolve()
        self.work_dir = self.root / "work"
        self.home_dir = self.root / ".sandbox_home"
        self.tmp_dir = self.root / ".sandbox_tmp"
        if self.root.exists():
            shutil.rmtree(self.root)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.home_dir.mkdir(parents=True, exist_ok=True)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

    def _safe_path(self, rel: str) -> Path:
        if not isinstance(rel, str) or not rel.strip():
            raise ValueError("path must be a non-empty relative path")
        p = Path(rel)
        if p.is_absolute():
            raise ValueError(f"absolute paths are forbidden inside sandbox: {rel}")
        resolved = (self.work_dir / p).resolve()
        try:
            resolved.relative_to(self.work_dir)
        except ValueError as exc:
            raise ValueError(f"path escapes sandbox: {rel}") from exc
        return resolved

    def _ensure_neutral_etc(self) -> tuple[Path, Path, Path]:
        """Generate neutral passwd, group, and hostname files mapping uid/gid to 'user'."""
        etc_dir = self.root / ".sandbox_etc"
        etc_dir.mkdir(parents=True, exist_ok=True)
        uid, gid = os.getuid(), os.getgid()
        passwd = etc_dir / "passwd"
        group = etc_dir / "group"
        hostname = etc_dir / "hostname"
        passwd.write_text(
            f"root:x:0:0:root:/root:/bin/bash\n"
            f"nobody:x:65534:65534:nobody:/nonexistent:/usr/sbin/nologin\n"
            f"user:x:{uid}:{gid}:Sandbox User:/sandbox/.sandbox_home:/bin/bash\n",
            encoding="utf-8",
        )
        group.write_text(
            f"root:x:0:\n"
            f"nogroup:x:65534:\n"
            f"user:x:{gid}:\n",
            encoding="utf-8",
        )
        hostname.write_text("sandbox\n", encoding="utf-8")
        return passwd, group, hostname

    def _clean_env(self) -> dict[str, str]:
        safe_path = "/home/linuxbrew/.linuxbrew/bin:/usr/local/bin:/usr/bin:/bin"
        return {
            "PATH": safe_path,
            "HOME": "/sandbox/.sandbox_home",
            "USER": "user",
            "LOGNAME": "user",
            "TMPDIR": "/tmp",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "GOCACHE": "/sandbox/.sandbox_home/.gocache",
            "GOPATH": "/sandbox/.sandbox_home/.gopath",
            "GOPROXY": "off",
            "CARGO_NET_OFFLINE": "true",
            "CARGO_HOME": "/sandbox/.sandbox_home/.cargo",
            "CARGO_TARGET_DIR": "/sandbox/.cargo_target",
            "PYTHONDONTWRITEBYTECODE": "1",
            "NO_COLOR": "1",
        }

    def bwrap_argv(self, inner_cmd: list[str]) -> list[str]:
        """Build an unprivileged bubblewrap command with full namespace/network isolation
        and neutral /etc/passwd, /etc/group, /etc/hostname (never exposing host username/hostname)."""
        passwd, group, hostname = self._ensure_neutral_etc()
        argv = [
            "bwrap",
            "--unshare-all",
            "--hostname", "sandbox",
            "--die-with-parent",
            "--clearenv",
            "--ro-bind", "/usr", "/usr",
            "--ro-bind", "/bin", "/bin",
            "--ro-bind", "/lib", "/lib",
        ]
        if Path("/lib64").exists():
            argv.extend(["--ro-bind", "/lib64", "/lib64"])
        for etc_item in (
            "/etc/alternatives",
            "/etc/ld.so.cache",
            "/etc/ld.so.conf",
            "/etc/ld.so.conf.d",
            "/etc/nsswitch.conf",
            "/etc/localtime",
            "/etc/ssl",
            "/etc/ca-certificates",
        ):
            if Path(etc_item).exists():
                argv.extend(["--ro-bind", etc_item, etc_item])
        argv.extend(
            [
                "--ro-bind", str(passwd), "/etc/passwd",
                "--ro-bind", str(group), "/etc/group",
                "--ro-bind", str(hostname), "/etc/hostname",
            ]
        )
        if Path("/home/linuxbrew/.linuxbrew").is_dir():
            argv.extend(["--ro-bind", "/home/linuxbrew/.linuxbrew", "/home/linuxbrew/.linuxbrew"])
        argv.extend(
            [
                "--tmpfs", "/tmp",
                "--proc", "/proc",
                "--dev", "/dev",
                "--bind", str(self.root), "/sandbox",
                "--chdir", "/sandbox/work",
            ]
        )
        for k, v in self._clean_env().items():
            argv.extend(["--setenv", k, v])
        argv.extend(["--", *inner_cmd])
        return argv

    def _truncate(self, text: str, limit: int = MAX_TOOL_OUTPUT_CHARS) -> str:
        if len(text) <= limit:
            return text
        head = limit // 2
        tail = limit - head - 80
        return (
            text[:head]
            + f"\n... [truncated {len(text) - head - tail} chars] ...\n"
            + text[-tail:]
        )

    def execute_tool(self, name: str, args: dict[str, Any]) -> str:
        try:
            if name == "list_files":
                rel = args.get("path", ".") or "."
                target = self._safe_path(rel)
                if not target.exists():
                    return f"Error: directory does not exist: {rel}"
                entries = []
                for p in sorted(target.rglob("*")):
                    if any(part.startswith(".git") or part == "target" for part in p.relative_to(self.work_dir).parts):
                        continue
                    rel_p = p.relative_to(self.work_dir).as_posix()
                    if p.is_file():
                        entries.append(f"{rel_p} ({p.stat().st_size} bytes)")
                    elif p.is_dir():
                        entries.append(f"{rel_p}/")
                return "\n".join(entries[:250]) if entries else "(empty workspace)"

            if name == "read_file":
                target = self._safe_path(args["path"])
                if not target.is_file():
                    return f"Error: file not found: {args['path']}"
                lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
                start = max(1, int(args.get("start_line") or 1))
                end = int(args.get("end_line") or len(lines))
                end = min(len(lines), max(start, end))
                body = "\n".join(f"{i}: {lines[i - 1]}" for i in range(start, end + 1))
                return self._truncate(body, MAX_READ_FILE_CHARS)

            if name == "write_file":
                target = self._safe_path(args["path"])
                content = str(args["content"])
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")
                return f"Wrote {args['path']} ({len(content)} chars, {len(content.splitlines())} lines)."

            if name == "edit_file":
                target = self._safe_path(args["path"])
                if not target.is_file():
                    return f"Error: file not found: {args['path']}"
                old = target.read_text(encoding="utf-8")
                needle = str(args["target"])
                repl = str(args["replacement"])
                count = old.count(needle)
                if count == 0:
                    return f"Error: target text not found in {args['path']}"
                new = old.replace(needle, repl)
                target.write_text(new, encoding="utf-8")
                return f"Edited {args['path']} ({count} replacement(s), now {len(new.splitlines())} lines)."

            if name == "run_shell":
                cmd = str(args["command"])
                # Secondary defense-in-depth blocklist (primary isolation is bwrap --unshare-all)
                forbidden = (
                    "openrouter-key",
                    ".config/cuteafd",
                    "huggingface/token",
                    "pkill ",
                    "rm -rf /",
                    "/etc/shadow",
                )
                if any(tok in cmd for tok in forbidden):
                    return "Error: command rejected by sandbox policy."

                def _limits() -> None:
                    os.setsid()
                    resource.setrlimit(resource.RLIMIT_CPU, (45, 45))

                bwrap_cmd = self.bwrap_argv(["/bin/bash", "-c", cmd])
                proc = subprocess.Popen(
                    bwrap_cmd,
                    cwd=self.work_dir,
                    env={},
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    preexec_fn=_limits,
                )
                try:
                    out, err = proc.communicate(timeout=50)
                    rc = proc.returncode
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    out, err = proc.communicate()
                    return self._truncate(f"TIMEOUT after 50s\nSTDOUT:\n{out}\nSTDERR:\n{err}")
                combined = f"exit_code={rc}\n"
                if out:
                    combined += f"STDOUT:\n{out}\n"
                if err:
                    combined += f"STDERR:\n{err}\n"
                return self._truncate(combined.strip())

            return f"Error: unknown tool '{name}'"
        except Exception as exc:
            return f"Error executing {name}: {type(exc).__name__}: {exc}"


def _read_openrouter_key() -> str:
    if not OPENROUTER_KEY_PATH.is_file():
        raise FileNotFoundError(f"OpenRouter key file missing: {OPENROUTER_KEY_PATH}")
    key = OPENROUTER_KEY_PATH.read_text(encoding="utf-8").strip()
    if not key:
        raise ValueError("OpenRouter key file is empty")
    return key


def _extract_sse_reasoning_delta(delta: dict[str, Any]) -> str:
    """Extract reasoning delta text once per SSE chunk without duplicating reasoning + reasoning_details."""
    if isinstance(delta.get("reasoning"), str) and delta["reasoning"]:
        return delta["reasoning"]
    if isinstance(delta.get("reasoning_content"), str) and delta["reasoning_content"]:
        return delta["reasoning_content"]
    out: list[str] = []
    for rd in delta.get("reasoning_details") or []:
        if isinstance(rd, dict):
            if isinstance(rd.get("text"), str) and rd["text"]:
                out.append(rd["text"])
            elif isinstance(rd.get("summary"), str) and rd["summary"]:
                out.append(rd["summary"])
    return "".join(out)


def call_openrouter(
    *,
    model: str,
    messages: list[dict[str, Any]],
    provider_only: tuple[str, ...],
    expected_provider_names: tuple[str, ...],
    tools: list[dict[str, Any]] | None = None,
    reasoning_effort: str = "medium",
    max_tokens: int | None = None,
    timeout_s: int = 120,
) -> dict[str, Any]:
    """Send one OpenRouter chat completion request using SSE streaming, strictly pinned to first-party.

    Always preserves full model reasoning (never sets reasoning.enabled=False) and uses
    each model's documented output ceiling (at least 32K-64K) so thinking is never cut short.
    """
    api_key = _read_openrouter_key()
    effective_max_tokens = max_tokens or MODEL_MAX_OUTPUT_TOKENS.get(model, 64000)
    wire_messages: list[dict[str, Any]] = []
    for m in messages:
        wm: dict[str, Any] = {"role": m["role"]}
        if m.get("content") is not None:
            wm["content"] = m["content"]
        if m.get("reasoning_content") and m["role"] == "assistant":
            wm["reasoning"] = m["reasoning_content"]
        if m.get("tool_call_id"):
            wm["tool_call_id"] = m["tool_call_id"]
        if m.get("name"):
            wm["name"] = m["name"]
        if m.get("tool_calls"):
            wire_tcs = []
            for tc in m["tool_calls"]:
                fn = tc["function"]
                raw_args = fn.get("arguments", {})
                arg_str = raw_args if isinstance(raw_args, str) else json.dumps(raw_args, ensure_ascii=False)
                wire_tcs.append(
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {"name": fn["name"], "arguments": arg_str},
                    }
                )
            wm["tool_calls"] = wire_tcs
        wire_messages.append(wm)

    payload: dict[str, Any] = {
        "model": model,
        "messages": wire_messages,
        "max_tokens": effective_max_tokens,
        "temperature": 0.4,
        "stream": True,
        "provider": {
            "only": list(provider_only),
            "allow_fallbacks": False,
        },
        "reasoning": {"effort": reasoning_effort},
    }
    if tools:
        payload["tools"] = tools

    data = json.dumps(payload).encode("utf-8")
    last_err: Exception | None = None
    for attempt in range(4):
        req = urllib.request.Request(
            OPENROUTER_URL,
            data=data,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/tpurtell/cuteafd",
                "X-Title": "cuteafd-long-context-bank",
            },
        )
        try:
            provider_seen = ""
            finish_reason = None
            content_parts: list[str] = []
            reasoning_parts: list[str] = []
            tc_acc: dict[int, dict[str, Any]] = {}

            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                for raw_line in resp:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line or line.startswith(":"):
                        continue
                    if not line.startswith("data: "):
                        continue
                    chunk_str = line[6:].strip()
                    if chunk_str == "[DONE]":
                        break
                    obj = json.loads(chunk_str)
                    if "error" in obj:
                        raise RuntimeError(f"OpenRouter SSE error: {obj['error']}")
                    if obj.get("provider"):
                        provider_seen = str(obj["provider"])
                    choices = obj.get("choices") or []
                    if not choices:
                        continue
                    ch = choices[0]
                    if ch.get("finish_reason"):
                        finish_reason = ch["finish_reason"]
                    delta = ch.get("delta") or {}
                    if isinstance(delta.get("content"), str) and delta["content"]:
                        content_parts.append(delta["content"])
                    reas_delta = _extract_sse_reasoning_delta(delta)
                    if reas_delta:
                        reasoning_parts.append(reas_delta)
                    for tc_delta in delta.get("tool_calls") or []:
                        idx = int(tc_delta.get("index", len(tc_acc)))
                        entry = tc_acc.setdefault(
                            idx,
                            {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
                        )
                        if tc_delta.get("id"):
                            entry["id"] = str(tc_delta["id"])
                        fn_d = tc_delta.get("function") or {}
                        if fn_d.get("name"):
                            entry["function"]["name"] += str(fn_d["name"])
                        if fn_d.get("arguments"):
                            entry["function"]["arguments"] += str(fn_d["arguments"])

            if expected_provider_names and provider_seen not in expected_provider_names:
                raise RuntimeError(
                    f"First-party provider violation for {model}: got {provider_seen!r}, expected one of {expected_provider_names!r}"
                )
            msg_out: dict[str, Any] = {
                "role": "assistant",
                "content": "".join(content_parts),
            }
            reas_str = "".join(reasoning_parts).strip()
            if reas_str:
                msg_out["reasoning"] = reas_str
            if tc_acc:
                msg_out["tool_calls"] = [tc_acc[i] for i in sorted(tc_acc)]
            return {
                "provider": provider_seen,
                "choices": [{"finish_reason": finish_reason, "message": msg_out}],
            }
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, RuntimeError) as exc:
            err_detail = str(exc)
            if isinstance(exc, urllib.error.HTTPError):
                try:
                    err_detail += " " + exc.read().decode("utf-8", errors="replace")[:400]
                except Exception:
                    pass
                if exc.code in (400, 401, 403, 404):
                    raise RuntimeError(f"OpenRouter permanent error ({model}): {err_detail}") from exc
            last_err = RuntimeError(f"OpenRouter transient error ({model}, attempt {attempt + 1}): {err_detail}")
            time.sleep(min(25, 3 * (2**attempt)))
    assert last_err is not None
    raise last_err


def _extract_reasoning(msg: dict[str, Any]) -> str:
    if isinstance(msg.get("reasoning"), str) and msg["reasoning"].strip():
        return msg["reasoning"].strip()
    if isinstance(msg.get("reasoning_content"), str) and msg["reasoning_content"].strip():
        return msg["reasoning_content"].strip()
    details = msg.get("reasoning_details") or []
    chunks = []
    for item in details:
        if not isinstance(item, dict):
            continue
        if isinstance(item.get("text"), str) and item["text"].strip():
            chunks.append(item["text"].strip())
        elif isinstance(item.get("summary"), str) and item["summary"].strip():
            chunks.append(item["summary"].strip())
    return "\n\n".join(chunks).strip()


def _parse_tool_args(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
            return {"value": parsed}
        except json.JSONDecodeError:
            return {"raw_arguments": raw}
    return {}


def normalize_assistant_message(raw_msg: dict[str, Any], call_counter: list[int]) -> dict[str, Any]:
    out: dict[str, Any] = {"role": "assistant", "content": raw_msg.get("content") or ""}
    reasoning = _extract_reasoning(raw_msg)
    if reasoning:
        out["reasoning_content"] = reasoning
    raw_tcs = raw_msg.get("tool_calls") or []
    if raw_tcs:
        norm_tcs = []
        for tc in raw_tcs:
            call_counter[0] += 1
            fn = tc.get("function") or {}
            tc_id = tc.get("id") or f"call_{call_counter[0]}"
            norm_tcs.append(
                {
                    "id": str(tc_id),
                    "type": "function",
                    "function": {
                        "name": str(fn.get("name") or "unknown"),
                        "arguments": _parse_tool_args(fn.get("arguments")),
                    },
                }
            )
        out["tool_calls"] = norm_tcs
    return out


_SHELL_TOOL_NAMES = {"run_shell", "shell_command", "Bash"}
_WRITE_TOOL_NAMES = {"write_file", "edit_file", "Write", "Edit"}


def analyze_agentic_signals(messages: list[dict[str, Any]]) -> dict[str, Any]:
    """Extract structural agentic quality signals from a message list."""
    tool_calls = 0
    shell_calls = 0
    shell_with_output = 0
    write_edit_calls = 0
    saw_fail = False
    fail_then_pass = False
    empty_no_tool_turns = 0
    empty_phase_ends = 0
    user_turns = 0
    assistant_turns = 0

    tool_name_by_id: dict[str, str] = {}
    for idx, m in enumerate(messages):
        role = m.get("role")
        if role == "user":
            user_turns += 1
        elif role == "assistant":
            assistant_turns += 1
            tcs = m.get("tool_calls") or []
            content = (m.get("content") or "").strip()
            if not tcs and not content:
                empty_no_tool_turns += 1
            # Phase end: no tool calls AND (last msg or next msg is user)
            is_phase_end = (not tcs) and (idx + 1 == len(messages) or messages[idx + 1].get("role") == "user")
            if is_phase_end and not content:
                empty_phase_ends += 1
            for tc in tcs:
                tool_calls += 1
                fn = (tc.get("function") or {}).get("name", "")
                tool_name_by_id[tc.get("id", "")] = fn
                if fn in _SHELL_TOOL_NAMES:
                    shell_calls += 1
                    cmd = str((tc.get("function") or {}).get("arguments", {}).get("command", ""))
                    if any(tok in cmd for tok in (">", "cat <<", "tee ", "python3 -c", "mkdir ")):
                        write_edit_calls += 1
                if fn in _WRITE_TOOL_NAMES:
                    write_edit_calls += 1
        elif role == "tool":
            tname = m.get("name") or tool_name_by_id.get(m.get("tool_call_id", ""), "")
            out = str(m.get("content") or "")
            if tname in _SHELL_TOOL_NAMES:
                if len(out.strip()) > 12:
                    shell_with_output += 1
                has_exit_zero = re.match(r"^\s*exit_code=0\b", out) is not None
                has_exit_nonzero = re.match(r"^\s*exit_code=[1-9]\d*\b", out) is not None
                if has_exit_nonzero:
                    is_fail = True
                    is_pass = False
                elif has_exit_zero:
                    is_fail = ("FAIL\t" in out) or ("FAILED (" in out)
                    is_pass = not is_fail
                else:
                    is_fail = (
                        "FAILED" in out
                        or "FAIL\t" in out
                        or "AssertionError" in out
                        or "error:" in out.lower()
                    )
                    is_pass = (
                        "test result: ok" in out
                        or "PASS" in out
                        or "\nOK" in out
                        or out.strip().endswith("OK")
                        or "# pass " in out
                    ) and not is_fail
                if is_fail:
                    saw_fail = True
                elif is_pass and saw_fail:
                    fail_then_pass = True

    return {
        "user_turns": user_turns,
        "assistant_turns": assistant_turns,
        "tool_calls": tool_calls,
        "shell_calls": shell_calls,
        "shell_with_output": shell_with_output,
        "write_edit_calls": write_edit_calls,
        "saw_fail": saw_fail,
        "fail_then_pass": fail_then_pass,
        "empty_no_tool_turns": empty_no_tool_turns,
        "empty_phase_ends": empty_phase_ends,
    }


def check_trace_quality(record: dict[str, Any], spec: CaseSpec) -> list[str]:
    """Acceptance gate for a generated record. Returns [] on pass, else list of defects."""
    defects: list[str] = []
    est = int(record.get("token_count_estimate", 0))
    if est < spec.min_tokens or est > spec.max_tokens:
        defects.append(f"token_count_estimate={est} outside [{spec.min_tokens}, {spec.max_tokens}]")

    msgs = record.get("messages") or []
    sig = analyze_agentic_signals(msgs)

    if sig["empty_no_tool_turns"] > 0:
        defects.append(f"has {sig['empty_no_tool_turns']} degenerate assistant turn(s) with empty content and no tool_calls")
    if sig["empty_phase_ends"] > 0:
        defects.append(f"has {sig['empty_phase_ends']} phase-ending assistant turn(s) with empty content")

    if spec.expect_reasoning:
        if not record.get("has_reasoning") or int(record.get("reasoning_chars", 0)) < 200:
            defects.append(f"expected visible reasoning, got has_reasoning={record.get('has_reasoning')} chars={record.get('reasoning_chars')}")

    if spec.source_kind in ("openrouter_agentic", "codex_cli", "claude_cli"):
        if spec.source_kind == "codex_cli":
            min_tools = 12 if spec.length_class == "32k" else 8
        else:
            min_tools = 15 if spec.length_class == "32k" else 12
        if sig["tool_calls"] < min_tools:
            defects.append(f"too few tool calls: {sig['tool_calls']} < {min_tools}")
        if sig["shell_with_output"] < 3:
            defects.append(f"too few real shell/test runs with output: {sig['shell_with_output']} < 3")
        if sig["write_edit_calls"] < 2:
            defects.append(f"too few code write/edit calls: {sig['write_edit_calls']} < 2")
        if not (sig["fail_then_pass"] or sig["shell_with_output"] >= 4):
            defects.append(
                f"missing failing-then-passing build/test cycle and fewer than 4 real shell/test runs "
                f"(saw_fail={sig['saw_fail']}, shell_with_output={sig['shell_with_output']})"
            )
    else:
        # Multi-turn long-document / OSS QA
        if sig["user_turns"] < 3 or sig["assistant_turns"] < 3:
            defects.append(f"multi-turn QA requires >=3 user and >=3 assistant turns (got {sig['user_turns']}/{sig['assistant_turns']})")

    return defects


def format_trace_summary(record: dict[str, Any]) -> str:
    """Format a compact per-message summary table (role, content/reasoning chars, tool names)."""
    lines = [
        f"Trace {record['id']} ({record['length_class']}, {record['split']}, {record['generating_model']}): "
        f"{record['token_count_estimate']} est tok | {record['message_count']} msgs | "
        f"{record['tool_call_count']} tool calls | reasoning={record['reasoning_chars']} chars"
    ]
    for i, m in enumerate(record.get("messages") or []):
        role = m.get("role", "")
        c_len = len(m.get("content") or "")
        r_len = len(m.get("reasoning_content") or "")
        tcs = [tc.get("function", {}).get("name", "?") for tc in (m.get("tool_calls") or [])]
        t_info = ",".join(tcs) if tcs else (m.get("name") or "-")
        lines.append(f"  [{i:02d}] {role:9s} content={c_len:6d}  reasoning={r_len:5d}  tools={t_info}")
    return "\n".join(lines)


def cut_at_message_boundary(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    estimator: TokenEstimator,
    *,
    target_tokens: int,
    min_tokens: int,
    max_tokens: int,
    min_user_turns: int = 1,
    min_tool_calls: int = 0,
    require_fail_then_pass: bool = False,
    case_id: str = "",
) -> tuple[list[dict[str, Any]], int, dict[str, int]]:
    """Cut a trace cleanly at a valid message boundary near target_tokens.

    Valid cut boundaries:
    - After an assistant message with NO pending tool_calls (complete turn), OR
    - After the final `tool` result message answering the preceding assistant's `tool_calls`.
    Prefers cuts inside [min_tokens, max_tokens] satisfying structural quality floors.
    If the final trace overshoots max_tokens, logs every candidate boundary's cumulative
    token count so any single-message jump is immediately visible. Never truncates inside a message.
    """
    valid_ends: list[int] = []
    i = 0
    while i < len(messages):
        m = messages[i]
        if m["role"] == "assistant":
            if m.get("tool_calls"):
                j = i + 1
                while j < len(messages) and messages[j]["role"] == "tool":
                    j += 1
                if j > i + 1:
                    valid_ends.append(j)
                i = j
                continue
            valid_ends.append(i + 1)
        i += 1

    if not valid_ends:
        valid_ends = [len(messages)]

    best_cut = valid_ends[-1]
    best_score = float("inf")
    best_med = 0
    best_by_tok: dict[str, int] = {}
    boundary_log: list[str] = []
    prev_med = 0

    for end in valid_ends:
        prefix = messages[:end]
        med, by_tok = estimator.estimate_messages(prefix, tools)
        sig = analyze_agentic_signals(prefix)
        in_band = min_tokens <= med <= max_tokens
        meets_struct = (
            sig["user_turns"] >= min_user_turns
            and sig["tool_calls"] >= min_tool_calls
            and (
                not require_fail_then_pass
                or sig["fail_then_pass"]
                or sig["shell_with_output"] >= 4
            )
        )
        boundary_log.append(
            f"end={end:02d}({prefix[-1]['role']:9s}) tok={med:6d} (+{med - prev_med:5d}) "
            f"u={sig['user_turns']} tc={sig['tool_calls']:2d} f2p={int(sig['fail_then_pass'])} "
            f"band={int(in_band)} struct={int(meets_struct)}"
        )
        prev_med = med
        end_penalty = 0 if prefix[-1]["role"] == "assistant" else 250
        # Strongly prioritize in-band + meets_struct; if an oversized next reasoning turn
        # would jump past max_tokens, an earlier in-band boundary wins cleanly.
        score = (
            abs(med - target_tokens)
            + end_penalty
            + (0 if in_band else 100_000)
            + (0 if meets_struct else 500_000)
        )
        if score < best_score:
            best_score = score
            best_cut = end
            best_med = med
            best_by_tok = by_tok

    if prev_med > max_tokens or not (min_tokens <= best_med <= max_tokens):
        tag = f" {case_id}" if case_id else ""
        print(
            f"  [BOUNDARIES{tag}] picked end={best_cut} ({best_med} tok, target={target_tokens}, "
            f"band=[{min_tokens},{max_tokens}]):\n    " + "\n    ".join(boundary_log),
            flush=True,
        )

    return messages[:best_cut], best_med, best_by_tok


def run_openrouter_agentic_case(
    spec: CaseSpec, sandbox_root: Path, estimator: TokenEstimator
) -> dict[str, Any]:
    """Run a real multi-phase bwrap-sandboxed agentic coding loop via OpenRouter first-party."""
    sandbox = WorkspaceSandbox(sandbox_root / spec.id)
    messages: list[dict[str, Any]] = [{"role": "system", "content": spec.system_prompt}]
    call_counter = [0]
    providers_seen: set[str] = set()
    api_calls = 0
    max_api_calls = 56 if spec.length_class == "32k" else 36
    min_tools = 15 if spec.length_class == "32k" else 12
    grow_threshold = spec.target_tokens

    def _should_stop_growing() -> bool:
        cur_med, _ = estimator.estimate_messages(messages, AGENTIC_TOOLS)
        sig = analyze_agentic_signals(messages)
        has_test_evidence = sig["fail_then_pass"] or sig["shell_with_output"] >= 4
        if cur_med >= spec.target_tokens and sig["tool_calls"] >= min_tools and has_test_evidence:
            return True
        # Hard early stop once past max_tokens so we never waste minutes generating far past target
        if cur_med >= spec.max_tokens:
            return True
        return False

    def _run_turn_loop(max_steps: int) -> None:
        nonlocal api_calls
        effort = spec.reasoning_effort
        steps = 0
        while steps < max_steps and api_calls < max_api_calls:
            if _should_stop_growing():
                break
            steps += 1
            api_calls += 1
            t_step = time.monotonic()
            resp = call_openrouter(
                model=spec.model,
                messages=messages,
                provider_only=spec.provider_only,
                expected_provider_names=spec.expected_provider_names,
                tools=AGENTIC_TOOLS,
                reasoning_effort=effort,
            )
            providers_seen.add(str(resp.get("provider", "")))
            choice = resp["choices"][0]
            finish_reason = choice.get("finish_reason")
            raw_msg = choice["message"]
            assistant_msg = normalize_assistant_message(raw_msg, call_counter)
            tcs = assistant_msg.get("tool_calls") or []
            content = (assistant_msg.get("content") or "").strip()
            cur_after, _ = estimator.estimate_messages(messages + [assistant_msg], AGENTIC_TOOLS)
            print(
                f"  [STEP] {spec.id} call={api_calls} fin={finish_reason} "
                f"tcs={len(tcs)} reas={len(assistant_msg.get('reasoning_content') or '')} "
                f"cont={len(content)} cum_tok={cur_after} ({time.monotonic() - t_step:.1f}s)",
                flush=True,
            )

            # On a length cut or empty no-tool turn: preserve any partial reasoning/content
            # and send a continue turn (never disable reasoning!).
            if finish_reason == "length" or (not tcs and not content):
                if assistant_msg.get("reasoning_content") or content:
                    if not content:
                        assistant_msg["content"] = "[Truncated at output limit; continuing in next turn.]"
                    messages.append(assistant_msg)
                if _should_stop_growing():
                    break
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Continue from where you left off: invoke the workspace file/shell tools "
                            "to implement the code and run tests."
                        ),
                    }
                )
                continue

            messages.append(assistant_msg)

            if not tcs:
                break

            for tc in tcs:
                fn = tc["function"]
                result = sandbox.execute_tool(fn["name"], fn["arguments"])
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "name": fn["name"],
                        "content": result,
                    }
                )
            if _should_stop_growing():
                break

    for phase_prompt in spec.phase_prompts:
        if _should_stop_growing():
            break
        messages.append({"role": "user", "content": phase_prompt})
        _run_turn_loop(max_steps=16 if spec.length_class == "32k" else 12)

    # Ensure we hit both token target and failing->passing test cycle
    extra_followups = (
        "Add a strict TDD regression test targeting 3 subtle boundary/malformed-input bugs "
        "(write the tests FIRST and run the test command to confirm they fail, then fix the "
        "implementation and re-run the test command to confirm all tests pass).",
        "Read the earliest source files you wrote in Phase 1, add a deterministic property/fuzz "
        "test suite exercising every public API, run the tests, fix any issues, and summarize.",
        "Add structured diagnostics/statistics output and an end-to-end integration test covering "
        "the full workflow, run the tests to verify zero warnings and 100% pass, and report.",
    )
    f_idx = 0
    while api_calls < max_api_calls and f_idx < len(extra_followups):
        if _should_stop_growing():
            break
        messages.append({"role": "user", "content": extra_followups[f_idx]})
        f_idx += 1
        _run_turn_loop(max_steps=10)

    trimmed, est_tokens, by_tok = cut_at_message_boundary(
        messages,
        AGENTIC_TOOLS,
        estimator,
        target_tokens=spec.target_tokens,
        min_tokens=spec.min_tokens,
        max_tokens=spec.max_tokens,
        min_user_turns=1,
        min_tool_calls=min_tools,
        require_fail_then_pass=True,
        case_id=spec.id,
    )
    return _package_record(
        spec=spec,
        messages=trimmed,
        tools=AGENTIC_TOOLS,
        est_tokens=est_tokens,
        by_tok=by_tok,
        provenance={
            "source_kind": spec.source_kind,
            "generating_model": spec.model,
            "providers_used": sorted(providers_seen),
            "generated_at_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "api_calls": api_calls,
            "raw_message_count": len(messages),
            "trimmed_message_count": len(trimmed),
            "sandbox_tool_calls": call_counter[0],
        },
    )


@contextmanager
def ephemeral_cli_cred_dir(
    sandbox_root: Path, cli_kind: str, host_home: Path | None = None
) -> Iterator[Path]:
    """Create a fresh per-case credential directory with ONLY the auth file and minimal config.

    Never binds or exposes real ~/.codex or ~/.claude history, memory, projects,
    CLAUDE.md, AGENTS.md, plugins, or settings. Always deletes the temp dir on exit.
    """
    real_home = host_home or Path(pwd.getpwuid(os.getuid()).pw_dir)
    cred_dir = (sandbox_root / f".ephemeral_cred_{cli_kind}").resolve()
    if cred_dir.exists():
        shutil.rmtree(cred_dir)
    cred_dir.mkdir(parents=True, mode=0o700)
    try:
        if cli_kind == "codex":
            src_auth = real_home / ".codex/auth.json"
            if src_auth.is_file():
                shutil.copy2(src_auth, cred_dir / "auth.json")
                os.chmod(cred_dir / "auth.json", 0o600)
            (cred_dir / "config.toml").write_text(
                'model = "gpt-6.2-sol"\nmodel_reasoning_effort = "medium"\n',
                encoding="utf-8",
            )
        elif cli_kind == "claude":
            src_cred = real_home / ".claude/.credentials.json"
            if src_cred.is_file():
                shutil.copy2(src_cred, cred_dir / ".credentials.json")
                os.chmod(cred_dir / ".credentials.json", 0o600)
            (cred_dir / ".claude.json").write_text(
                json.dumps({"hasCompletedOnboarding": True}) + "\n",
                encoding="utf-8",
            )
        else:
            raise ValueError(f"Unknown cli_kind: {cli_kind}")
        yield cred_dir
    finally:
        shutil.rmtree(cred_dir, ignore_errors=True)


def build_cli_bwrap_argv(
    *,
    sandbox_root: Path,
    cred_dir: Path,
    cli_kind: str,
    inner_cmd: list[str],
    host_home: Path | None = None,
) -> list[str]:
    """Build a bwrap command for Codex/Claude CLI: network allowed for its API,
    strictly isolated filesystem with ONLY its ephemeral auth dir and the sandbox workspace.
    Host /home/tj, /etc/passwd, and /etc/group are NOT mounted; synthetic passwd/group map
    the uid/gid to 'user' and hostname to 'sandbox'.
    """
    real_home = host_home or Path(pwd.getpwuid(os.getuid()).pw_dir)
    etc_dir = sandbox_root / ".sandbox_etc"
    etc_dir.mkdir(parents=True, exist_ok=True)
    uid, gid = os.getuid(), os.getgid()
    passwd = etc_dir / "passwd"
    group = etc_dir / "group"
    hostname = etc_dir / "hostname"
    hosts = etc_dir / "hosts"
    passwd.write_text(
        f"root:x:0:0:root:/root:/bin/bash\n"
        f"nobody:x:65534:65534:nobody:/nonexistent:/usr/sbin/nologin\n"
        f"user:x:{uid}:{gid}:Sandbox User:/sandbox_home:/bin/bash\n",
        encoding="utf-8",
    )
    group.write_text(
        f"root:x:0:\n"
        f"nogroup:x:65534:\n"
        f"user:x:{gid}:\n",
        encoding="utf-8",
    )
    hostname.write_text("sandbox\n", encoding="utf-8")
    hosts.write_text("127.0.0.1 localhost sandbox\n::1 localhost ip6-localhost ip6-loopback\n", encoding="utf-8")

    argv = [
        "bwrap",
        "--unshare-user",
        "--unshare-pid",
        "--unshare-ipc",
        "--unshare-uts",
        "--hostname", "sandbox",
        "--die-with-parent",
        "--clearenv",
        "--ro-bind", "/usr", "/usr",
        "--ro-bind", "/bin", "/bin",
        "--ro-bind", "/lib", "/lib",
    ]
    if Path("/lib64").exists():
        argv.extend(["--ro-bind", "/lib64", "/lib64"])
    for etc_item in (
        "/etc/alternatives",
        "/etc/ld.so.cache",
        "/etc/ld.so.conf",
        "/etc/ld.so.conf.d",
        "/etc/nsswitch.conf",
        "/etc/localtime",
        "/etc/ssl",
        "/etc/ca-certificates",
        "/etc/resolv.conf",
        "/etc/protocols",
        "/etc/services",
    ):
        if Path(etc_item).exists():
            argv.extend(["--ro-bind", etc_item, etc_item])
    argv.extend(
        [
            "--ro-bind", str(passwd), "/etc/passwd",
            "--ro-bind", str(group), "/etc/group",
            "--ro-bind", str(hostname), "/etc/hostname",
            "--ro-bind", str(hosts), "/etc/hosts",
        ]
    )
    if Path("/run/systemd/resolve").is_dir():
        argv.extend(["--ro-bind", "/run/systemd/resolve", "/run/systemd/resolve"])
    if Path("/home/linuxbrew/.linuxbrew").is_dir():
        argv.extend(["--ro-bind", "/home/linuxbrew/.linuxbrew", "/home/linuxbrew/.linuxbrew"])
    argv.extend(
        [
            "--tmpfs", "/tmp",
            "--proc", "/proc",
            "--dev", "/dev",
            "--tmpfs", "/sandbox_home",
            "--bind", str(sandbox_root), "/sandbox",
            "--chdir", "/sandbox/work",
        ]
    )

    cli_env: dict[str, str] = {
        "HOME": "/sandbox_home",
        "USER": "user",
        "LOGNAME": "user",
        "TMPDIR": "/tmp",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "GOCACHE": "/sandbox/.sandbox_home/.gocache",
        "GOPATH": "/sandbox/.sandbox_home/.gopath",
        "GOPROXY": "off",
        "CARGO_NET_OFFLINE": "true",
        "CARGO_TARGET_DIR": "/sandbox/.cargo_target",
        "PYTHONDONTWRITEBYTECODE": "1",
        "NO_COLOR": "1",
    }

    if cli_kind == "codex":
        argv.extend(["--bind", str(cred_dir), "/sandbox_cred/codex"])
        cli_env["CODEX_HOME"] = "/sandbox_cred/codex"
        cli_env["PATH"] = "/home/linuxbrew/.linuxbrew/bin:/usr/local/bin:/usr/bin:/bin"
    elif cli_kind == "claude":
        nvm_dir = real_home / ".nvm"
        if nvm_dir.is_dir():
            argv.extend(["--ro-bind", str(nvm_dir), "/opt/nvm"])
        argv.extend(["--bind", str(cred_dir), "/sandbox_cred/claude"])
        cli_env["CLAUDE_CONFIG_DIR"] = "/sandbox_cred/claude"
        cands = sorted((nvm_dir / "versions/node").glob("*/bin/claude")) if nvm_dir.is_dir() else []
        node_ver = cands[-1].parent.parent.name if cands else "v24.18.0"
        cli_env["PATH"] = f"/opt/nvm/versions/node/{node_ver}/bin:/home/linuxbrew/.linuxbrew/bin:/usr/local/bin:/usr/bin:/bin"
    else:
        raise ValueError(f"Unknown cli_kind: {cli_kind}")

    for k, v in cli_env.items():
        argv.extend(["--setenv", k, v])
    argv.extend(["--", *inner_cmd])
    return argv


def run_codex_cli_case(
    spec: CaseSpec, sandbox_root: Path, estimator: TokenEstimator
) -> dict[str, Any]:
    """Drive real `codex exec` + `codex exec resume` inside bwrap with ephemeral CODEX_HOME."""
    sandbox = WorkspaceSandbox(sandbox_root / spec.id)
    work = sandbox.work_dir
    subprocess.run(["git", "init", "-q"], cwd=work, check=True)

    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "You are Codex, an autonomous coding agent operating in a sandboxed workspace "
                "with shell execution tools."
            ),
        }
    ]
    codex_tools = [
        {
            "type": "function",
            "function": {
                "name": "shell_command",
                "description": "Execute a shell command in the workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            },
        }
    ]
    thread_id: str | None = None
    call_idx = 0
    raw_events_count = 0
    min_tools = 12 if spec.length_class == "32k" else 8

    all_prompts = list(spec.phase_prompts) + [
        "Follow TDD: FIRST write failing unit tests in `tests/test_edge_cases.py` for queue overflow, "
        "link flapping, and packet reordering, run `python3 -m unittest discover -s tests -v` to confirm "
        "they fail, THEN implement the fixes and re-run the test suite until all tests pass.",
        "Add a performance/scalability stress test running 10 concurrent flows over a 6-node mesh "
        "topology, verify all invariants pass cleanly, and summarize the architecture.",
    ]

    with ephemeral_cli_cred_dir(sandbox.root, "codex") as cred_dir:
        for prompt in all_prompts:
            cur_med, _ = estimator.estimate_messages(messages, codex_tools)
            sig = analyze_agentic_signals(messages)
            if cur_med >= spec.target_tokens + 600 and sig["tool_calls"] >= min_tools and sig["fail_then_pass"]:
                break
            messages.append({"role": "user", "content": prompt})
            if thread_id is None:
                inner_cmd = [
                    "codex",
                    "exec",
                    "--skip-git-repo-check",
                    "--json",
                    "--dangerously-bypass-approvals-and-sandbox",
                    "-C",
                    "/sandbox/work",
                    prompt,
                ]
            else:
                inner_cmd = [
                    "codex",
                    "exec",
                    "resume",
                    "--skip-git-repo-check",
                    "--json",
                    "--dangerously-bypass-approvals-and-sandbox",
                    thread_id,
                    prompt,
                ]
            cmd = build_cli_bwrap_argv(
                sandbox_root=sandbox.root,
                cred_dir=cred_dir,
                cli_kind="codex",
                inner_cmd=inner_cmd,
            )
            proc = subprocess.run(
                cmd,
                cwd=work,
                env={},
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=360,
            )
            if proc.returncode != 0 and not proc.stdout.strip():
                raise RuntimeError(f"codex exec failed (rc={proc.returncode}): {proc.stderr[:400]}")

            pending_text = ""
            for line in proc.stdout.splitlines():
                line = line.strip()
                if not line.startswith("{"):
                    continue
                ev = json.loads(line)
                raw_events_count += 1
                ev_type = ev.get("type")
                if ev_type == "thread.started" and ev.get("thread_id"):
                    thread_id = ev["thread_id"]
                elif ev_type == "item.completed":
                    item = ev.get("item") or {}
                    itype = item.get("type")
                    if itype == "agent_message" and item.get("text"):
                        txt = _sanitize_sandbox_paths(item["text"], work)
                        pending_text = (pending_text + "\n" + txt).strip() if pending_text else txt
                    elif itype == "command_execution":
                        call_idx += 1
                        tc_id = f"codex_cmd_{call_idx}"
                        cmd_str = _sanitize_sandbox_paths(item.get("command") or "", work)
                        out_str = _sanitize_sandbox_paths(item.get("aggregated_output") or "", work)
                        exit_code = item.get("exit_code")
                        if len(out_str) > MAX_TOOL_OUTPUT_CHARS:
                            out_str = out_str[:2500] + "\n... [truncated] ...\n" + out_str[-2500:]
                        messages.append(
                            {
                                "role": "assistant",
                                "content": pending_text,
                                "tool_calls": [
                                    {
                                        "id": tc_id,
                                        "type": "function",
                                        "function": {
                                            "name": "shell_command",
                                            "arguments": {"command": cmd_str},
                                        },
                                    }
                                ],
                            }
                        )
                        pending_text = ""
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc_id,
                                "name": "shell_command",
                                "content": f"exit_code={exit_code}\n{out_str}".strip(),
                            }
                        )
            if pending_text:
                messages.append({"role": "assistant", "content": pending_text})

    trimmed, est_tokens, by_tok = cut_at_message_boundary(
        messages,
        codex_tools,
        estimator,
        target_tokens=spec.target_tokens,
        min_tokens=spec.min_tokens,
        max_tokens=spec.max_tokens,
        min_user_turns=2,
        min_tool_calls=min_tools,
        require_fail_then_pass=True,
    )
    return _package_record(
        spec=spec,
        messages=trimmed,
        tools=codex_tools,
        est_tokens=est_tokens,
        by_tok=by_tok,
        provenance={
            "source_kind": spec.source_kind,
            "generating_model": spec.model,
            "cli_driver": "codex-cli",
            "generated_at_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "raw_events": raw_events_count,
            "raw_message_count": len(messages),
            "trimmed_message_count": len(trimmed),
            "sandbox_tool_calls": call_idx,
        },
    )


def run_claude_cli_case(
    spec: CaseSpec, sandbox_root: Path, estimator: TokenEstimator
) -> dict[str, Any]:
    """Drive real `claude -p --output-format stream-json` inside bwrap with ephemeral CLAUDE_CONFIG_DIR."""
    sandbox = WorkspaceSandbox(sandbox_root / spec.id)
    work = sandbox.work_dir
    subprocess.run(["git", "init", "-q"], cwd=work, check=True)

    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "You are Claude Code, an autonomous software engineering agent operating inside "
                "a sandboxed workspace with Read, Write, Edit, Bash, Glob, and Grep tools."
            ),
        }
    ]
    claude_tools = [
        {"type": "function", "function": {"name": "Read", "description": "Read a file.", "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}}}}},
        {"type": "function", "function": {"name": "Write", "description": "Write a file.", "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}, "content": {"type": "string"}}}}},
        {"type": "function", "function": {"name": "Edit", "description": "Edit a file.", "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}, "old_string": {"type": "string"}, "new_string": {"type": "string"}}}}},
        {"type": "function", "function": {"name": "Bash", "description": "Run a shell command.", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}}}},
        {"type": "function", "function": {"name": "Glob", "description": "Find files by glob.", "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}}}}},
        {"type": "function", "function": {"name": "Grep", "description": "Search file contents.", "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}}}}},
    ]

    session_id: str | None = None
    model_seen: str = spec.model
    raw_events_count = 0
    tool_calls_count = 0
    tool_names_by_id: dict[str, str] = {}
    min_tools = 15 if spec.length_class == "32k" else 12

    all_prompts = list(spec.phase_prompts) + [
        "Follow TDD: FIRST add failing unit tests for snapshot compaction under concurrent leader "
        "step-down and run `go test -race -v ./...` to observe the failure, THEN implement the fix "
        "and re-run `go test -race -v ./...` until all tests pass.",
        "Add a linearizability checker test that records every client submit/claim/complete "
        "invocation and return timestamp and validates single-owner lease invariants under leader "
        "churn, then run `go test -race ./...`.",
    ]

    with ephemeral_cli_cred_dir(sandbox.root, "claude") as cred_dir:
        for prompt in all_prompts:
            cur_med, _ = estimator.estimate_messages(messages, claude_tools)
            sig = analyze_agentic_signals(messages)
            if cur_med >= spec.target_tokens + 1500 and sig["tool_calls"] >= min_tools and sig["fail_then_pass"]:
                break
            messages.append({"role": "user", "content": prompt})
            inner_cmd = [
                "claude",
                "-p",
                "--output-format",
                "stream-json",
                "--verbose",
                "--permission-mode",
                "bypassPermissions",
                "--tools",
                "Read,Write,Edit,Bash,Glob,Grep",
                "--max-turns",
                "10",
            ]
            if session_id is not None:
                inner_cmd.extend(["--resume", session_id])
            inner_cmd.append(prompt)

            cmd = build_cli_bwrap_argv(
                sandbox_root=sandbox.root,
                cred_dir=cred_dir,
                cli_kind="claude",
                inner_cmd=inner_cmd,
            )
            proc = subprocess.run(
                cmd,
                cwd=work,
                env={},
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=600,
            )
            if proc.returncode != 0 and not proc.stdout.strip():
                raise RuntimeError(f"claude CLI failed (rc={proc.returncode}): {proc.stderr[:400]}")

            pending_assistant_id: str | None = None
            pending_assistant: dict[str, Any] | None = None

            def flush_assistant() -> None:
                nonlocal pending_assistant_id, pending_assistant
                if pending_assistant is not None:
                    if pending_assistant.get("content") or pending_assistant.get("tool_calls"):
                        messages.append(pending_assistant)
                    pending_assistant_id = None
                    pending_assistant = None

            for line in proc.stdout.splitlines():
                line = line.strip()
                if not line.startswith("{"):
                    continue
                ev = json.loads(line)
                raw_events_count += 1
                if ev.get("session_id") and session_id is None:
                    session_id = ev["session_id"]
                etype = ev.get("type")
                if etype == "assistant":
                    mobj = ev.get("message") or {}
                    if mobj.get("model"):
                        model_seen = str(mobj["model"])
                    mid = mobj.get("id")
                    if pending_assistant is None or mid != pending_assistant_id:
                        flush_assistant()
                        pending_assistant_id = mid
                        pending_assistant = {"role": "assistant", "content": ""}
                    for blk in mobj.get("content") or []:
                        btype = blk.get("type")
                        if btype == "text" and blk.get("text"):
                            sep = "\n" if pending_assistant["content"] else ""
                            pending_assistant["content"] += sep + _sanitize_sandbox_paths(blk["text"], work)
                        elif btype == "tool_use":
                            tool_calls_count += 1
                            tc_id = str(blk.get("id") or f"claude_tool_{tool_calls_count}")
                            tname = str(blk.get("name") or "unknown")
                            targs = _sanitize_sandbox_paths(blk.get("input") or {}, work)
                            tool_names_by_id[tc_id] = tname
                            pending_assistant.setdefault("tool_calls", []).append(
                                {
                                    "id": tc_id,
                                    "type": "function",
                                    "function": {"name": tname, "arguments": targs},
                                }
                            )
                elif etype == "user":
                    flush_assistant()
                    mobj = ev.get("message") or {}
                    for blk in mobj.get("content") or []:
                        if isinstance(blk, dict) and blk.get("type") == "tool_result":
                            tc_id = str(blk.get("tool_use_id") or "")
                            raw_c = blk.get("content", "")
                            if isinstance(raw_c, list):
                                txt = "\n".join(
                                    x.get("text", "") for x in raw_c if isinstance(x, dict)
                                )
                            else:
                                txt = str(raw_c)
                            txt = _sanitize_sandbox_paths(txt, work)
                            if len(txt) > MAX_TOOL_OUTPUT_CHARS:
                                txt = txt[:2500] + "\n... [truncated] ...\n" + txt[-2500:]
                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tc_id,
                                    "name": tool_names_by_id.get(tc_id, "tool"),
                                    "content": txt,
                                }
                            )
            flush_assistant()

    trimmed, est_tokens, by_tok = cut_at_message_boundary(
        messages,
        claude_tools,
        estimator,
        target_tokens=spec.target_tokens,
        min_tokens=spec.min_tokens,
        max_tokens=spec.max_tokens,
        min_user_turns=2,
        min_tool_calls=min_tools,
        require_fail_then_pass=True,
    )
    return _package_record(
        spec=spec,
        messages=trimmed,
        tools=claude_tools,
        est_tokens=est_tokens,
        by_tok=by_tok,
        provenance={
            "source_kind": spec.source_kind,
            "generating_model": model_seen,
            "cli_driver": "claude-code-cli",
            "generated_at_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "raw_events": raw_events_count,
            "raw_message_count": len(messages),
            "trimmed_message_count": len(trimmed),
            "sandbox_tool_calls": tool_calls_count,
        },
    )


def _sanitize_sandbox_paths(obj: Any, work_dir: Path) -> Any:
    prefixes = (str(work_dir), "/sandbox/work", "/sandbox")
    if isinstance(obj, str):
        out = obj
        for p in prefixes:
            out = out.replace(p, "/workspace")
        return out
    if isinstance(obj, dict):
        return {k: _sanitize_sandbox_paths(v, work_dir) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_sandbox_paths(v, work_dir) for v in obj]
    return obj


def _trim_markdown_to_token_budget(text: str, budget_tokens: int, estimator: TokenEstimator) -> str:
    """Trim markdown cleanly at paragraph/section boundaries to fit within budget_tokens."""
    paragraphs = text.split("\n\n")
    kept: list[str] = []
    running = ""
    for para in paragraphs:
        cand = (running + "\n\n" + para) if running else para
        med = sorted(estimator.estimate_text(cand).values())[1]
        if med > budget_tokens and kept:
            break
        kept.append(para)
        running = cand
    return "\n\n".join(kept)


def _training_data_git_rev() -> str:
    return subprocess.check_output(
        ["git", "-C", str(TRAINING_DATA_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()


def _call_qa_turn(
    *,
    spec: CaseSpec,
    messages: list[dict[str, Any]],
    call_counter: list[int],
    providers_seen: set[str],
) -> dict[str, Any]:
    """Run one QA turn with full documented max_tokens and medium/high reasoning effort.

    If a call hits finish_reason=length, keeps the partial reasoning/content and issues
    a continue turn while keeping reasoning enabled (never disables reasoning).
    """
    effort = spec.reasoning_effort
    for attempt in range(3):
        t0 = time.monotonic()
        resp = call_openrouter(
            model=spec.model,
            messages=messages,
            provider_only=spec.provider_only,
            expected_provider_names=spec.expected_provider_names,
            tools=None,
            reasoning_effort=effort,
        )
        providers_seen.add(str(resp.get("provider", "")))
        choice = resp["choices"][0]
        msg = normalize_assistant_message(choice["message"], call_counter)
        print(
            f"  [QA]   {spec.id} fin={choice.get('finish_reason')} "
            f"reas={len(msg.get('reasoning_content') or '')} cont={len(msg.get('content') or '')} "
            f"({time.monotonic() - t0:.1f}s)",
            flush=True,
        )
        if choice.get("finish_reason") != "length" and (msg.get("content") or "").strip():
            return msg
        # Preserve partial reasoning and ask the model to continue
        if msg.get("reasoning_content") or (msg.get("content") or "").strip():
            if not (msg.get("content") or "").strip():
                msg["content"] = "[Truncated at output limit; continuing in next turn.]"
            messages.append(msg)
        messages.append({"role": "user", "content": "Continue from where you left off and finish your answer."})
    return msg


def run_wiki_qa_case(spec: CaseSpec, estimator: TokenEstimator) -> dict[str, Any]:
    """Build a long-document Wikipedia dossier + multi-turn reasoning QA referencing early/middle/late sections."""
    doc_budget = int(spec.extra["doc_budget_tokens"])
    per_doc_budget = doc_budget // len(spec.local_sources)

    doc_blocks: list[str] = []
    source_records: list[dict[str, Any]] = []
    for idx, rel in enumerate(spec.local_sources, 1):
        p = TRAINING_DATA_ROOT / rel
        if not p.is_file():
            raise FileNotFoundError(f"Missing local wiki article: {p}")
        raw_bytes = p.read_bytes()
        full_text = raw_bytes.decode("utf-8")
        trimmed_text = _trim_markdown_to_token_budget(full_text, per_doc_budget, estimator)
        wiki_lang = rel.split("/")[2]
        inv_path = TRAINING_DATA_ROOT / "inventory/wikipedia-pages-articles" / f"{wiki_lang}-20260701-full.json"
        dump_date = "20260701"
        if inv_path.is_file():
            inv = json.loads(inv_path.read_text(encoding="utf-8"))
            dump_date = str(inv.get("dump_date") or inv.get("snapshot_id") or "20260701")
        source_records.append(
            {
                "source_path": rel,
                "wiki_db": wiki_lang,
                "dump": dump_date,
                "full_sha256": hashlib.sha256(raw_bytes).hexdigest(),
                "excerpt_sha256": hashlib.sha256(trimmed_text.encode("utf-8")).hexdigest(),
                "excerpt_chars": len(trimmed_text),
            }
        )
        doc_blocks.append(f"=== DOCUMENT {idx}: {rel} ===\n\n{trimmed_text}")

    dossier = "\n\n".join(doc_blocks)
    qa_turns: tuple[str, ...] = tuple(spec.extra["qa_turns"])

    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "You are a meticulous research analyst. Answer every question strictly from the "
                "provided reference documents, citing specific early, middle, and late sections, "
                "dates, and names accurately without fabricating details. Keep each answer thorough "
                "but focused (~600-1000 words)."
            ),
        },
        {
            "role": "user",
            "content": f"{dossier}\n\n---\n\n{qa_turns[0]}",
        },
    ]

    providers_seen: set[str] = set()
    call_counter = [0]
    api_calls = 0

    for turn_idx in range(len(qa_turns)):
        if turn_idx > 0:
            cur_med, _ = estimator.estimate_messages(messages, None)
            if cur_med >= spec.target_tokens and turn_idx >= 3:
                break
            if cur_med >= spec.max_tokens:
                break
            messages.append({"role": "user", "content": qa_turns[turn_idx]})
        api_calls += 1
        messages.append(
            _call_qa_turn(
                spec=spec,
                messages=messages,
                call_counter=call_counter,
                providers_seen=providers_seen,
            )
        )

    cur_med, _ = estimator.estimate_messages(messages, None)
    if cur_med < spec.min_tokens:
        messages.append(
            {
                "role": "user",
                "content": (
                    "Finally, compile a concise chronological timeline table (15-20 entries) spanning "
                    "the earliest to latest dated events across all provided documents, tagging each "
                    "row with its document number and section topic."
                ),
            }
        )
        api_calls += 1
        messages.append(
            _call_qa_turn(
                spec=spec,
                messages=messages,
                call_counter=call_counter,
                providers_seen=providers_seen,
            )
        )

    trimmed, est_tokens, by_tok = cut_at_message_boundary(
        messages,
        None,
        estimator,
        target_tokens=spec.target_tokens,
        min_tokens=spec.min_tokens,
        max_tokens=spec.max_tokens,
        min_user_turns=3,
        case_id=spec.id,
    )
    return _package_record(
        spec=spec,
        messages=trimmed,
        tools=None,
        est_tokens=est_tokens,
        by_tok=by_tok,
        provenance={
            "source_kind": spec.source_kind,
            "training_data_revision": _training_data_git_rev(),
            "wiki_sources": source_records,
            "generating_model": spec.model,
            "providers_used": sorted(providers_seen),
            "generated_at_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "api_calls": api_calls,
        },
    )


def run_oss_repo_qa_case(spec: CaseSpec, estimator: TokenEstimator) -> dict[str, Any]:
    """Build a multi-file OSS repository context + multi-turn deep code audit with reasoning."""
    repo_root = TRAINING_DATA_ROOT / spec.oss_repo_dir
    if not repo_root.is_dir():
        raise FileNotFoundError(f"Missing local OSS repo dir: {repo_root}")

    doc_budget = int(spec.extra["doc_budget_tokens"])
    file_blocks: list[str] = []
    source_files: list[dict[str, Any]] = []
    used_tokens = 0

    for rel in spec.oss_include_globs:
        p = repo_root / rel
        if not p.is_file():
            raise FileNotFoundError(f"Missing OSS source file: {p}")
        raw_bytes = p.read_bytes()
        lines = raw_bytes.decode("utf-8", errors="replace").splitlines()
        numbered = "\n".join(f"{i:4d}: {line}" for i, line in enumerate(lines, 1))
        remaining = doc_budget - used_tokens
        if remaining < 800:
            break
        cand_tok = sorted(estimator.estimate_text(numbered).values())[1]
        if cand_tok > remaining:
            lo, hi = 1, len(lines)
            while lo < hi:
                mid = (lo + hi + 1) // 2
                s = "\n".join(f"{i:4d}: {lines[i - 1]}" for i in range(1, mid + 1))
                if sorted(estimator.estimate_text(s).values())[1] <= remaining:
                    lo = mid
                else:
                    hi = mid - 1
            numbered = "\n".join(f"{i:4d}: {lines[i - 1]}" for i in range(1, lo + 1))
            included_lines = lo
        else:
            included_lines = len(lines)
        block = f"=== FILE: {spec.oss_repo_slug}/{rel} (lines 1-{included_lines} of {len(lines)}) ===\n{numbered}"
        used_tokens += sorted(estimator.estimate_text(block).values())[1]
        file_blocks.append(block)
        source_files.append(
            {
                "repo": spec.oss_repo_slug,
                "commit": spec.oss_repo_commit,
                "source_path": f"{spec.oss_repo_dir}/{rel}",
                "file": rel,
                "included_lines": included_lines,
                "total_lines": len(lines),
                "full_sha256": hashlib.sha256(raw_bytes).hexdigest(),
                "excerpt_sha256": hashlib.sha256(block.encode("utf-8")).hexdigest(),
            }
        )

    code_bundle = "\n\n".join(file_blocks)
    qa_turns: tuple[str, ...] = tuple(spec.extra["qa_turns"])

    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "You are a senior systems and compiler engineer performing a deep multi-file "
                "source code audit. Cite exact file paths, symbol names, and line numbers from "
                "the provided source files. Keep each turn thorough and concise (~600-1000 words)."
            ),
        },
        {
            "role": "user",
            "content": f"{code_bundle}\n\n---\n\n{qa_turns[0]}",
        },
    ]

    providers_seen: set[str] = set()
    call_counter = [0]
    api_calls = 0

    for turn_idx in range(len(qa_turns)):
        if turn_idx > 0:
            cur_med, _ = estimator.estimate_messages(messages, None)
            if cur_med >= spec.target_tokens and turn_idx >= 3:
                break
            if cur_med >= spec.max_tokens:
                break
            messages.append({"role": "user", "content": qa_turns[turn_idx]})
        api_calls += 1
        messages.append(
            _call_qa_turn(
                spec=spec,
                messages=messages,
                call_counter=call_counter,
                providers_seen=providers_seen,
            )
        )

    cur_med, _ = estimator.estimate_messages(messages, None)
    if cur_med < spec.min_tokens:
        messages.append(
            {
                "role": "user",
                "content": (
                    "Write a comprehensive unit/integration test plan (with concrete input shapes, "
                    "boundary cases, and assertions referencing exact functions from the earliest and "
                    "latest files above) to guard against regressions in every invariant you identified."
                ),
            }
        )
        api_calls += 1
        messages.append(
            _call_qa_turn(
                spec=spec,
                messages=messages,
                call_counter=call_counter,
                providers_seen=providers_seen,
            )
        )

    trimmed, est_tokens, by_tok = cut_at_message_boundary(
        messages,
        None,
        estimator,
        target_tokens=spec.target_tokens,
        min_tokens=spec.min_tokens,
        max_tokens=spec.max_tokens,
        min_user_turns=3,
        case_id=spec.id,
    )
    return _package_record(
        spec=spec,
        messages=trimmed,
        tools=None,
        est_tokens=est_tokens,
        by_tok=by_tok,
        provenance={
            "source_kind": spec.source_kind,
            "training_data_revision": _training_data_git_rev(),
            "repo": spec.oss_repo_slug,
            "repo_commit": spec.oss_repo_commit,
            "oss_files": source_files,
            "generating_model": spec.model,
            "providers_used": sorted(providers_seen),
            "generated_at_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "api_calls": api_calls,
        },
    )


def canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _package_record(
    *,
    spec: CaseSpec,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    est_tokens: int,
    by_tok: dict[str, int],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    has_reasoning = any(
        isinstance(m.get("reasoning_content"), str) and bool(m["reasoning_content"].strip())
        for m in messages
        if m.get("role") == "assistant"
    )
    reasoning_chars = sum(
        len(m.get("reasoning_content") or "") for m in messages if m.get("role") == "assistant"
    )
    tool_call_count = sum(
        len(m.get("tool_calls") or []) for m in messages if m.get("role") == "assistant"
    )
    payload_for_hash = {"messages": messages, "tools": tools}
    content_sha256 = hashlib.sha256(canonical_json_bytes(payload_for_hash)).hexdigest()
    source_identity = hashlib.sha256(
        canonical_json_bytes({"id": spec.id, "group": spec.group, "content_sha256": content_sha256})
    ).hexdigest()

    record: dict[str, Any] = {
        "id": spec.id,
        "component": "long_context",
        "length_class": spec.length_class,
        "split": spec.split,
        "group": spec.group,
        "source": spec.source_kind,
        "archetype": spec.archetype,
        "generating_model": provenance.get("generating_model", spec.model),
        "task": spec.task_name,
        "language": list(spec.languages),
        "has_reasoning": has_reasoning,
        "reasoning_chars": reasoning_chars,
        "tool_call_count": tool_call_count,
        "message_count": len(messages),
        "token_count_estimate": est_tokens,
        "token_counts_by_reference_tokenizer": by_tok,
        "content_sha256": content_sha256,
        "source_identity": source_identity,
        "license_notes": spec.license_notes,
        "provenance": provenance,
        "messages": messages,
    }
    if tools is not None:
        record["tools"] = tools
    return record


def generate_case(spec: CaseSpec, sandbox_root: Path, estimator: TokenEstimator) -> dict[str, Any]:
    if spec.source_kind == "openrouter_agentic":
        return run_openrouter_agentic_case(spec, sandbox_root, estimator)
    if spec.source_kind == "codex_cli":
        return run_codex_cli_case(spec, sandbox_root, estimator)
    if spec.source_kind == "claude_cli":
        return run_claude_cli_case(spec, sandbox_root, estimator)
    if spec.source_kind == "wiki_qa":
        return run_wiki_qa_case(spec, estimator)
    if spec.source_kind == "oss_repo_qa":
        return run_oss_repo_qa_case(spec, estimator)
    raise ValueError(f"Unknown source_kind: {spec.source_kind}")
