#!/usr/bin/env python3
"""Generate, assemble, validate, and (when explicitly approved) publish the
``long_context`` calibration bank (16K and 32K) as the third component of
``agentic-quant-panel-v3`` (preserving ``agentic`` and ``exl3_standard``
byte-for-byte from ``agentic-quant-panel-v2``).

Subcommands:
- ``generate``: run specified (or all) cases into scratch ``raw_traces/``, enforcing
  privacy/secret scanning and per-trace quality acceptance gates before saving.
- ``assemble``: write ``long_context/{16k,32k}.jsonl``, ``long_context/manifest.json``,
  Parquet exports, and stage a complete updated revision of the HF dataset repo
  preserving every existing v1 and v2 file byte-for-byte.
- ``validate``: run privacy/secret scanning, quality gates, Standard Fidelity exclusion
  checks, split disjointness checks, schema/hash checks, and Hugging Face
  ``apply_chat_template`` checks across local target tokenizers.
- ``publish``: gated upload over HTTP LFS (``HF_HUB_DISABLE_XET=1``) using
  ``huggingface-hub==1.33.0`` and the default HF token file. Requires ``--confirm-upload``.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import sys
import time
import traceback
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.corpus.long_context_harness import (  # noqa: E402
    TokenEstimator,
    canonical_json_bytes,
    check_trace_quality,
    format_trace_summary,
    generate_case,
)
from scripts.corpus.long_context_specs import (  # noqa: E402
    CASES,
    CASES_BY_ID,
    CHAT_TEMPLATE_VERIFY_TOKENIZERS,
    FIDELITY_CACHE_ROOT,
    OPENROUTER_KEY_PATH,
    V2_PANEL_SNAPSHOT,
)


DEFAULT_WORK_ROOT = Path.home() / ".cache/cuteafd/builds/long-context-bank"
HF_V2_REPO = "wrldsuksgo2mars/agentic-quant-panel-v2"
HF_V2_REVISION = "54683731eb01c7c2fa5b801a0888e029a0e1b1c7"
HF_DATASET_REPO = "wrldsuksgo2mars/agentic-quant-panel-v3"

# Secret & credential patterns that must never appear in any trace or metadata.
SECRET_REGEXES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("openrouter_key", re.compile(r"sk-or-v1-[A-Za-z0-9]{16,}")),
    ("sk_token", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}")),
    ("hf_token", re.compile(r"\bhf_[A-Za-z0-9]{20,}")),
    ("github_pat", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}")),
    ("aws_access_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("private_key_block", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----")),
    ("bearer_secret", re.compile(r"Bearer\s+[A-Za-z0-9._-]{20,}", re.IGNORECASE)),
)

def _build_privacy_regexes() -> tuple[tuple[str, re.Pattern[str]], ...]:
    host = socket.gethostname().strip()
    hostnames = [
        "raptor", "maxq", "ostrich", "dodo", "emu", "kiwi",
        "rhea", "moa", "kakapo", "cassowary", "sparknest",
    ]
    if host and host.lower() not in {h.lower() for h in hostnames} and host.lower() not in ("localhost", "sandbox"):
        hostnames.append(re.escape(host))
        short_host = host.split(".")[0]
        if short_host and short_host.lower() not in {h.lower() for h in hostnames}:
            hostnames.append(re.escape(short_host))
    return (
        ("private_hostname", re.compile(r"\b(?:" + "|".join(hostnames) + r")\b", re.IGNORECASE)),
        ("home_tj_path", re.compile(r"/home/tj\b", re.IGNORECASE)),
        ("host_username_tj", re.compile(r"\btj\b", re.IGNORECASE)),
        ("ls_owner_tj", re.compile(r"\btj\s+tj\b", re.IGNORECASE)),
        ("dev_dir_path", re.compile(r"(?:~/Developer|Developer/cuteafd)", re.IGNORECASE)),
        ("private_handle", re.compile(r"\b(?:wrldsuksgo2mars|tpurtell)\b", re.IGNORECASE)),
        ("cuteafd_internal", re.compile(r"\b(?:cuteafd|sparkinfer|hwlock|wip\.sh)\b", re.IGNORECASE)),
        ("email_address", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    )


# TJ-private hostnames (including real host), username, paths, handles, internals, and emails.
PRIVACY_REGEXES: tuple[tuple[str, re.Pattern[str]], ...] = _build_privacy_regexes()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_live_secret_literals() -> list[tuple[str, str]]:
    """Load local secret file values in memory ONLY for exact-substring leakage scanning."""
    secrets: list[tuple[str, str]] = []
    for label, path in (
        ("openrouter_key_file", OPENROUTER_KEY_PATH),
        ("hf_token_file", Path.home() / ".cache/huggingface/token"),
    ):
        if path.is_file():
            val = path.read_text(encoding="utf-8", errors="ignore").strip()
            if len(val) >= 12:
                secrets.append((label, val))
    return secrets


def scan_text_for_secrets(text: str, live_literals: list[tuple[str, str]] | None = None) -> list[str]:
    """Scan text for API keys, tokens, and live credential substrings."""
    findings: list[str] = []
    for label, pat in SECRET_REGEXES:
        if pat.search(text):
            findings.append(f"secret:{label}")
    for label, lit in (live_literals or []):
        if lit in text:
            findings.append(f"literal:{label}")
    return findings


def scan_text_for_privacy_and_secrets(
    text: str, live_literals: list[tuple[str, str]] | None = None
) -> list[str]:
    """Scan text for both credentials/secrets AND TJ-private hostnames/paths/internals/emails."""
    findings = scan_text_for_secrets(text, live_literals)
    for label, pat in PRIVACY_REGEXES:
        m = pat.search(text)
        if m:
            findings.append(f"privacy:{label}:{m.group(0)}")
    return findings


def load_fidelity_exclusions() -> tuple[set[str], set[str]]:
    """Collect all Standard Fidelity source paths and distinctive fixture identifiers."""
    paths: set[str] = set()
    markers: set[str] = {"scripts/fixtures/agentic-repo/ledger", "cuteafd-fidelity"}
    if FIDELITY_CACHE_ROOT.is_dir():
        for wf in FIDELITY_CACHE_ROOT.rglob("windows.json"):
            try:
                data = json.loads(wf.read_text(encoding="utf-8"))
                for src in data.get("sources", []):
                    p = src.get("path")
                    if isinstance(p, str) and p:
                        paths.add(p)
            except Exception:
                continue
    return paths, markers


def cmd_generate(args: argparse.Namespace) -> int:
    work_root = Path(args.work_root).expanduser().resolve()
    raw_dir = work_root / "raw_traces"
    sandbox_root = work_root / "sandboxes"
    raw_dir.mkdir(parents=True, exist_ok=True)
    sandbox_root.mkdir(parents=True, exist_ok=True)

    selected = list(CASES) if not args.cases else [CASES_BY_ID[cid] for cid in args.cases]
    estimator = TokenEstimator()
    live_secrets = _load_live_secret_literals()

    todo = []
    for spec in selected:
        out_path = raw_dir / f"{spec.id}.json"
        if out_path.exists() and not args.force:
            print(f"[SKIP] {spec.id} already exists at {out_path}", flush=True)
            continue
        todo.append(spec)

    if not todo:
        print("All requested cases already exist.", flush=True)
        return 0

    def _run_one(spec):
        last_issues: list[str] = []
        for attempt in range(1, args.retries + 2):
            t0 = time.monotonic()
            print(
                f"[START] {spec.id} (attempt {attempt}, {spec.length_class}, {spec.split}, {spec.model})",
                flush=True,
            )
            record = generate_case(spec, sandbox_root, estimator)
            serialized = json.dumps(record, ensure_ascii=False, sort_keys=True)
            privacy_hits = scan_text_for_privacy_and_secrets(serialized, live_secrets)
            quality_defects = check_trace_quality(record, spec)
            dt_s = time.monotonic() - t0
            if not privacy_hits and not quality_defects:
                out_path = raw_dir / f"{spec.id}.json"
                tmp_path = raw_dir / f".{spec.id}.json.tmp"
                tmp_path.write_text(
                    json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                tmp_path.replace(out_path)
                print(
                    f"[PASS]  {spec.id} in {dt_s:.1f}s\n{format_trace_summary(record)}",
                    flush=True,
                )
                return record
            last_issues = [f"privacy={privacy_hits}", f"quality={quality_defects}"]
            if not privacy_hits:
                rej_dir = raw_dir / "rejected"
                rej_dir.mkdir(exist_ok=True)
                (rej_dir / f"{spec.id}.attempt{attempt}.json").write_text(
                    json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            print(
                f"[REJECT] {spec.id} attempt {attempt} ({dt_s:.1f}s): {last_issues}\n"
                f"{format_trace_summary(record)}",
                flush=True,
            )
        raise RuntimeError(f"Case {spec.id} failed acceptance/privacy gate after retries: {last_issues}")

    failures: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futs = {pool.submit(_run_one, spec): spec for spec in todo}
        for fut in as_completed(futs):
            spec = futs[fut]
            try:
                fut.result()
            except Exception as exc:
                failures.append((spec.id, f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"))
                print(f"[FAIL]  {spec.id}: {type(exc).__name__}: {exc}", flush=True)

    if failures:
        print(f"\n{len(failures)} case(s) failed:", file=sys.stderr)
        for cid, tb in failures:
            print(f"--- {cid} ---\n{tb}", file=sys.stderr)
        return 1
    return 0


def sanitize_reproduction_script(filename: str, text: str) -> str:
    """Produce a public-safe reproduction copy with neutral env-var paths,
    generic HTTP headers, externalized privacy denylist, and no private infrastructure strings."""
    out = text
    # 1. Strip private paths & replace with neutral env-var defaults
    replacements = [
        (
            'TRAINING_DATA_ROOT = Path.home() / "Developer/training-data"',
            'TRAINING_DATA_ROOT = Path(os.environ.get("TRAINING_DATA_ROOT", "./training-data"))',
        ),
        (
            'V2_PANEL_SNAPSHOT = Path(\n'
            '    "/mnt/sparknest/hf-home/hub/datasets--wrldsuksgo2mars--agentic-quant-panel-v2"\n'
            '    "/snapshots/54683731eb01c7c2fa5b801a0888e029a0e1b1c7"\n'
            ')',
            'V2_PANEL_SNAPSHOT = Path(os.environ.get("LC_BASE_PANEL_SNAPSHOT", "./agentic-quant-panel-v2"))',
        ),
        (
            'FIDELITY_CACHE_ROOT = Path.home() / ".cache/cuteafd/fidelity/wrldsuksgo2mars--cuteafd-fidelity"',
            'FIDELITY_CACHE_ROOT = Path(os.environ.get("FIDELITY_CACHE_ROOT", "./fidelity-reference"))',
        ),
        (
            'OPENROUTER_KEY_PATH = Path.home() / ".config/cuteafd/secrets/openrouter-key"',
            'OPENROUTER_KEY_PATH = Path(os.environ.get("OPENROUTER_KEY_FILE", "./openrouter-key"))',
        ),
        (
            'DEFAULT_WORK_ROOT = Path.home() / ".cache/cuteafd/builds/long-context-bank"',
            'DEFAULT_WORK_ROOT = Path(os.environ.get("LC_BANK_WORK_ROOT", "./work/long-context-bank"))',
        ),
        (
            'HF_V2_REPO = "wrldsuksgo2mars/agentic-quant-panel-v2"\n'
            'HF_V2_REVISION = "54683731eb01c7c2fa5b801a0888e029a0e1b1c7"\n'
            'HF_DATASET_REPO = "wrldsuksgo2mars/agentic-quant-panel-v3"',
            'HF_V2_REPO = os.environ.get("LC_HF_V2_REPO", "agentic-quant-panel-v2")\n'
            'HF_V2_REVISION = "54683731eb01c7c2fa5b801a0888e029a0e1b1c7"\n'
            'HF_DATASET_REPO = os.environ.get("LC_HF_DATASET_REPO", "agentic-quant-panel-v3")',
        ),
        (
            '    "deepseek_v41": (\n'
            '        "/mnt/sparknest/hf-home/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017"\n'
            '        "/snapshots/1bae0e1088993b9b82de08f59de5be5a34e42170"\n'
            '    ),\n'
            '    "glm_53": (\n'
            '        "/mnt/sparknest/hf-home/hub/models--zai-org--GLM-5.3"\n'
            '        "/snapshots/aca966e4e02791568aa6a4ced368624b3d897f42"\n'
            '    ),\n'
            '    "qwen_38": (\n'
            '        "/mnt/sparknest/hf-home/hub/models--Qwen--Qwen3.8-27B"\n'
            '        "/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"\n'
            '    ),',
            '    "deepseek_v41": os.environ.get("LC_TOK_DEEPSEEK_V41", str(Path(os.environ.get("HF_HUB_CACHE", "./hf-cache")) / "deepseek-v41")),\n'
            '    "glm_53": os.environ.get("LC_TOK_GLM_53", str(Path(os.environ.get("HF_HUB_CACHE", "./hf-cache")) / "glm-53")),\n'
            '    "qwen_38": os.environ.get("LC_TOK_QWEN_38", str(Path(os.environ.get("HF_HUB_CACHE", "./hf-cache")) / "qwen-38")),',
        ),
        (
            '    "hy4_flash": (\n'
            '        "/mnt/sparknest/hf-home/hub/models--tencent--Hy4-flash"\n'
            '        "/snapshots/d88c9dfd96ba4f2e6b92adba0b8e8f1c8696f959"\n'
            '    ),',
            '    "hy4_flash": os.environ.get("LC_TOK_HY4_FLASH", str(Path(os.environ.get("HF_HUB_CACHE", "./hf-cache")) / "hy4-flash")),',
        ),
        (
            '"HTTP-Referer": "https://github.com/tpurtell/cuteafd",\n                "X-Title": "cuteafd-long-context-bank",',
            '"X-Title": "long-context-calibration-bank",',
        ),
        (
            '            raise RuntimeError("No local reference tokenizers found under /mnt/sparknest/hf-home/hub")',
            '            raise RuntimeError("No local reference tokenizers found under $HF_HUB_CACHE")',
        ),
        (
            '                    ".config/cuteafd",\n',
            '                    ".config/",\n',
        ),
        (
            'markers: set[str] = {"scripts/fixtures/agentic-repo/ledger", "cuteafd-fidelity"}',
            'markers: set[str] = {"scripts/fixtures/agentic-repo/ledger", "fidelity-panel"}',
        ),
    ]
    for old, new in replacements:
        out = out.replace(old, new)

    # 2. Ensure `import os` is present in long_context_specs.py if env vars were introduced
    if filename == "long_context_specs.py" and "\nimport os\n" not in out:
        out = out.replace("from pathlib import Path\n", "import os\nfrom pathlib import Path\n", 1)

    # 3. Replace inlined hostname/handle/internal privacy denylist in long_context_bank.py
    #    with an optional external file loader ($LC_PRIVACY_DENYLIST_FILE)
    if filename == "long_context_bank.py":
        start_marker = "def _build_privacy_regexes() -> tuple[tuple[str, re.Pattern[str]], ...]:"
        end_marker = "PRIVACY_REGEXES: tuple[tuple[str, re.Pattern[str]], ...] = _build_privacy_regexes()"
        s_idx = out.find(start_marker)
        e_idx = out.find(end_marker)
        if s_idx != -1 and e_idx != -1:
            replacement_block = (
                "def _build_privacy_regexes() -> tuple[tuple[str, re.Pattern[str]], ...]:\n"
                '    """Load optional local privacy denylist terms from $LC_PRIVACY_DENYLIST_FILE (one per line)."""\n'
                "    patterns: list[tuple[str, re.Pattern[str]]] = [\n"
                '        ("email_address", re.compile(r"\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}\\b")),\n'
                "    ]\n"
                '    deny_file = os.environ.get("LC_PRIVACY_DENYLIST_FILE", "")\n'
                "    if deny_file and Path(deny_file).is_file():\n"
                "        terms = [\n"
                '            line.strip() for line in Path(deny_file).read_text(encoding="utf-8").splitlines()\n'
                '            if line.strip() and not line.strip().startswith("#")\n'
                "        ]\n"
                "        if terms:\n"
                '            patterns.append(("local_denylist", re.compile("|".join(re.escape(t) for t in terms), re.IGNORECASE)))\n'
                "    return tuple(patterns)\n\n\n"
                "PRIVACY_REGEXES: tuple[tuple[str, re.Pattern[str]], ...] = _build_privacy_regexes()"
            )
            out = out[:s_idx] + replacement_block + out[e_idx + len(end_marker):]

        # Also remove the sanitize_reproduction_script helper's own literal patterns from the published copy
        san_start = out.find("def sanitize_reproduction_script(")
        san_end = out.find("\n\ndef _parquet_row(")
        if san_start != -1 and san_end != -1:
            out = (
                out[:san_start]
                + "def sanitize_reproduction_script(filename: str, text: str) -> str:\n"
                + '    """Identity sanitizer in already-sanitized public reproduction script."""\n'
                + "    return text"
                + out[san_end:]
            )

    # 4. Neutralize docstring/comment references to host paths, usernames, and private infra names
    doc_replacements = [
        ("``~/.cache/cuteafd/builds/long-context-bank/sandboxes/<case_id>``", "``$LC_BANK_WORK_ROOT/sandboxes/<case_id>``"),
        ("no /mnt/sparknest", "no shared storage mounts"),
        ("or /mnt/sparknest", "or shared storage mounts"),
        ("Host ``/home/tj``", "Host ``$HOME``"),
        ("Host /home/tj", "Host $HOME"),
        ("~/.config/cuteafd,", "host config dirs,"),
        ("``wrldsuksgo2mars/agentic-quant-panel-v2``", "``agentic-quant-panel-v2``"),
        ("TJ-private hostnames (including real host), username, paths, handles, internals, and emails.", "Optional local privacy denylist plus email address scanner."),
        ("TJ-private hostnames/paths/internals/emails", "local privacy denylist terms and emails"),
        ("(coordinator/TJ confirmation required)", "(explicit maintainer confirmation required)"),
        (
            "        # README.md and panel.json inherit the public HF repo namespace wrldsuksgo2mars/agentic-quant-panel-v1/v2\n"
            "        # from the existing v2 dataset card; check strictly for everything else (and zero hits on reproduction/ & long_context/).\n"
            '        if rel in ("README.md", "panel.json"):\n'
            '            hits = [h for h in hits if h != "privacy:private_handle:wrldsuksgo2mars"]\n',
            "",
        ),
    ]
    for old, new in doc_replacements:
        out = out.replace(old, new)

    # Remove any remaining comment/docstring mention of private tool/host names
    out = re.sub(r"\bcuteafd\b", "engine", out, flags=re.IGNORECASE)
    return out


def _parquet_row(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": record["id"],
        "component": record["component"],
        "length_class": record["length_class"],
        "split": record["split"],
        "group": record["group"],
        "source": record["source"],
        "archetype": record["archetype"],
        "generating_model": record["generating_model"],
        "task": record["task"],
        "language": list(record["language"]),
        "has_reasoning": bool(record["has_reasoning"]),
        "reasoning_chars": int(record["reasoning_chars"]),
        "tool_call_count": int(record["tool_call_count"]),
        "message_count": int(record["message_count"]),
        "token_count_estimate": int(record["token_count_estimate"]),
        "token_counts_json": json.dumps(record["token_counts_by_reference_tokenizer"], sort_keys=True),
        "content_sha256": record["content_sha256"],
        "source_identity": record["source_identity"],
        "license_notes": record["license_notes"],
        "provenance_json": json.dumps(record["provenance"], ensure_ascii=False, sort_keys=True),
        "messages_json": json.dumps(record["messages"], ensure_ascii=False),
        "tools_json": json.dumps(record["tools"], ensure_ascii=False) if "tools" in record else None,
    }


def _updated_readme(v2_readme: str, manifest: dict[str, Any], v2_sums: dict[str, str], v2_sums_sha256: str) -> str:
    # Retarget card metadata & title to Agentic Quant Panel v3
    out = v2_readme.replace("pretty_name: Agentic Quant Panel v2", "pretty_name: Agentic Quant Panel v3")
    out = out.replace(
        f"https://huggingface.co/datasets/{HF_V2_REPO}/blob/main/README.md#licensing-and-attribution",
        f"https://huggingface.co/datasets/{HF_DATASET_REPO}/blob/main/README.md#licensing-and-attribution",
    )
    out = out.replace("# Agentic Quant Panel v2", "# Agentic Quant Panel v3", 1)
    out = out.replace(
        "A calibration panel for post-training quantization. It has two components that are used together:",
        "A calibration panel for post-training quantization. It extends "
        f"[Agentic Quant Panel v2](https://huggingface.co/datasets/{HF_V2_REPO}) "
        f"(`{HF_V2_REVISION[:10]}`) with a third component, `long_context` (16K and 32K sequences), "
        "keeping all `agentic` and `exl3_standard` files byte-for-byte identical:",
        1,
    )
    out = out.replace(
        "| `exl3_standard` | The [ExLlamaV3](https://github.com/turboderp-org/exllamav3) standard calibration data (C4, code, multilingual, technical, Wikipedia, tiny stories, plus random-token rows), pre-split into 2,726 ordered text segments | **Raw text**, packed into fixed-width rows with the target tokenizer: 250 × 2,048 tokens (about 512K) by default |\n",
        "| `exl3_standard` | The [ExLlamaV3](https://github.com/turboderp-org/exllamav3) standard calibration data (C4, code, multilingual, technical, Wikipedia, tiny stories, plus random-token rows), pre-split into 2,726 ordered text segments | **Raw text**, packed into fixed-width rows with the target tokenizer: 250 × 2,048 tokens (about 512K) by default |\n"
        "| `long_context` | Optional 16K and 32K multi-turn agentic coding traces, OSS codebase audits, and multilingual long-document QA (16 sequences, ~365K est. tokens) with calibration and heldout splits | Multi-turn chat `messages` (+ optional `tools` and `reasoning_content`), rendered with the **target model's chat template** |\n",
        1,
    )
    out = out.replace(
        "Neither component contains model outputs or generated reasoning traces. Every token is a prompt or raw source text.",
        "Neither `agentic` nor `exl3_standard` contains model outputs or generated reasoning traces; every token in those two components is a prompt or raw source text. The optional `long_context` component adds full multi-turn trajectories including tool calls, tool outputs, and model reasoning traces.",
        1,
    )
    out = out.replace("@misc{agentic_quant_panel_v2,", "@misc{agentic_quant_panel_v3,")
    out = out.replace("title = {Agentic Quant Panel v2},", "title = {Agentic Quant Panel v3},")
    out = out.replace(
        f"url = {{https://huggingface.co/datasets/{HF_V2_REPO}}}",
        f"url = {{https://huggingface.co/datasets/{HF_DATASET_REPO}}}",
    )

    fm_addition = (
        "  - config_name: long_context_16k\n"
        "    data_files:\n"
        "      - split: calibration\n"
        "        path: data/long_context_16k_calibration.parquet\n"
        "      - split: heldout\n"
        "        path: data/long_context_16k_heldout.parquet\n"
        "  - config_name: long_context_32k\n"
        "    data_files:\n"
        "      - split: calibration\n"
        "        path: data/long_context_32k_calibration.parquet\n"
        "      - split: heldout\n"
        "        path: data/long_context_32k_heldout.parquet\n"
    )
    parts = out.split("---\n", 2)
    if len(parts) == 3 and "config_name: long_context_16k" not in parts[1]:
        frontmatter = parts[1].rstrip("\n") + "\n" + fm_addition
        body = parts[2]
        out = f"---\n{frontmatter}---\n{body}"

    s16 = manifest["subsets"]["16k"]
    s32 = manifest["subsets"]["32k"]
    key_v2_files = (
        "calibration.jsonl",
        "screening.jsonl",
        "heldout.jsonl",
        "manifest.json",
        "data/calibration.parquet",
        "data/screening.parquet",
        "data/heldout.parquet",
        "data/exl3_standard.parquet",
        "exl3_standard/manifest.json",
        "exl3_standard/segments.jsonl",
    )
    v2_hash_rows = "\n".join(
        f"| `{name}` | `{v2_sums[name]}` |" for name in key_v2_files if name in v2_sums
    )
    section = f"""
## Component 3: `long_context` (16K and 32K)

`long_context` extends the panel with 16K and 32K long-context sequences for long-range attention and MLA/DSA sparse-attention quantization (where top-k pruning activates beyond 2,048 tokens). All `agentic` and `exl3_standard` files are copied **byte-for-byte** from [`{HF_V2_REPO}@{HF_V2_REVISION[:10]}`](https://huggingface.co/datasets/{HF_V2_REPO}/tree/{HF_V2_REVISION}) (v2 `SHA256SUMS` digest `{v2_sums_sha256}`).

| Subset | Files | Calibration | Heldout | Total est. tokens |
| --- | --- | --- | --- | ---: |
| `16k` | `long_context/16k.jsonl` | {s16['splits']['calibration']['records']} seqs ({s16['splits']['calibration']['token_count_estimate']:,} tok) | {s16['splits']['heldout']['records']} seqs ({s16['splits']['heldout']['token_count_estimate']:,} tok) | {s16['total_token_count_estimate']:,} |
| `32k` | `long_context/32k.jsonl` | {s32['splits']['calibration']['records']} seqs ({s32['splits']['calibration']['token_count_estimate']:,} tok) | {s32['splits']['heldout']['records']} seqs ({s32['splits']['heldout']['token_count_estimate']:,} tok) | {s32['total_token_count_estimate']:,} |

### Format and rendering

- Records are stored in model-neutral Hugging Face / OpenAI chat format (`messages` plus optional `tools`). Never store or reuse one model's token IDs.
- Assistant turns with visible thinking store it in `reasoning_content`; tool calls store `function.arguments` as parsed JSON objects so `tokenizer.apply_chat_template(record["messages"], tools=record.get("tools"), tokenize=True)` renders natively across target families.
- `calibration` and `heldout` splits are strictly disjoint by `group`, `source_identity`, and `content_sha256`, and exclude all Standard Fidelity panel sources.
- Full per-record provenance, generating models, local dump/commit pins, privacy/secret-scan status, and license notes are recorded in [`long_context/manifest.json`](long_context/manifest.json).

### Byte-identical v2 provenance (`agentic` & `exl3_standard`)

All 319 inherited files from `{HF_V2_REPO}@{HF_V2_REVISION}` retain their exact bytes and SHA-256 digests (key files below; full inventory in [`SHA256SUMS`](SHA256SUMS)):

| Inherited v2 file | SHA-256 |
| --- | --- |
{v2_hash_rows}
"""
    if "## Component 3: `long_context`" not in out:
        marker = "## Manifests and integrity"
        if marker in out:
            out = out.replace(marker, section.lstrip("\n") + "\n" + marker, 1)
        else:
            out = out.rstrip("\n") + "\n\n" + section.lstrip("\n")
    return out


def cmd_assemble(args: argparse.Namespace) -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq

    work_root = Path(args.work_root).expanduser().resolve()
    raw_dir = work_root / "raw_traces"
    stage_dir = work_root / "staged_dataset"
    v2_base = Path(args.v2_snapshot).expanduser().resolve()

    records: list[dict[str, Any]] = []
    for spec in CASES:
        p = raw_dir / f"{spec.id}.json"
        if not p.is_file():
            raise FileNotFoundError(f"Missing generated trace for {spec.id}: {p}")
        records.append(json.loads(p.read_text(encoding="utf-8")))

    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    shutil.copytree(v2_base, stage_dir, symlinks=False, ignore=shutil.ignore_patterns(".cache", "__pycache__"))

    lc_dir = stage_dir / "long_context"
    lc_dir.mkdir(parents=True, exist_ok=True)
    data_dir = stage_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    by_class: dict[str, list[dict[str, Any]]] = {"16k": [], "32k": []}
    for r in records:
        by_class[r["length_class"]].append(r)

    subsets_manifest: dict[str, Any] = {}
    for cls in ("16k", "32k"):
        cls_records = by_class[cls]
        jsonl_path = lc_dir / f"{cls}.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as f:
            for r in cls_records:
                f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")

        split_summaries: dict[str, Any] = {}
        for split in ("calibration", "heldout"):
            srecs = [r for r in cls_records if r["split"] == split]
            pq_path = data_dir / f"long_context_{cls}_{split}.parquet"
            pq.write_table(pa.Table.from_pylist([_parquet_row(r) for r in srecs]), pq_path)
            split_summaries[split] = {
                "records": len(srecs),
                "token_count_estimate": sum(r["token_count_estimate"] for r in srecs),
                "groups": [r["group"] for r in srecs],
                "parquet": f"data/long_context_{cls}_{split}.parquet",
                "parquet_sha256": sha256_file(pq_path),
            }

        subsets_manifest[cls] = {
            "file": f"long_context/{cls}.jsonl",
            "sha256": sha256_file(jsonl_path),
            "records": len(cls_records),
            "total_token_count_estimate": sum(r["token_count_estimate"] for r in cls_records),
            "splits": split_summaries,
        }

    cal_groups = sorted({r["group"] for r in records if r["split"] == "calibration"})
    hold_groups = sorted({r["group"] for r in records if r["split"] == "heldout"})
    fidelity_paths, fidelity_markers = load_fidelity_exclusions()
    fidelity_hits = 0
    for r in records:
        raw_str = json.dumps(r, ensure_ascii=False)
        if any(fp in raw_str for fp in fidelity_paths) or any(mk in raw_str for mk in fidelity_markers):
            fidelity_hits += 1
    if fidelity_hits:
        raise RuntimeError(f"Refusing assembly: {fidelity_hits} record(s) matched Standard Fidelity exclusions")

    lc_manifest = {
        "schema": "agentic-quant-panel-v3-long-context-v1",
        "component": "long_context",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "rendering": (
            "model-neutral chat messages (+ optional tools); render with each target model's "
            "own tokenizer and chat_template at calibration time; never reuse token IDs"
        ),
        "token_count_note": (
            "token_count_estimate is the median token count across local reference tokenizers "
            "(DeepSeek-V4.1-Flash, GLM-5.3, Qwen3.8-27B) in neutral chat serialization; exact "
            "rendered counts vary by target tokenizer and chat template."
        ),
        "splits": ["calibration", "heldout"],
        "split_disjointness": {
            "calibration_groups": cal_groups,
            "heldout_groups": hold_groups,
            "group_overlap": sorted(set(cal_groups) & set(hold_groups)),
            "content_hash_overlap": sorted(
                {r["content_sha256"] for r in records if r["split"] == "calibration"}
                & {r["content_sha256"] for r in records if r["split"] == "heldout"}
            ),
            "standard_fidelity_excluded": True,
            "standard_fidelity_exclusion": {
                "checked": True,
                "excluded_source_paths_count": len(fidelity_paths),
                "excluded_marker_count": len(fidelity_markers),
                "matched_records": fidelity_hits,
                "method": (
                    "Scanned all serialized records against all recorded Standard Fidelity "
                    "reference source paths and fixture markers; verified 0 matches."
                ),
            },
        },
        "summary": {
            "total_records": len(records),
            "calibration_records": sum(1 for r in records if r["split"] == "calibration"),
            "heldout_records": sum(1 for r in records if r["split"] == "heldout"),
            "total_token_count_estimate": sum(r["token_count_estimate"] for r in records),
            "calibration_token_count_estimate": sum(
                r["token_count_estimate"] for r in records if r["split"] == "calibration"
            ),
            "heldout_token_count_estimate": sum(
                r["token_count_estimate"] for r in records if r["split"] == "heldout"
            ),
            "models_used": sorted({r["generating_model"] for r in records}),
            "languages": sorted({lang for r in records for lang in r["language"]}),
            "records_with_reasoning": sum(1 for r in records if r["has_reasoning"]),
            "records_without_reasoning": sum(1 for r in records if not r["has_reasoning"]),
        },
        "subsets": subsets_manifest,
        "records": [
            {
                "id": r["id"],
                "length_class": r["length_class"],
                "split": r["split"],
                "group": r["group"],
                "source": r["source"],
                "archetype": r["archetype"],
                "generating_model": r["generating_model"],
                "task": r["task"],
                "language": r["language"],
                "has_reasoning": r["has_reasoning"],
                "reasoning_chars": r["reasoning_chars"],
                "tool_call_count": r["tool_call_count"],
                "message_count": r["message_count"],
                "token_count_estimate": r["token_count_estimate"],
                "token_counts_by_reference_tokenizer": r["token_counts_by_reference_tokenizer"],
                "content_sha256": r["content_sha256"],
                "source_identity": r["source_identity"],
                "license_notes": r["license_notes"],
                "provenance": r["provenance"],
            }
            for r in records
        ],
    }
    lc_manifest_path = lc_dir / "manifest.json"
    lc_manifest_path.write_text(
        json.dumps(lc_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    repro_dir = stage_dir / "reproduction"
    repro_dir.mkdir(exist_ok=True)
    for fname in ("long_context_specs.py", "long_context_harness.py", "long_context_bank.py"):
        src_text = Path(__file__).with_name(fname).read_text(encoding="utf-8")
        sanitized = sanitize_reproduction_script(fname, src_text)
        (repro_dir / fname).write_text(sanitized, encoding="utf-8")

    v2_sums = dict(
        line.split(maxsplit=1)[::-1]
        for line in (v2_base / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    )
    v2_sums_sha256 = sha256_file(v2_base / "SHA256SUMS")

    panel_path = stage_dir / "panel.json"
    panel = json.loads(panel_path.read_text(encoding="utf-8"))
    panel["schema"] = "agentic-quant-panel-v3"
    panel["repo_id"] = HF_DATASET_REPO
    panel["base_v2"] = {
        "repo": HF_V2_REPO,
        "revision": HF_V2_REVISION,
        "sha256sums_sha256": v2_sums_sha256,
        "byte_identical_components": ["agentic", "exl3_standard"],
        "file_sha256": {
            name: v2_sums[name]
            for name in (
                "calibration.jsonl",
                "screening.jsonl",
                "heldout.jsonl",
                "manifest.json",
                "data/calibration.parquet",
                "data/screening.parquet",
                "data/heldout.parquet",
                "data/exl3_standard.parquet",
                "exl3_standard/manifest.json",
                "exl3_standard/segments.jsonl",
                "exl3_standard/upstream/tokenizer.py",
                "reproduction/pack_exl3_standard.py",
            )
            if name in v2_sums
        },
    }
    comps = [c for c in panel.get("components", []) if c.get("name") != "long_context"]
    for c in comps:
        if c.get("name") == "agentic":
            c["inherited_from_v2"] = {
                "repo": HF_V2_REPO,
                "revision": HF_V2_REVISION,
                "byte_identical": True,
                "file_sha256": {f: v2_sums[f] for f in c.get("files", []) if f in v2_sums},
            }
        elif c.get("name") == "exl3_standard":
            exl3_files = sorted(k for k in v2_sums if k.startswith("exl3_standard/") or k == "data/exl3_standard.parquet" or k == "reproduction/pack_exl3_standard.py")
            c["inherited_from_v2"] = {
                "repo": HF_V2_REPO,
                "revision": HF_V2_REVISION,
                "byte_identical": True,
                "file_sha256": {f: v2_sums[f] for f in exl3_files},
            }
    comps.append(
        {
            "name": "long_context",
            "optional": True,
            "manifest": "long_context/manifest.json",
            "files": [
                "long_context/16k.jsonl",
                "long_context/32k.jsonl",
                "long_context/manifest.json",
                "data/long_context_16k_calibration.parquet",
                "data/long_context_16k_heldout.parquet",
                "data/long_context_32k_calibration.parquet",
                "data/long_context_32k_heldout.parquet",
            ],
            "subsets": ["16k", "32k"],
            "rendering": "model-neutral chat messages (+ optional tools); render with target model's chat template",
            "splits": ["calibration", "heldout"],
        }
    )
    panel["components"] = comps
    panel_path.write_text(json.dumps(panel, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    readme_path = stage_dir / "README.md"
    readme_path.write_text(
        _updated_readme(readme_path.read_text(encoding="utf-8"), lc_manifest, v2_sums, v2_sums_sha256),
        encoding="utf-8",
    )

    gitattr = stage_dir / ".gitattributes"
    if not gitattr.is_file():
        gitattr.write_text("*.parquet filter=lfs diff=lfs merge=lfs -text\n", encoding="utf-8")

    all_files = sorted(p for p in stage_dir.rglob("*") if p.is_file() and p.name != "SHA256SUMS")
    (stage_dir / "SHA256SUMS").write_text(
        "".join(f"{sha256_file(p)}  {p.relative_to(stage_dir).as_posix()}\n" for p in all_files),
        encoding="utf-8",
    )

    print(f"Assembled staged dataset at {stage_dir} ({len(all_files)} files).")
    print(json.dumps(lc_manifest["summary"], indent=2))
    return 0


def validate_staged_dataset(stage_dir: Path, v2_base: Path) -> dict[str, Any]:
    """Run exhaustive integrity, privacy/secret, quality, overlap, split, and chat-template checks."""
    from transformers import AutoTokenizer

    errors: list[str] = []
    live_secrets = _load_live_secret_literals()
    fidelity_paths, fidelity_markers = load_fidelity_exclusions()

    # 1. Verify all original v1/v2 data files remain byte-for-byte identical
    v2_sums = dict(
        line.split(maxsplit=1)[::-1]
        for line in (v2_base / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    )
    stage_sums = dict(
        line.split(maxsplit=1)[::-1]
        for line in (stage_dir / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    )
    mutable_top = {"README.md", "panel.json", "SHA256SUMS"}
    for rel, expected_sha in v2_sums.items():
        if rel in mutable_top:
            continue
        p = stage_dir / rel
        if not p.is_file():
            errors.append(f"Missing preserved v2 file: {rel}")
        elif sha256_file(p) != expected_sha:
            errors.append(f"Preserved v2 file hash changed: {rel}")

    for rel, expected_sha in stage_sums.items():
        p = stage_dir / rel
        if not p.is_file() or sha256_file(p) != expected_sha:
            errors.append(f"SHA256SUMS mismatch: {rel}")

    # 2. Load v2 calibration/heldout groups for cross-component disjointness verification
    v2_manifest = json.loads((v2_base / "manifest.json").read_text(encoding="utf-8"))
    v2_cal_groups = {r["group"] for r in v2_manifest["splits"]["calibration"]["records"]}
    v2_hold_groups = {r["group"] for r in v2_manifest["splits"]["heldout"]["records"]}

    # 3. Validate long_context records & manifest
    lc_manifest = json.loads((stage_dir / "long_context/manifest.json").read_text(encoding="utf-8"))
    all_records: list[dict[str, Any]] = []
    for cls in ("16k", "32k"):
        jl = stage_dir / f"long_context/{cls}.jsonl"
        for line in jl.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r["length_class"] != cls:
                errors.append(f"{r['id']}: length_class {r['length_class']} != {cls}")
            all_records.append(r)

    seen_ids: set[str] = set()
    cal_groups: set[str] = set()
    hold_groups: set[str] = set()
    cal_hashes: set[str] = set()
    hold_hashes: set[str] = set()
    cal_sids: set[str] = set()
    hold_sids: set[str] = set()

    privacy_scan_counts: dict[str, int] = {}

    for r in all_records:
        rid = r["id"]
        if rid in seen_ids:
            errors.append(f"Duplicate id: {rid}")
        seen_ids.add(rid)

        for forbidden_key in ("input_ids", "token_ids", "tokens"):
            if forbidden_key in r:
                errors.append(f"{rid}: stores forbidden token-ID field '{forbidden_key}'")

        payload_hash = hashlib.sha256(
            canonical_json_bytes({"messages": r["messages"], "tools": r.get("tools")})
        ).hexdigest()
        if payload_hash != r["content_sha256"]:
            errors.append(f"{rid}: content_sha256 mismatch")

        # Privacy + secret scan across entire serialized record
        raw_str = json.dumps(r, ensure_ascii=False)
        leaks = scan_text_for_privacy_and_secrets(raw_str, live_secrets)
        privacy_scan_counts[rid] = len(leaks)
        if leaks:
            errors.append(f"{rid}: PRIVACY/SECRET LEAK DETECTED: {leaks}")

        # Quality acceptance gate
        spec = CASES_BY_ID.get(rid)
        if spec is not None:
            q_defects = check_trace_quality(r, spec)
            for qd in q_defects:
                errors.append(f"{rid}: QUALITY GATE FAILED: {qd}")

        # Standard Fidelity exclusion check
        for fp in fidelity_paths:
            if fp in raw_str:
                errors.append(f"{rid}: references Standard Fidelity source path '{fp}'")
        for mk in fidelity_markers:
            if mk in raw_str:
                errors.append(f"{rid}: contains Standard Fidelity marker '{mk}'")

        # Message structure & tool-call pairing check
        msgs = r["messages"]
        if not msgs or msgs[0]["role"] not in ("system", "user"):
            errors.append(f"{rid}: invalid first message role")
        pending_tc_ids: set[str] = set()
        for m in msgs:
            role = m["role"]
            if role == "assistant":
                if pending_tc_ids:
                    errors.append(f"{rid}: unanswered tool_calls before next assistant turn: {pending_tc_ids}")
                for tc in m.get("tool_calls") or []:
                    fn = tc.get("function") or {}
                    if not isinstance(fn.get("arguments"), dict):
                        errors.append(f"{rid}: tool_call arguments not dict for {tc.get('id')}")
                    pending_tc_ids.add(tc["id"])
            elif role == "tool":
                pending_tc_ids.discard(m.get("tool_call_id", ""))
        if pending_tc_ids:
            errors.append(f"{rid}: ends with unanswered tool_calls: {pending_tc_ids}")

        # Split tracking
        if r["split"] == "calibration":
            cal_groups.add(r["group"])
            cal_hashes.add(r["content_sha256"])
            cal_sids.add(r["source_identity"])
            if r["group"] in v2_hold_groups:
                errors.append(f"{rid}: calibration group {r['group']} overlaps v2 heldout!")
        elif r["split"] == "heldout":
            hold_groups.add(r["group"])
            hold_hashes.add(r["content_sha256"])
            hold_sids.add(r["source_identity"])
            if r["group"] in v2_cal_groups:
                errors.append(f"{rid}: heldout group {r['group']} overlaps v2 calibration!")
        else:
            errors.append(f"{rid}: invalid split {r['split']}")

    if cal_groups & hold_groups:
        errors.append(f"Split group overlap: {cal_groups & hold_groups}")
    if cal_hashes & hold_hashes:
        errors.append(f"Split content_sha256 overlap: {cal_hashes & hold_hashes}")
    if cal_sids & hold_sids:
        errors.append(f"Split source_identity overlap: {cal_sids & hold_sids}")

    # 4. Scan published reproduction scripts, README.md, panel.json, and long_context/manifest.json
    #    (root manifest.json is pre-existing v1 and unchanged).
    published_file_privacy_hits: dict[str, list[str]] = {}
    for rel in (
        "reproduction/long_context_specs.py",
        "reproduction/long_context_harness.py",
        "reproduction/long_context_bank.py",
        "long_context/manifest.json",
        "README.md",
        "panel.json",
    ):
        p = stage_dir / rel
        if not p.is_file():
            errors.append(f"Missing published file for privacy scan: {rel}")
            continue
        text = p.read_text(encoding="utf-8")
        hits = scan_text_for_privacy_and_secrets(text, live_secrets)
        # README.md and panel.json inherit the public HF repo namespace wrldsuksgo2mars/agentic-quant-panel-v1/v2
        # from the existing v2 dataset card; check strictly for everything else (and zero hits on reproduction/ & long_context/).
        if rel in ("README.md", "panel.json"):
            hits = [h for h in hits if h != "privacy:private_handle:wrldsuksgo2mars"]
        published_file_privacy_hits[rel] = hits
        if hits:
            errors.append(f"Published file {rel} failed privacy/secret scan: {hits}")

    # 5. Verify chat template rendering across local target models
    template_results: dict[str, dict[str, int]] = {}
    for tname, snap in CHAT_TEMPLATE_VERIFY_TOKENIZERS.items():
        if not Path(snap).is_dir():
            continue
        tok = AutoTokenizer.from_pretrained(snap, local_files_only=True)
        template_results[tname] = {}
        for r in all_records:
            try:
                ids = tok.apply_chat_template(
                    r["messages"],
                    tools=r.get("tools"),
                    tokenize=True,
                    add_generation_prompt=False,
                )
                if isinstance(ids, dict):
                    ids = ids.get("input_ids", [])
                elif hasattr(ids, "input_ids"):
                    ids = ids.input_ids
                template_results[tname][r["id"]] = len(ids)
            except Exception as exc:
                errors.append(f"Template render failed for {r['id']} on {tname}: {type(exc).__name__}: {exc}")

    return {
        "ok": len(errors) == 0,
        "errors": errors,
        "record_count": len(all_records),
        "preserved_v2_files_checked": len(v2_sums) - len(mutable_top),
        "privacy_secret_hits_per_record": privacy_scan_counts,
        "published_file_privacy_hits": published_file_privacy_hits,
        "template_rendered_tokens": template_results,
        "manifest_summary": lc_manifest["summary"],
    }


def cmd_validate(args: argparse.Namespace) -> int:
    work_root = Path(args.work_root).expanduser().resolve()
    stage_dir = work_root / "staged_dataset"
    v2_base = Path(args.v2_snapshot).expanduser().resolve()
    report = validate_staged_dataset(stage_dir, v2_base)
    out = work_root / "validation_report.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


def _git_blob_id(path: Path) -> str:
    raw = path.read_bytes()
    return hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()


def verify_published_v3_hub_tree(
    *,
    api: Any,
    v2_repo: str,
    v2_revision: str,
    v3_repo: str,
    v3_revision: str,
    stage_dir: Path,
) -> dict[str, Any]:
    """Verify the newly published v3 dataset repo against BOTH v2@54683731eb and local staging:
    - v2 head is still untouched at v2_revision
    - Every inherited agentic/exl3_standard/v1/provenance/license file in v3 has the exact
      same size and LFS sha256 (or Git blob_id) as in v2@v2_revision AND local staging
    - Every new/updated v3 file (long_context/*, data/long_context_*, reproduction/long_context_*,
      README.md, panel.json, SHA256SUMS) matches local staging's size and sha256/blob_id.
    """
    from huggingface_hub.hf_api import RepoFile

    def _tree(repo: str, rev: str) -> dict[str, dict[str, Any]]:
        return {
            e.path: {
                "size": e.size,
                "lfs_sha256": e.lfs.sha256 if e.lfs else None,
                "blob_id": e.blob_id,
            }
            for e in api.list_repo_tree(repo, revision=rev, recursive=True, repo_type="dataset")
            if isinstance(e, RepoFile)
        }

    v2_info = api.repo_info(v2_repo, repo_type="dataset")
    if v2_info.sha != v2_revision:
        raise RuntimeError(f"v2 head changed unexpectedly: {v2_info.sha} != {v2_revision}")

    v2_tree = _tree(v2_repo, v2_revision)
    v3_tree = _tree(v3_repo, v3_revision)
    mutable_top = {"README.md", "panel.json", "SHA256SUMS", ".gitattributes"}

    inherited_checked = 0
    for rel, v2_entry in v2_tree.items():
        if rel in mutable_top:
            continue
        v3_entry = v3_tree.get(rel)
        if v3_entry != v2_entry:
            raise RuntimeError(f"v3 inherited file differs from v2: {rel} ({v3_entry} != {v2_entry})")
        inherited_checked += 1

    stage_files = {
        p.relative_to(stage_dir).as_posix(): p
        for p in stage_dir.rglob("*")
        if p.is_file()
    }
    missing = set(stage_files) - set(v3_tree)
    extra = set(v3_tree) - set(stage_files) - {".gitattributes"}
    if missing or extra:
        raise RuntimeError(f"v3 file inventory mismatch: missing={sorted(missing)} extra={sorted(extra)}")

    verified_all = 0
    verified_new: dict[str, dict[str, Any]] = {}
    for rel, local in sorted(stage_files.items()):
        if rel == ".gitattributes":
            continue
        cur = v3_tree[rel]
        local_size = local.stat().st_size
        local_sha256 = sha256_file(local)
        local_git_blob = _git_blob_id(local)
        if cur["size"] != local_size:
            raise RuntimeError(f"Size mismatch on Hub for {rel}: {cur['size']} != {local_size}")
        if cur["lfs_sha256"] is not None:
            if cur["lfs_sha256"] != local_sha256:
                raise RuntimeError(f"LFS sha256 mismatch for {rel}: {cur['lfs_sha256']} != {local_sha256}")
            kind = "lfs"
        else:
            if cur["blob_id"] != local_git_blob:
                raise RuntimeError(f"Git blob_id mismatch for {rel}: {cur['blob_id']} != {local_git_blob}")
            kind = "git"
        verified_all += 1
        if rel not in v2_tree or rel in mutable_top:
            verified_new[rel] = {
                "storage": kind,
                "bytes": local_size,
                "sha256": local_sha256,
                "hub_oid": cur["lfs_sha256"] or cur["blob_id"],
            }

    return {
        "ok": True,
        "v2_repo": v2_repo,
        "v2_revision_untouched": v2_info.sha,
        "v3_repo": v3_repo,
        "v3_revision": v3_revision,
        "inherited_v2_files_verified": inherited_checked,
        "total_staged_files_verified": verified_all,
        "new_and_updated_files": verified_new,
    }


def cmd_publish(args: argparse.Namespace) -> int:
    if not args.confirm_upload:
        raise SystemExit(
            "Refusing to publish without --confirm-upload (coordinator/TJ confirmation required)."
        )
    if os.environ.get("HF_HUB_DISABLE_XET") != "1":
        raise SystemExit("Refusing to publish unless HF_HUB_DISABLE_XET=1 is set in environment.")

    import huggingface_hub
    from huggingface_hub import CommitOperationAdd, HfApi

    if huggingface_hub.__version__ != "1.33.0":
        raise SystemExit(f"Require huggingface-hub==1.33.0, got {huggingface_hub.__version__}")

    work_root = Path(args.work_root).expanduser().resolve()
    stage_dir = work_root / "staged_dataset"
    v2_base = Path(args.v2_snapshot).expanduser().resolve()

    report = validate_staged_dataset(stage_dir, v2_base)
    if not report["ok"]:
        raise SystemExit(f"Validation failed; refusing upload: {report['errors']}")

    # Never touch v2; create v3 with the exact same visibility as v2.
    if args.repo_id == HF_V2_REPO:
        raise SystemExit(f"Refusing to overwrite {HF_V2_REPO}; publish to a new v3 dataset repo.")

    api = HfApi()
    v2_info = api.repo_info(HF_V2_REPO, repo_type="dataset")
    if v2_info.sha != HF_V2_REVISION:
        raise SystemExit(f"Expected {HF_V2_REPO} at {HF_V2_REVISION}, found {v2_info.sha}")

    api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=bool(v2_info.private),
        exist_ok=True,
    )

    # Upload all staged files in bounded HTTP LFS batches (<=160 files/commit, .gitattributes first).
    all_files = sorted(
        p.relative_to(stage_dir).as_posix()
        for p in stage_dir.rglob("*")
        if p.is_file()
    )
    ordered = ([".gitattributes"] if ".gitattributes" in all_files else []) + [
        f for f in all_files if f != ".gitattributes"
    ]
    batch_size = 160
    info = api.repo_info(args.repo_id, repo_type="dataset")
    head = info.sha
    for idx in range(0, len(ordered), batch_size):
        batch = ordered[idx : idx + batch_size]
        ops = [
            CommitOperationAdd(path_in_repo=rel, path_or_fileobj=str(stage_dir / rel))
            for rel in batch
        ]
        batch_num = idx // batch_size + 1
        total_batches = (len(ordered) + batch_size - 1) // batch_size
        commit = api.create_commit(
            repo_id=args.repo_id,
            repo_type="dataset",
            operations=ops,
            parent_commit=head,
            commit_message=(
                f"Publish agentic-quant-panel-v3 (batch {batch_num}/{total_batches}: "
                "agentic + exl3_standard byte-identical to v2@54683731eb + long_context 16K/32K)"
            ),
        )
        head = commit.oid

    verify = verify_published_v3_hub_tree(
        api=api,
        v2_repo=HF_V2_REPO,
        v2_revision=HF_V2_REVISION,
        v3_repo=args.repo_id,
        v3_revision=head,
        stage_dir=stage_dir,
    )
    receipt_path = work_root / "publish_receipt.json"
    receipt_path.write_text(json.dumps(verify, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(verify, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT)
    p.add_argument("--v2-snapshot", type=Path, default=V2_PANEL_SNAPSHOT)
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="Generate raw traces for specified or all cases")
    g.add_argument("--cases", nargs="*", default=[])
    g.add_argument("--workers", type=int, default=4)
    g.add_argument("--retries", type=int, default=2)
    g.add_argument("--force", action="store_true")
    g.set_defaults(func=cmd_generate)

    a = sub.add_parser("assemble", help="Assemble JSONL, Parquet, manifest, and staged dataset")
    a.set_defaults(func=cmd_assemble)

    v = sub.add_parser("validate", help="Validate staged dataset integrity, privacy, secrets, splits, and templates")
    v.set_defaults(func=cmd_validate)

    pub = sub.add_parser("publish", help="Upload verified staged dataset to HF Hub over HTTP LFS")
    pub.add_argument("--repo-id", default=HF_DATASET_REPO)
    pub.add_argument("--confirm-upload", action="store_true")
    pub.set_defaults(func=cmd_publish)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
