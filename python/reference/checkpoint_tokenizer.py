"""Resolve checkpoint-native tokenizers without converting their vocabulary or chat format."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import re


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _native_files(snapshot):
    config = json.loads((snapshot / "tokenizer_config.json").read_text())
    mapping = config.get("auto_map", {}).get("AutoTokenizer")
    entry = mapping[0] if isinstance(mapping, (list, tuple)) else mapping
    if not isinstance(entry, str) or not re.fullmatch(r"[A-Za-z_]\w*\.[A-Za-z_]\w*", entry):
        raise ValueError("native tokenizer requires a local AutoTokenizer source mapping")
    pending, sources = [entry.split(".")[0] + ".py"], set()
    while pending:
        name = pending.pop()
        if name in sources:
            continue
        path = snapshot / name
        if not path.is_file():
            raise ValueError("native tokenizer source closure is incomplete")
        sources.add(name)
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.level:
                if node.level != 1 or not node.module or "." in node.module:
                    raise ValueError("unsupported native tokenizer relative import")
                pending.append(node.module + ".py")
    # Include all root-level tokenizer resources, including optional special-token
    # maps. Source imports are closed above; no weight shards or model code load.
    resources = {p.name for p in snapshot.iterdir() if p.is_file() and (
        p.name in ("tokenizer_config.json", "special_tokens_map.json", "added_tokens.json", "chat_template.jinja")
        or p.suffix in (".model", ".tiktoken", ".vocab", ".merges")
        or p.name in ("vocab.json", "vocab.txt", "merges.txt"))}
    if not resources - {"tokenizer_config.json"}:
        raise ValueError("native tokenizer has no local vocabulary resources")
    return {name: _digest(snapshot / name) for name in sorted(sources | resources)}


def tokenizer_identity(snapshot):
    snapshot = Path(snapshot)
    fast = snapshot / "tokenizer.json"
    if fast.is_file():
        return {"tokenizer_sha256": _digest(fast)}
    files = _native_files(snapshot)
    payload = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {"tokenizer_sha256": hashlib.sha256(payload).hexdigest(),
            "tokenizer_kind": "checkpoint_python", "tokenizer_files": files}


def load_tokenizer(path, *, chat=False):
    """Trust only the named local snapshot; bind Python and vocabulary bytes together."""
    path = Path(path)
    snapshot = path if path.is_dir() else path.parent
    identity = tokenizer_identity(snapshot)
    if identity.get("tokenizer_kind") == "checkpoint_python":
        if not path.is_dir() and path.name not in identity["tokenizer_files"]:
            raise ValueError("tokenizer path is not a pinned checkpoint resource")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(snapshot, trust_remote_code=True,
                                                  local_files_only=True, use_fast=False)
        if tokenizer_identity(snapshot) != identity:
            raise ValueError("native tokenizer files changed while loading")
        tokenizer._fidelity_identity = identity
        return tokenizer
    if not path.is_dir() and path.read_bytes() != (snapshot / "tokenizer.json").read_bytes():
        raise ValueError("tokenizer path differs from checkpoint tokenizer.json")
    if chat:
        from transformers import PreTrainedTokenizerFast
        return PreTrainedTokenizerFast.from_pretrained(snapshot, local_files_only=True)
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(snapshot / "tokenizer.json"))
