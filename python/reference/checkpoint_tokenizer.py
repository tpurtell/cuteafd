"""Resolve checkpoint-native tokenizers without converting their vocabulary or chat format."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys


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


class _ErnieSegmentTokenizer:
    """Wrap checkpoint tokenizer.json with ERNIE's source-aware segment encoding."""

    def __init__(self, snapshot: Path, render_chat, chat_identity: dict):
        from tokenizers import Tokenizer
        raw = json.loads((snapshot / "tokenizer.json").read_text())
        added = raw.get("added_tokens", [])
        if any(t.get(k) for t in added for k in ("single_word", "lstrip", "rstrip", "normalized")):
            raise ValueError("ERNIE segment encoding requires literal added tokens")
        self._mapping = {t["content"]: t["id"] for t in added}
        self._reserved = frozenset(self._mapping.values())
        self._pattern = (
            re.compile("|".join(re.escape(s) for s in sorted(self._mapping, key=len, reverse=True)))
            if self._mapping else None
        )
        raw["added_tokens"] = []
        raw["padding"] = None
        raw["truncation"] = None
        self._ordinary = Tokenizer.from_str(json.dumps(raw))
        self._full = Tokenizer.from_file(str(snapshot / "tokenizer.json"))
        self._fidelity_render_chat = render_chat
        self._fidelity_chat_identity = chat_identity

    def encode_segments(self, segments) -> list[int]:
        ids: list[int] = []
        buf: list[str] = []

        def flush():
            if not buf:
                return
            run_ids = self._ordinary.encode("".join(buf), add_special_tokens=False).ids
            if not self._reserved.isdisjoint(run_ids):
                raise ValueError("ordinary BPE path produced an added-token ID")
            ids.extend(run_ids)
            buf.clear()

        for segment in segments:
            text = segment.text
            if segment.allow_special and self._pattern is not None:
                cursor = 0
                for match in self._pattern.finditer(text):
                    if match.start() > cursor:
                        buf.append(text[cursor:match.start()])
                    flush()
                    ids.append(self._mapping[match.group()])
                    cursor = match.end()
                text = text[cursor:]
            if text:
                buf.append(text)
        flush()
        return ids

    def encode(self, text: str, add_special_tokens: bool = False):
        return self._full.encode(text, add_special_tokens=add_special_tokens)

    def decode(self, ids, skip_special_tokens: bool = False, clean_up_tokenization_spaces: bool = False):
        return self._full.decode(ids, skip_special_tokens=skip_special_tokens)


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
        config = json.loads((snapshot / "config.json").read_text())
        if config.get("model_type") == "ernie5_5_moe":
            encoder = snapshot / "encoding_ernie.py"
            if not encoder.is_file():
                raise ValueError("ERNIE 5.5 checkpoint requires its official encoding_ernie.py")
            encoder_hash = _digest(encoder)
            mod_name = "fidelity_checkpoint_ernie_" + encoder_hash
            spec = importlib.util.spec_from_file_location(mod_name, encoder)
            module = importlib.util.module_from_spec(spec)
            sys.modules[mod_name] = module
            spec.loader.exec_module(module)
            if _digest(encoder) != encoder_hash:
                raise ValueError("checkpoint chat encoder changed while loading")
            chat_identity = {
                "chat_template_kind": "checkpoint_encoder",
                "chat_template_sha256": encoder_hash,
                "chat_encoder_file": "encoding_ernie.py"}
            return _ErnieSegmentTokenizer(snapshot, module.render_chat, chat_identity)
        from transformers import PreTrainedTokenizerFast
        tokenizer = PreTrainedTokenizerFast.from_pretrained(snapshot, local_files_only=True)
        if config.get("model_type") == "deepseek_v41":
            encoder = snapshot / "encoding/encoding.py"
            if not encoder.is_file():
                raise ValueError("DeepSeek checkpoint requires its official encoding/encoding.py")
            encoder_hash = _digest(encoder)
            spec = importlib.util.spec_from_file_location("fidelity_checkpoint_encoding_" + encoder_hash, encoder)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            if _digest(encoder) != encoder_hash:
                raise ValueError("checkpoint chat encoder changed while loading")
            tokenizer._fidelity_encode_messages = module.encode_messages
            tokenizer._fidelity_parse_completion = getattr(module, "parse_message_from_completion_text", None)
            tokenizer._fidelity_chat_identity = {
                "chat_template_kind": "checkpoint_encoder",
                "chat_template_sha256": encoder_hash,
                "chat_encoder_file": "encoding/encoding.py"}
        return tokenizer
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(snapshot / "tokenizer.json"))
