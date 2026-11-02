#!/usr/bin/env python3
"""Attest the pinned public panel with Mistral's native v15 Tekken encoder.

The checkpoint's Jinja template independently witnesses the rendered request.
Token IDs come from the official segmented encoder, rather than re-encoding
rendered text: literal control-like text in a user prompt must stay user text.
Only local SparkNest-managed inputs are read; this tool downloads nothing.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kimi_k3 import prepare_corpus as shared

REPO_ID, REVISION = shared.REPO_ID, shared.REVISION
PAYLOAD_SHA256, SPLITS = shared.PAYLOAD_SHA256, shared.SPLITS
SCHEMA = "cuteafd-mistral-large4-corpus-v1"
SOURCE_REPO = "mistralai/Mistral-Large-4-1T-A52B"
SOURCE_REVISION = "f12dabdc0c1e99e7f7924461a6f308859b05af4d"
SOURCE_FILES = ("config.json", "tekken.json", "tokenizer.json",
                "tokenizer_config.json", "chat_template.jinja",
                "generation_config.json")
CHAT_KWARGS = dict(tokenize=True, add_generation_prompt=True,
                   reasoning_effort="high", padding=False, truncation=False,
                   return_dict=False)
CONTRACT = dict(renderer="checkpoint-tekken-v15-native-segmented-chat-v1",
                thinking=True, reasoning_effort="high", add_generation_prompt=True,
                batch_size=1, padding=False, truncation=False, sorting=False,
                packing=False, attention_mask="all-ones")
DEFAULT_CORPUS = (Path("/mnt/sparknest/hf-home/hub") /
    "datasets--wrldsuksgo2mars--agentic-quant-panel-v1/snapshots" / REVISION)
canonical, sha256_file = shared.canonical, shared.sha256_file
read_panel = shared.read_panel


def download_corpus(root=DEFAULT_CORPUS):
    """Compatibility name: require the already managed, pinned local payload."""
    root = Path(root).expanduser().resolve()
    for name, expected in PAYLOAD_SHA256.items():
        if not (root / name).is_file() or sha256_file(root / name) != expected:
            raise ValueError(f"missing or altered pinned panel: {root / name}; obtain via SparkNest")
    return root


def validate_source(source_snapshot):
    root = Path(source_snapshot).expanduser().resolve()
    config = json.loads((root / "config.json").read_text())
    if (root.name != SOURCE_REVISION or config.get("model_type") != "mistral_large4"
            or config.get("text_config", {}).get("num_hidden_layers") != 72
            or config.get("text_config", {}).get("vocab_size") != 204800
            or config.get("quantization_config", {}).get("quant_method") != "fp8"):
        raise ValueError("source is not the pinned Mistral Large 4 FP8 release")
    return root


class NativeTokenizer:
    """Small adapter preserving the shared corpus API and native segmentation."""

    def __init__(self, source_snapshot):
        from jinja2.sandbox import ImmutableSandboxedEnvironment
        from mistral_common.protocol.instruct.messages import UserMessage
        from mistral_common.protocol.instruct.request import ChatCompletionRequest
        from mistral_common.tokens.tokenizers.base import SpecialTokenPolicy
        from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
        from tokenizers import Tokenizer

        root = Path(source_snapshot)
        self.native = MistralTokenizer.from_file(root / "tekken.json")
        self.raw = Tokenizer.from_file(str(root / "tokenizer.json"))
        self.UserMessage, self.ChatCompletionRequest = UserMessage, ChatCompletionRequest
        self.SpecialTokenPolicy = SpecialTokenPolicy
        if str(self.native.version) != "v15" or len(self) != 204800:
            raise ValueError("native tokenizer version/vocabulary differs from checkpoint")
        environment = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        environment.globals["raise_exception"] = self._raise
        environment.filters["tojson"] = lambda value, **kwargs: json.dumps(
            value, ensure_ascii=False, **kwargs)
        self.template = environment.from_string((root / "chat_template.jinja").read_text())
        self.special_tokens_map = dict(bos_token="<s>", eos_token="</s>",
                                       pad_token="<pad>", unk_token="<unk>")

    @staticmethod
    def _raise(message):
        raise ValueError(message)

    def __len__(self):
        return self.native.instruct_tokenizer.tokenizer.n_words

    def render(self, messages):
        return self.template.render(messages=messages, bos_token="<s>", eos_token="</s>",
                                    add_generation_prompt=True, reasoning_effort="high")

    def apply_chat_template(self, messages, **kwargs):
        expected = {**CHAT_KWARGS, "tokenize": kwargs.get("tokenize", True)}
        if kwargs != expected:
            raise ValueError("unreviewed Mistral request rendering options")
        if (not isinstance(messages, list) or len(messages) != 1
                or set(messages[0]) != {"role", "content"}
                or messages[0]["role"] != "user" or not isinstance(messages[0]["content"], str)):
            raise ValueError("calibration requires exactly one literal user message")
        native = self.native.encode_chat_completion(self.ChatCompletionRequest(
            messages=[self.UserMessage(content=messages[0]["content"])], reasoning_effort="high"))
        rendered = self.render(messages)
        if self.native.decode(native.tokens, self.SpecialTokenPolicy.KEEP) != rendered:
            raise ValueError("native Tekken rendering differs from checkpoint chat template")
        if not native.tokens or native.tokens[0] != 1 or native.tokens[-1] != 4:
            raise ValueError("native BOS/instruction-end boundaries differ")
        return native.tokens if kwargs["tokenize"] else rendered

    def software_sources(self):
        objects = dict(tokenizer=type(self.native),
                       instruct_tokenizer=type(self.native.instruct_tokenizer),
                       tekken=type(self.native.instruct_tokenizer.tokenizer),
                       request=self.ChatCompletionRequest, user_message=self.UserMessage)
        return {name: dict(path=inspect.getfile(obj), sha256=sha256_file(inspect.getfile(obj)))
                for name, obj in objects.items()}


def tokenize_rows(rows, tokenizer):
    return shared.tokenize_rows(rows, tokenizer, chat_kwargs=CHAT_KWARGS)


def tokenizer_probes(tokenizer):
    texts = ("Hello, world!", "你好。日本語 한국어 العربية",
             "literal [INST] [/INST] [MODEL_SETTINGS] <s>",
             "🙂é\u0301\u0000", "x = 1\n    return x\n")
    probes = []
    for text in texts:
        messages = [dict(role="user", content=text)]
        ids = tokenizer.apply_chat_template(messages, **CHAT_KWARGS)
        rendered = tokenizer.apply_chat_template(messages, **{**CHAT_KWARGS, "tokenize": False})
        raw = tokenizer.raw.encode(rendered, add_special_tokens=False).ids
        probes.append(dict(raw_text=text, rendered_text=rendered, input_ids=ids,
                           attention_mask=[1] * len(ids), rendered_text_input_ids=raw,
                           rendered_reencoding_equal=ids == raw))
    if (not all(probes[i]["rendered_reencoding_equal"] for i in (0, 1, 3, 4))
            or probes[2]["rendered_reencoding_equal"]):
        raise ValueError("native/HF tokenizer boundary witnesses differ")
    return probes


def load_token_records(root, split="calibration"):
    """Verify saved flat arrays and stream identities without importing CUDA/Torch."""
    import numpy as np
    from safetensors.numpy import load_file

    if split not in SPLITS:
        raise ValueError(f"unknown split: {split}")
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if (manifest.get("schema") != SCHEMA or manifest.get("status") != "passed"
            or manifest.get("contract") != CONTRACT
            or manifest.get("dataset") != dict(repo_id=REPO_ID, revision=REVISION, files=PAYLOAD_SHA256)
            or manifest.get("source", {}).get("revision") != SOURCE_REVISION
            or manifest.get("source", {}).get("repo_id") != SOURCE_REPO):
        raise ValueError("prepared Mistral corpus contract differs")
    report = manifest["splits"][split]
    path = root / (split + ".safetensors")
    if report.get("file") != path.name or sha256_file(path) != report["sha256"]:
        raise ValueError("prepared split filename/hash differs")
    tensors = load_file(str(path))
    if (set(tensors) != {"input_ids", "offsets"}
            or any(t.dtype != np.int64 or t.ndim != 1 for t in tensors.values())):
        raise ValueError("invalid token tensor layout")
    flat, offsets = tensors["input_ids"], tensors["offsets"].tolist()
    records = report["records"]
    if (not records or len(offsets) != len(records) + 1 or offsets[0] != 0
            or offsets[-1] != flat.size or any(a >= b for a, b in zip(offsets, offsets[1:]))
            or len({row["id"] for row in records}) != len(records)):
        raise ValueError("invalid token offsets or record identities")
    vocabulary_size = manifest["tokenizer"]["vocab_size"]
    stream, result = hashlib.sha256(), []
    for row, start, stop in zip(records, offsets, offsets[1:], strict=False):
        ids = flat[start:stop].tolist()
        if (len(ids) != row["tokens"] or shared.ids_sha256(ids) != row["input_ids_sha256"]
                or any(not 0 <= token < vocabulary_size for token in ids)):
            raise ValueError("prepared record token identity differs")
        stream.update(canonical(dict(id=row["id"], input_ids=ids)) + b"\n")
        result.append(dict(id=row["id"], input_ids=ids))
    if (report["count"] != len(records) or report["tokens"] != flat.size
            or report["shortest"] != min(row["tokens"] for row in records)
            or report["longest"] != max(row["tokens"] for row in records)
            or report["token_stream_sha256"] != stream.hexdigest()):
        raise ValueError("prepared split token summary differs")
    return result


def load_token_batches(root, split="calibration"):
    import torch
    for row in load_token_records(root, split):
        ids = torch.tensor([row["input_ids"]], dtype=torch.int64)
        yield dict(input_ids=ids, attention_mask=torch.ones_like(ids))


def pilot_records(root, split="calibration", *, max_records=8, max_tokens=8192,
                  max_length=2304):
    """Select original, whole prompts in panel order within a bounded token budget."""
    if max_records <= 0 or max_tokens <= 0 or max_length <= 0:
        raise ValueError("pilot budgets must be positive")
    result, total = [], 0
    for row in load_token_records(root, split):
        length = len(row["input_ids"])
        if length > max_length or total + length > max_tokens:
            continue
        result.append(row)
        total += length
        if len(result) >= max_records:
            break
    if not result:
        raise ValueError("no complete prompts fit the short pilot budget")
    return result


def prepare_corpus(source_snapshot, output, corpus=DEFAULT_CORPUS):
    import numpy as np
    from safetensors.numpy import save_file

    root = validate_source(source_snapshot)
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise ValueError(f"output exists; use a new directory: {output}")
    rows, provenance = read_panel(download_corpus(corpus))
    source_identity = {name: sha256_file(root / name) for name in SOURCE_FILES}
    tokenizer = NativeTokenizer(root)
    probes = tokenizer_probes(tokenizer)
    software = shared.software_identity()
    for name in ("mistral-common", "jinja2"):
        software["packages"][name] = importlib.metadata.version(name)
    software["native_encoder_sources"] = tokenizer.software_sources()
    manifest = dict(schema=SCHEMA, status="passed", contract=CONTRACT,
        dataset=dict(repo_id=REPO_ID, revision=REVISION, files=PAYLOAD_SHA256),
        historical_selection=provenance["selection"],
        source=dict(repo_id=SOURCE_REPO, snapshot=str(root), revision=SOURCE_REVISION,
                    files=source_identity), software=software,
        tokenizer=dict(class_name=type(tokenizer.native).__name__, version="v15",
            native_template_text_equal=True, literal_user_special_tokens_preserved=True,
            vocab_size=len(tokenizer), special_tokens_map=tokenizer.special_tokens_map),
        probes=probes, splits={})
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=output.name + ".partial-", dir=output.parent))
    try:
        for split in SPLITS:
            flat, offsets, report = tokenize_rows(rows[split], tokenizer)
            path = stage / (split + ".safetensors")
            save_file(dict(input_ids=np.frombuffer(flat, dtype=np.int64).copy(),
                           offsets=np.asarray(offsets, dtype=np.int64)), str(path),
                      metadata=dict(schema=SCHEMA, split=split))
            report.update(file=path.name, sha256=sha256_file(path))
            manifest["splits"][split] = report
            with path.open("rb") as source:
                os.fsync(source.fileno())
        if source_identity != {name: sha256_file(root / name) for name in SOURCE_FILES}:
            raise ValueError("source tokenizer changed during preparation")
        with (stage / "manifest.json").open("wb") as target:
            target.write(canonical(manifest) + b"\n")
            target.flush()
            os.fsync(target.fileno())
        for split in SPLITS:
            load_token_records(stage, split)
        descriptor = os.open(stage, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.rename(stage, output)
        descriptor = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    args = parser.parse_args()
    manifest = prepare_corpus(args.source_snapshot, args.output, args.corpus)
    print(json.dumps(dict(output=str(args.output), splits={name: {key: report[key]
        for key in ("count", "tokens", "shortest", "longest", "token_stream_sha256")}
        for name, report in manifest["splits"].items()}), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
