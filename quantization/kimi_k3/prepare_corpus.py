#!/usr/bin/env python3
"""Attest the public agentic panel with Kimi's own segmented chat encoder.

No weights are loaded. Each published prompt becomes one unpadded batch, in
published order. Safetensors stores flat int64 IDs and record boundaries so
the quantizer never has to tokenize or render the prompt a second time.
"""
from __future__ import annotations

import argparse
from array import array
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import urllib.request

REPO_ID = "wrldsuksgo2mars/agentic-quant-panel-v1"
REVISION = "875ff1d3786bf4346ee7ff52ea5e3f1f698f5902"
SCHEMA = "cuteafd-kimi-k3-corpus-v1"
SPLITS = ("calibration", "screening", "heldout")
PAYLOAD_SHA256 = {
    "SHA256SUMS": "6bcb96ce1e6c20f5a1b16509292f14e0b56b4d575fe51ec27bfe8bc9a1b51ba2",
    "manifest.json": "e769a9b519f8bf82a4341f1e528fb25d5285d4f9229d46fdd67d5a57bf9615c4",
    "publication.json": "afac5e1273e4ccb08bfaf4027bbab58e51a2c92a45762e9c43921d360fb046d1",
    "statistics.json": "4f85426ecb69b04dc84e7661f669c08534052f7fe71985a979902d8b7a3fc601",
    "README.md": "95543c01fbaad55465c6e34e9036c9e10619870e020dd431e5a4036624e04dc7",
    "calibration.jsonl": "4e569625d97865777da92167b8fbf6fabb4ab7adf55baa5d676c81ce8dd95244",
    "screening.jsonl": "04afb4bf9e23d5e1788822c78b9363b8bc6459575b35ee70675ddaf7bf6e2ff0",
    "heldout.jsonl": "8491342fa0b0d4c6dc4a15969ab79a214a5c67cf82f0d81156b6a2164ad9e673",
}
SOURCE_FILES = ("config.json", "tokenizer_config.json", "tiktoken.model",
                "tokenization_kimi.py", "encoding_k3.py")
CHAT_KWARGS = dict(tokenize=True, add_generation_prompt=True, thinking=True,
                   thinking_effort="max", padding=False, truncation=False)
CONTRACT = dict(renderer="checkpoint-segmented-chat-v1", thinking=True,
                thinking_effort="max", add_generation_prompt=True,
                batch_size=1, padding=False, truncation=False, sorting=False,
                packing=False, attention_mask="all-ones")
DEFAULT_CORPUS = Path.home() / ".cache/cuteafd/quantization/kimi-k3-exl3-k2/corpus"


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def sha256_file(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def ids_sha256(ids):
    return hashlib.sha256(canonical(list(ids))).hexdigest()


def download_corpus(root=DEFAULT_CORPUS):
    """Download only the small pinned panel payloads; reject altered caches."""
    root = Path(root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    for name, expected in PAYLOAD_SHA256.items():
        target = root / name
        if target.exists():
            if sha256_file(target) != expected:
                raise ValueError(f"cached dataset hash differs: {target}")
            continue
        url = f"https://huggingface.co/datasets/{REPO_ID}/resolve/{REVISION}/{name}"
        fd, partial = tempfile.mkstemp(prefix=name + ".", dir=root)
        try:
            with os.fdopen(fd, "wb") as output, urllib.request.urlopen(url, timeout=120) as response:
                shutil.copyfileobj(response, output)
                output.flush()
                os.fsync(output.fileno())
            if sha256_file(partial) != expected:
                raise ValueError(f"downloaded dataset hash differs: {name}")
            os.replace(partial, target)
        finally:
            Path(partial).unlink(missing_ok=True)
    return root


def read_panel(root):
    """Validate original bytes, ordered row provenance and split separation."""
    root = Path(root)
    for name, expected in PAYLOAD_SHA256.items():
        if sha256_file(root / name) != expected:
            raise ValueError(f"dataset hash differs: {name}")
    provenance = json.loads((root / "manifest.json").read_text())
    checksums = dict(line.split(maxsplit=1)[::-1]
                     for line in (root / "SHA256SUMS").read_text().splitlines())
    rows = {}
    for split in SPLITS:
        name = split + ".jsonl"
        if checksums[name] != PAYLOAD_SHA256[name]:
            raise ValueError(f"dataset checksum declaration differs: {split}")
        expected = provenance["splits"][split]
        rows[split] = [json.loads(line) for line in (root / name).read_text().splitlines()]
        validate_rows(rows[split], expected)
    calibration = provenance["splits"]["calibration"]["records"]
    heldout = provenance["splits"]["heldout"]["records"]
    screening = provenance["splits"]["screening"]["records"]
    if ({r["group"] for r in calibration} & {r["group"] for r in heldout}
            or {r["prompt_sha256"] for r in calibration} & {r["prompt_sha256"] for r in heldout}
            or not {r["source_identity"] for r in screening} <= {r["source_identity"] for r in calibration}):
        raise ValueError("dataset split separation differs")
    return rows, provenance


def validate_rows(rows, provenance):
    if len(rows) != provenance["summary"]["records"] or len(rows) != len(provenance["records"]):
        raise ValueError("dataset row count differs")
    seen = set()
    for row, original in zip(rows, provenance["records"], strict=True):
        if (not isinstance(row.get("id"), str) or row["id"] in seen
                or not isinstance(row.get("prompt"), str)
                or row["id"] != original["id"]
                or hashlib.sha256(row["prompt"].encode()).hexdigest() != original["prompt_sha256"]):
            raise ValueError("dataset row identity or prompt hash differs")
        seen.add(row["id"])


def tokenize_rows(rows, tokenizer, *, chat_kwargs=None):
    """Use the checkpoint encoder directly, preserving user/control separation."""
    chat_kwargs = CHAT_KWARGS if chat_kwargs is None else chat_kwargs
    flat, offsets, records = array("q"), [0], []
    stream = hashlib.sha256()
    vocabulary_size = len(tokenizer)
    for row in rows:
        messages = [dict(role="user", content=row["prompt"])]
        ids = tokenizer.apply_chat_template(messages, **chat_kwargs)
        if (not isinstance(ids, list) or not ids
                or any(type(token) is not int or not 0 <= token < vocabulary_size for token in ids)):
            raise ValueError(f"invalid tokenizer output: {row['id']}")
        # Text is a diagnostic only. Re-encoding this string would erase the
        # renderer's special-token permissions for user text and attributes.
        rendered = tokenizer.apply_chat_template(messages, **{**chat_kwargs, "tokenize": False})
        if not isinstance(rendered, str):
            raise ValueError("tokenizer did not render a string")
        flat.extend(ids)
        offsets.append(len(flat))
        entry = dict(id=row["id"], tokens=len(ids),
                     prompt_sha256=hashlib.sha256(row["prompt"].encode()).hexdigest(),
                     rendered_sha256=hashlib.sha256(rendered.encode()).hexdigest(),
                     input_ids_sha256=ids_sha256(ids))
        stream.update(canonical(dict(id=row["id"], input_ids=ids)) + b"\n")
        records.append(entry)
    if not records:
        raise ValueError("empty tokenized split")
    return flat, offsets, dict(records=records, count=len(records), tokens=len(flat),
                              shortest=min(r["tokens"] for r in records),
                              longest=max(r["tokens"] for r in records),
                              token_stream_sha256=stream.hexdigest())


def software_identity():
    versions = {}
    for name in ("transformers", "tiktoken", "tokenizers", "torch", "safetensors", "tokenicer"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return dict(packages=versions, python=sys.version)


def tokenizer_probes(tokenizer, *, chat_kwargs=None):
    """Retain exact boundary witnesses alongside the hashed full panel stream."""
    chat_kwargs = CHAT_KWARGS if chat_kwargs is None else chat_kwargs
    result = []
    for text in ("Hello, world!", "你好。日本語 한국어 العربية", "literal <|open|> and [BOS]",
                 "🙂é\u0301\u0000", "x = 1\n    return x\n"):
        messages = [dict(role="user", content=text)]
        ids = tokenizer.apply_chat_template(messages, **chat_kwargs)
        rendered = tokenizer.apply_chat_template(messages, **{**chat_kwargs, "tokenize": False})
        result.append(dict(raw_text=text, rendered_text=rendered, input_ids=ids,
                           attention_mask=[1] * len(ids)))
    return result


def prepare_corpus(source_snapshot, output, corpus=DEFAULT_CORPUS):
    import torch
    from safetensors.torch import save_file
    from transformers import AutoTokenizer

    source_snapshot, output = Path(source_snapshot).expanduser().resolve(), Path(output).expanduser().resolve()
    if output.exists():
        raise ValueError(f"output exists; use a new directory: {output}")
    rows, provenance = read_panel(download_corpus(corpus))
    source_identity = {name: sha256_file(source_snapshot / name) for name in SOURCE_FILES}
    if json.loads((source_snapshot / "config.json").read_text()).get("model_type") != "kimi_k3":
        raise ValueError("source snapshot is not Kimi K3")
    tokenizer = AutoTokenizer.from_pretrained(source_snapshot, trust_remote_code=True, local_files_only=True)
    if type(tokenizer).__name__ != "TikTokenTokenizer":
        raise ValueError("source did not load its native Kimi tokenizer")
    renderer = tokenizer.apply_chat_template.__func__.__globals__["build_chat_segments"]
    if (sha256_file(inspect.getfile(type(tokenizer))) != source_identity["tokenization_kimi.py"]
            or sha256_file(inspect.getfile(renderer)) != source_identity["encoding_k3.py"]):
        raise ValueError("loaded tokenizer or renderer differs from source")
    manifest = dict(schema=SCHEMA, status="passed", contract=CONTRACT,
                    dataset=dict(repo_id=REPO_ID, revision=REVISION, files=PAYLOAD_SHA256),
                    historical_selection=provenance["selection"],
                    source=dict(snapshot=str(source_snapshot), revision=source_snapshot.name,
                                files=source_identity),
                    software=software_identity(), tokenizer=dict(class_name=type(tokenizer).__name__,
                    vocab_size=len(tokenizer), special_tokens_map=tokenizer.special_tokens_map),
                    probes=tokenizer_probes(tokenizer), splits={})
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=output.name + ".partial-", dir=output.parent))
    try:
        for split in SPLITS:
            flat, offsets, report = tokenize_rows(rows[split], tokenizer)
            path = stage / (split + ".safetensors")
            tensors = dict(input_ids=torch.frombuffer(flat, dtype=torch.int64).clone(),
                           offsets=torch.tensor(offsets, dtype=torch.int64))
            save_file(tensors, str(path), metadata=dict(schema=SCHEMA, split=split))
            report.update(file=path.name, sha256=sha256_file(path))
            manifest["splits"][split] = report
            with path.open("rb") as source:
                os.fsync(source.fileno())
        # The encoder source must not change while its token arrays are built.
        if source_identity != {name: sha256_file(source_snapshot / name) for name in SOURCE_FILES}:
            raise ValueError("source tokenizer changed during preparation")
        with (stage / "manifest.json").open("wb") as target:
            target.write(canonical(manifest) + b"\n")
            target.flush()
            os.fsync(target.fileno())
        for split in SPLITS:
            load_token_records(stage, split)  # independent saved-array validation
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


def load_token_records(root, split="calibration", *, schema=SCHEMA, contract=CONTRACT):
    """Verify a prepared split and return ordered IDs without re-tokenizing."""
    import torch
    from safetensors.torch import load_file

    if split not in SPLITS:
        raise ValueError(f"unknown split: {split}")
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if (manifest.get("schema") != schema or manifest.get("status") != "passed"
            or manifest.get("contract") != contract
            or manifest.get("dataset") != dict(repo_id=REPO_ID, revision=REVISION, files=PAYLOAD_SHA256)):
        raise ValueError("prepared corpus contract differs")
    report = manifest["splits"][split]
    if report.get("file") != split + ".safetensors":
        raise ValueError("prepared split file differs")
    path = root / report["file"]
    if sha256_file(path) != report["sha256"]:
        raise ValueError("prepared token file hash differs")
    tensors = load_file(str(path), device="cpu")
    if (set(tensors) != {"input_ids", "offsets"}
            or any(t.dtype != torch.int64 or t.ndim != 1 for t in tensors.values())):
        raise ValueError("invalid token tensor layout")
    flat, offsets = tensors["input_ids"], tensors["offsets"].tolist()
    records = report["records"]
    if (not records or len(offsets) != len(records) + 1 or offsets[0] != 0
            or offsets[-1] != flat.numel() or any(a >= b for a, b in zip(offsets, offsets[1:]))
            or len({row["id"] for row in records}) != len(records)):
        raise ValueError("invalid token offsets or record identities")
    stream, result = hashlib.sha256(), []
    vocabulary_size = manifest["tokenizer"]["vocab_size"]
    for row, start, stop in zip(records, offsets, offsets[1:], strict=False):
        ids = flat[start:stop].tolist()
        if (len(ids) != row["tokens"] or ids_sha256(ids) != row["input_ids_sha256"]
                or any(not 0 <= token < vocabulary_size for token in ids)):
            raise ValueError("prepared record token identity differs")
        stream.update(canonical(dict(id=row["id"], input_ids=ids)) + b"\n")
        result.append(dict(id=row["id"], input_ids=ids))
    if (report["count"] != len(records) or report["tokens"] != flat.numel()
            or report["shortest"] != min(row["tokens"] for row in records)
            or report["longest"] != max(row["tokens"] for row in records)
            or report["token_stream_sha256"] != stream.hexdigest()):
        raise ValueError("prepared split token summary differs")
    return result


def load_token_batches(root, split="calibration", *, schema=SCHEMA, contract=CONTRACT):
    """Yield GPTQModel calibration dictionaries, one unpadded CPU batch each."""
    import torch
    for row in load_token_records(root, split, schema=schema, contract=contract):
        ids = torch.tensor([row["input_ids"]], dtype=torch.int64)
        yield dict(input_ids=ids, attention_mask=torch.ones_like(ids))


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
