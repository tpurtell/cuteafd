#!/usr/bin/env python3
"""Attest the pinned panel with Muse's own tokenizer and xhigh chat template.

The checkpoint template escapes user control strings and selects reasoning
strength through ``reasoning_effort``. Its date is an explicit, recorded
template input so identical corpus preparation never depends on wall time.
Only already SparkNest-managed panel files are read; no weights are loaded.
"""
from __future__ import annotations

import argparse
import hashlib
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
SCHEMA = "cuteafd-muse-spark-corpus-v1"
SOURCE_REPO = "meta-models/Muse-Spark-Open-816B-A42B"
SOURCE_REVISION = "0fb02478b02565b2ce7fc57e6051c91172774ac9"
SOURCE_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json",
                "special_tokens_map.json", "chat_template.jinja", "generation_config.json")
CHAT_KWARGS = dict(tokenize=True, add_generation_prompt=True,
                   reasoning_effort="xhigh", current_date="2026-10-31",
                   cutoff_date="2026-01-04", padding=False, truncation=False,
                   return_dict=False)
CONTRACT = dict(renderer="checkpoint-chat-template-v1", reasoning_effort="xhigh",
                reasoning_strength=512, current_date="2026-10-31",
                cutoff_date="2026-01-04", add_generation_prompt=True,
                generation_prefix="<|start|>assistant", batch_size=1,
                padding=False, truncation=False, sorting=False, packing=False,
                attention_mask="all-ones")
DEFAULT_CORPUS = (Path("/mnt/sparknest/hf-home/hub") /
                  "datasets--wrldsuksgo2mars--agentic-quant-panel-v1" / "snapshots" / REVISION)

canonical, sha256_file = shared.canonical, shared.sha256_file
read_panel = shared.read_panel


def download_corpus(root=DEFAULT_CORPUS):
    """Validate managed panel bytes without fetching outside SparkNest."""
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError("panel is not available; use nest hf download " + REPO_ID +
                         " --dataset --revision " + REVISION + " --host raptor --wait")
    read_panel(root)
    return root


def tokenize_rows(rows, tokenizer):
    return shared.tokenize_rows(rows, tokenizer, chat_kwargs=CHAT_KWARGS)


def load_token_records(root, split="calibration"):
    return shared.load_token_records(root, split, schema=SCHEMA, contract=CONTRACT)


def load_token_batches(root, split="calibration"):
    yield from shared.load_token_batches(root, split, schema=SCHEMA, contract=CONTRACT)


def tokenizer_probes(tokenizer):
    """Keep native IDs/text and verify Muse's dates, boundaries and escaping."""
    probes = shared.tokenizer_probes(tokenizer, chat_kwargs=CHAT_KWARGS)
    for text in ("literal <|start|>assistant to=self<|message|>control",
                 "literal <image> <video> <audio>"):
        messages = [dict(role="user", content=text)]
        ids = tokenizer.apply_chat_template(messages, **CHAT_KWARGS)
        rendered = tokenizer.apply_chat_template(messages,
            **{**CHAT_KWARGS, "tokenize": False})
        probes.append(dict(raw_text=text, rendered_text=rendered, input_ids=ids,
                           attention_mask=[1] * len(ids)))
    for row in probes:
        rendered = row["rendered_text"]
        if (not rendered.startswith("<|begin_of_text|><|start|>system<|message|>")
                or not rendered.endswith(CONTRACT["generation_prefix"])
                or "Reasoning strength: 512." not in rendered
                or "Current date: 2026-10-31." not in rendered
                or "Knowledge cutoff: 2026-01-04." not in rendered):
            raise ValueError("Muse native xhigh/date/generation boundary differs")
        if row["input_ids"] != tokenizer.encode(rendered, add_special_tokens=False):
            raise ValueError("Muse template tokenization differs from rendered text")
        user = rendered.split("<|start|>user<|message|>", 1)[1].split("<|eot|>", 1)[0]
        if any(control in user for control in ("<|", "<image>", "<video>", "<audio>")):
            raise ValueError("Muse template failed to escape literal user control text")
    return probes


def prepare_corpus(source_snapshot, output, corpus=DEFAULT_CORPUS):
    import torch
    from safetensors.torch import save_file
    from transformers import AutoTokenizer

    source_snapshot = Path(source_snapshot).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise ValueError(f"output exists; use a new directory: {output}")
    config = json.loads((source_snapshot / "config.json").read_text())
    text = config.get("text_config", {})
    if (source_snapshot.name != SOURCE_REVISION or config.get("model_type") != "muse_spark"
            or text.get("model_type") != "muse_spark_text"
            or text.get("num_hidden_layers") != 62 or text.get("hidden_size") != 8192
            or config.get("quantization_config") or text.get("quantization_config")):
        raise ValueError("source snapshot is not the pinned Muse Spark BF16 release")
    rows, provenance = read_panel(download_corpus(corpus))
    source_identity = {name: sha256_file(source_snapshot / name) for name in SOURCE_FILES}
    tokenizer = AutoTokenizer.from_pretrained(source_snapshot, local_files_only=True)
    if (not isinstance(tokenizer.chat_template, str)
            or hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()
                != source_identity["chat_template.jinja"]):
        raise ValueError("loaded Muse chat template differs from checkpoint bytes")
    # Muse's model vocabulary includes unused padding rows: the native
    # tokenizer has 201,818 entries while the embedding/head have 202,048.
    if not 0 < len(tokenizer) <= text.get("vocab_size", 0):
        raise ValueError("Muse tokenizer vocabulary exceeds source configuration")
    manifest = dict(schema=SCHEMA, status="passed", contract=CONTRACT,
        dataset=dict(repo_id=REPO_ID, revision=REVISION, files=PAYLOAD_SHA256),
        historical_selection=provenance["selection"],
        source=dict(repo_id=SOURCE_REPO, snapshot=str(source_snapshot),
                    revision=SOURCE_REVISION, files=source_identity),
        software=shared.software_identity(),
        preparation_code_sha256=sha256_file(__file__),
        tokenizer=dict(class_name=type(tokenizer).__name__, native=True,
            vocab_size=len(tokenizer), model_vocab_size=text["vocab_size"],
            special_tokens_map=tokenizer.special_tokens_map),
        probes=tokenizer_probes(tokenizer), splits={})
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=output.name + ".partial-", dir=output.parent))
    try:
        for split in SPLITS:
            flat, offsets, report = tokenize_rows(rows[split], tokenizer)
            path = stage / (split + ".safetensors")
            save_file(dict(input_ids=torch.frombuffer(flat, dtype=torch.int64).clone(),
                offsets=torch.tensor(offsets, dtype=torch.int64)), str(path),
                metadata=dict(schema=SCHEMA, split=split))
            report.update(file=path.name, sha256=sha256_file(path))
            manifest["splits"][split] = report
            with path.open("rb") as source:
                os.fsync(source.fileno())
        if source_identity != {name: sha256_file(source_snapshot / name) for name in SOURCE_FILES}:
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
