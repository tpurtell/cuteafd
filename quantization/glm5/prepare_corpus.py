#!/usr/bin/env python3
"""Attest the official panel using GLM-5.3's native thinking chat template.

The flat int64 IDs, prompt boundaries, row hashes and replay validation are
shared with the Kimi preparation code. Only the model's rendering contract
and source files differ. No weights are loaded.
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
SCHEMA = "cuteafd-glm5-corpus-v1"
SOURCE_REPO = "zai-org/GLM-5.3-BF16"
SOURCE_REVISION = "9d2398f478cab2de883137db3a36ad2c96205e24"
SOURCE_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json",
                "chat_template.jinja", "generation_config.json")
CHAT_KWARGS = dict(tokenize=True, add_generation_prompt=True,
                   reasoning_effort="max", padding=False, truncation=False,
                   return_dict=False)
CONTRACT = dict(renderer="checkpoint-chat-template-v1", thinking=True,
                reasoning_effort="max", add_generation_prompt=True,
                batch_size=1, padding=False, truncation=False, sorting=False,
                packing=False, attention_mask="all-ones")
DEFAULT_CORPUS = shared.DEFAULT_CORPUS

# Keep the same payload, array and attestation implementations in both families.
canonical, sha256_file = shared.canonical, shared.sha256_file
read_panel, download_corpus = shared.read_panel, shared.download_corpus


def tokenize_rows(rows, tokenizer):
    return shared.tokenize_rows(rows, tokenizer, chat_kwargs=CHAT_KWARGS)


def load_token_records(root, split="calibration"):
    return shared.load_token_records(root, split, schema=SCHEMA, contract=CONTRACT)


def load_token_batches(root, split="calibration"):
    yield from shared.load_token_batches(root, split, schema=SCHEMA, contract=CONTRACT)


def tokenizer_probes(tokenizer, direct):
    """Compare exact native text/IDs, including control-like user content."""
    probes = shared.tokenizer_probes(tokenizer, chat_kwargs=CHAT_KWARGS)
    native = shared.tokenizer_probes(direct, chat_kwargs=CHAT_KWARGS)
    if probes != native:
        raise ValueError("Tokenicer differs from the native checkpoint tokenizer")
    for row in probes:
        if (not row["rendered_text"].startswith("[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>")
                or not row["rendered_text"].endswith("<|assistant|><think>")):
            raise ValueError("GLM native thinking/max-effort boundary differs")
        if row["input_ids"] != direct.encode(row["rendered_text"], add_special_tokens=False):
            raise ValueError("GLM native template tokenization differs from its rendered text")
    return probes


def prepare_corpus(source_snapshot, output, corpus=DEFAULT_CORPUS):
    import torch
    from safetensors.torch import save_file
    from tokenicer import Tokenicer
    from transformers import AutoConfig, AutoTokenizer

    source_snapshot = Path(source_snapshot).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise ValueError(f"output exists; use a new directory: {output}")
    config = json.loads((source_snapshot / "config.json").read_text())
    if (source_snapshot.name != SOURCE_REVISION or config.get("model_type") != "glm_moe_dsa"
            or config.get("num_hidden_layers") != 78 or config.get("quantization_config")):
        raise ValueError("source snapshot is not the pinned GLM-5.3 BF16 release")
    rows, provenance = read_panel(download_corpus(corpus))
    source_identity = {name: sha256_file(source_snapshot / name) for name in SOURCE_FILES}
    direct = AutoTokenizer.from_pretrained(source_snapshot, local_files_only=True)
    tokenizer = Tokenicer.load(str(source_snapshot), local_files_only=True,
        model_config=AutoConfig.from_pretrained(source_snapshot, local_files_only=True))
    if (not isinstance(direct.chat_template, str)
            or hashlib.sha256(direct.chat_template.encode()).hexdigest() != source_identity["chat_template.jinja"]):
        raise ValueError("loaded GLM chat template differs from checkpoint bytes")
    probes = tokenizer_probes(tokenizer, direct)
    manifest = dict(schema=SCHEMA, status="passed", contract=CONTRACT,
        dataset=dict(repo_id=REPO_ID, revision=REVISION, files=PAYLOAD_SHA256),
        historical_selection=provenance["selection"],
        source=dict(repo_id=SOURCE_REPO, snapshot=str(source_snapshot),
                    revision=SOURCE_REVISION, files=source_identity),
        software=shared.software_identity(),
        tokenizer=dict(class_name=type(tokenizer).__name__,
            native_class_name=type(direct).__name__, native_probes_equal=True,
            vocab_size=len(tokenizer), special_tokens_map=tokenizer.special_tokens_map),
        probes=probes, splits={})
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
