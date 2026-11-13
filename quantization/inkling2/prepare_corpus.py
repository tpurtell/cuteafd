#!/usr/bin/env python3
"""Attest the standard own-text panel with Inkling's native fast tokenizer.

This is prompt preparation, not a model-loader normalization. It consumes the
checkpoint's declared tokenizer.json and chat template without corrective
patches; reusable loading corrections continue to belong in Tokenicer.
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
from inkling2.recipe import BASE_REPO, BASE_REVISION

REPO_ID, REVISION = shared.REPO_ID, shared.REVISION
PAYLOAD_SHA256, SPLITS = shared.PAYLOAD_SHA256, shared.SPLITS
SCHEMA = "cuteafd-inkling2-corpus-v1"
FROZEN_TEMPLATE_DATE = "2026-11-13"
CHAT_KWARGS = dict(tokenize=True, add_generation_prompt=True, reasoning_effort="max",
                   date_string=FROZEN_TEMPLATE_DATE, padding=False, truncation=False, return_dict=False)
CONTRACT = dict(renderer="checkpoint-chat-template-v1", reasoning_effort="max",
    template_date=FROZEN_TEMPLATE_DATE, add_generation_prompt=True, batch_size=1,
    padding=False, truncation=False, sorting=False, packing=False, attention_mask="all-ones")
DEFAULT_CORPUS = Path.home() / ".cache/cuteafd/quantization/inkling2-k4-mc/corpus"
SOURCE_FILES = {
    "config.json": "7da68011c7c85af5e040114d3ebd5c8fdc8ce9030851a5706ebe88c379632405",
    "tokenizer_config.json": "6addc2bf8dc756dea8edd083a4ea3b67a66b8f513e9745a36e4a7cc0cc9f55ed",
    "tokenizer.json": "7f332727f9a3f093dd452a39cc978e6757627e5dad1b0673aa4e75174f236dd6",
    "chat_template.jinja": "74eb94918c5410bfab8d0af83ce1204fa8af08531f742c18ff4758b92b602d37",
    "tiktoken/tokenizer.model": "f170372dd655728909228220d77cf6b8104a59f38a91262a7d2c4027906ec0d8",
}
canonical, sha256_file = shared.canonical, shared.sha256_file
download_corpus, read_panel = shared.download_corpus, shared.read_panel


def tokenize_rows(rows, tokenizer):
    return shared.tokenize_rows(rows, tokenizer, chat_kwargs=CHAT_KWARGS)


def tokenizer_probes(tokenizer):
    """Attest exact native render/IDs, including control-like user content."""
    probes = shared.tokenizer_probes(tokenizer, chat_kwargs=CHAT_KWARGS)
    for row in probes:
        rendered = row["rendered_text"]
        effort = "<|message_system|><|content_text|>Thinking effort level: 0.99<|end_message|>"
        if not rendered.startswith(effort) or "<|message_user|><|content_text|>" not in rendered:
            raise ValueError("Inkling native thinking/user boundary differs")
        if row["input_ids"] != tokenizer.encode(rendered, add_special_tokens=False):
            raise ValueError("native template tokenization differs from rendered-text encoding")
    return probes


def long_context_smoke(tokenizer):
    """Create a deterministic heldout-only witness across the SWA cutoff."""
    sentence = ("A telescope records the colour and position of each moving star. "
                "The observer preserves every earlier observation before checking the next one. ")
    for repetitions in (16, 32, 64, 128):
        prompt = ("Read this observation log and explain how the final observation relates to the first.\n" +
                  "\n".join(f"Observation {n}: {sentence}" for n in range(repetitions)))
        flat, _, report = tokenize_rows([dict(id="inkling2-swa-cutoff-smoke", prompt=prompt)], tokenizer)
        if len(flat) > 1024:
            return dict(input_ids=list(flat), raw_text=prompt,
                        hessian_eligible=False, purpose="heldout sliding-window cutoff witness",
                        **report["records"][0])
    raise ValueError("long-context smoke did not cross the 1024-token window")


def load_token_records(root, split="calibration"):
    # The smoke is intentionally not a split and cannot enter Hessian loaders.
    return shared.load_token_records(root, split, schema=SCHEMA, contract=CONTRACT)


def load_token_batches(root, split="calibration"):
    yield from shared.load_token_batches(root, split, schema=SCHEMA, contract=CONTRACT)


def load_long_context_smoke(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema") != SCHEMA or manifest.get("contract") != CONTRACT:
        raise ValueError("prepared corpus contract differs")
    report = manifest["long_context_smoke"]
    if report.get("file") != "long-context-smoke.json" or report.get("hessian_eligible") is not False:
        raise ValueError("long-context witness is not heldout-only")
    path = root / report["file"]
    if sha256_file(path) != report["sha256"]:
        raise ValueError("long-context witness hash differs")
    row = json.loads(path.read_text())
    ids = row.get("input_ids", [])
    if (row.get("hessian_eligible") is not False or len(ids) <= 1024
            or len(ids) != row.get("tokens") or shared.ids_sha256(ids) != row.get("input_ids_sha256")
            or any(type(n) is not int or not 0 <= n < manifest["tokenizer"]["vocab_size"] for n in ids)):
        raise ValueError("long-context witness token identity differs")
    return row


def prepare_corpus(source_snapshot, output, corpus=DEFAULT_CORPUS):
    import torch
    from safetensors.torch import save_file
    from transformers import PreTrainedTokenizerFast

    source_snapshot = Path(source_snapshot).expanduser().absolute()
    output = Path(output).expanduser().absolute()
    if output.exists():
        raise ValueError("output exists; use a new directory: " + str(output))
    if source_snapshot.name != BASE_REVISION:
        raise ValueError("source is not the pinned Inkling snapshot")
    source_identity = {name: sha256_file(source_snapshot / name) for name in SOURCE_FILES}
    if source_identity != SOURCE_FILES:
        raise ValueError("source tokenizer/config assets differ from reviewed Inkling release")
    config = json.loads((source_snapshot / "config.json").read_text())
    tokenizer_config = json.loads((source_snapshot / "tokenizer_config.json").read_text())
    if config.get("model_type") != "inkling2" or tokenizer_config.get("tokenizer_class") != "PreTrainedTokenizerFast":
        raise ValueError("source tokenizer class or model family differs")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(source_snapshot, local_files_only=True)
    if (not tokenizer.is_fast or not isinstance(tokenizer.chat_template, str)
            or hashlib.sha256(tokenizer.chat_template.encode()).hexdigest() != source_identity["chat_template.jinja"]):
        raise ValueError("loaded native tokenizer/template differs from source")
    probes = tokenizer_probes(tokenizer)
    smoke = long_context_smoke(tokenizer)
    rows, provenance = read_panel(download_corpus(corpus))
    manifest = dict(schema=SCHEMA, status="passed", contract=CONTRACT,
        dataset=dict(repo_id=REPO_ID, revision=REVISION, files=PAYLOAD_SHA256),
        historical_selection=provenance["selection"],
        source=dict(repo_id=BASE_REPO, revision=BASE_REVISION, snapshot=str(source_snapshot), files=source_identity),
        software=shared.software_identity(),
        tokenizer=dict(class_name=type(tokenizer).__name__, source_class="PreTrainedTokenizerFast",
            vocab_size=len(tokenizer), special_tokens_map=tokenizer.special_tokens_map,
            tiktoken_asset="tiktoken/tokenizer.model"), probes=probes, splits={})
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
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
        smoke_path = stage / "long-context-smoke.json"
        smoke_path.write_bytes(canonical(smoke) + b"\n")
        with smoke_path.open("rb") as stream:
            os.fsync(stream.fileno())
        manifest["long_context_smoke"] = dict(file=smoke_path.name, sha256=sha256_file(smoke_path),
                                              tokens=smoke["tokens"], hessian_eligible=False)
        if source_identity != {name: sha256_file(source_snapshot / name) for name in SOURCE_FILES}:
            raise ValueError("source tokenizer changed during preparation")
        with (stage / "manifest.json").open("wb") as stream:
            stream.write(canonical(manifest) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        for split in SPLITS:
            load_token_records(stage, split)
        load_long_context_smoke(stage)
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
    parser.add_argument("--source-snapshot", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    args = parser.parse_args()
    manifest = prepare_corpus(args.source_snapshot, args.output, args.corpus)
    print(json.dumps(dict(output=str(args.output), splits={name: {k: report[k]
        for k in ("count", "tokens", "shortest", "longest", "token_stream_sha256")}
        for name, report in manifest["splits"].items()}, long_context_smoke=manifest["long_context_smoke"])))


if __name__ == "__main__":
    main()
