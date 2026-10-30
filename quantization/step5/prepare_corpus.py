#!/usr/bin/env python3
"""Attest the official panel using Step-5's native thinking chat template.

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
SCHEMA = "cuteafd-step5-corpus-v1"
SOURCE_REPO = "stepfun-ai/Step-5-Preview-BF16"
SOURCE_REVISION = "7734b7ffac118d70d8398bfa3eb0fab007eb45b1"
SOURCE_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json",
                "chat_template.jinja", "special_tokens_map.json")
CHAT_KWARGS = dict(tokenize=True, add_generation_prompt=True,
                   reasoning_effort="max", padding=False, truncation=False,
                   return_dict=False)
CONTRACT = dict(renderer="checkpoint-chat-template-v1", thinking=True,
                reasoning_effort="max", add_generation_prompt=True,
                batch_size=1, padding=False, truncation=False, sorting=False,
                packing=False, attention_mask="all-ones")
DEFAULT_CORPUS = (Path("/mnt/sparknest/hf-home/hub") /
                  "datasets--wrldsuksgo2mars--agentic-quant-panel-v1" / "snapshots" / REVISION)

# Keep the same payload, array and attestation implementations in both families.
canonical, sha256_file = shared.canonical, shared.sha256_file
read_panel = shared.read_panel


def download_corpus(root=DEFAULT_CORPUS):
    """Use only already managed SparkNest panel bytes; never a direct HF fetch."""
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError("panel is not available; use nest hf download " + REPO_ID +
                         " --dataset --revision " + REVISION + " --host maxq --wait")
    read_panel(root)
    return root


def tokenize_rows(rows, tokenizer):
    return shared.tokenize_rows(rows, tokenizer, chat_kwargs=CHAT_KWARGS)


def load_token_records(root, split="calibration"):
    return shared.load_token_records(root, split, schema=SCHEMA, contract=CONTRACT)


def load_token_batches(root, split="calibration"):
    yield from shared.load_token_batches(root, split, schema=SCHEMA, contract=CONTRACT)


def tokenizer_probes(tokenizer):
    """Attest exact native text/IDs and the declared thinking boundary."""
    probes = shared.tokenizer_probes(tokenizer, chat_kwargs=CHAT_KWARGS)
    for row in probes:
        text = row["rendered_text"]
        if (not text.startswith(tokenizer.bos_token + "<|im_start|>system\nReasoning: max\n\n<|im_end|>\n")
                or not text.endswith("<|im_start|>assistant\n<think>")):
            raise ValueError("Step-5 native thinking/max-effort boundary differs")
        if row["input_ids"] != tokenizer.encode(text, add_special_tokens=False):
            raise ValueError("Step-5 native template tokenization differs from its rendered text")
    return probes


def verify_native_tokenizer(tokenizer, source_snapshot):
    """Reject heuristic regex/model substitutions against serialized checkpoint bytes.

    Some Transformers versions print a generic Mistral-regex warning for this
    non-Mistral tokenizer. Native fidelity means retaining Step-5's own regex,
    not applying that unrelated model family's suggested patch.
    """
    serialized = json.loads((Path(source_snapshot) / "tokenizer.json").read_text())
    loaded = json.loads(tokenizer.backend_tokenizer.to_str())
    fields = ("model", "normalizer", "pre_tokenizer", "decoder")
    if any(loaded.get(key) != serialized.get(key) for key in fields):
        raise ValueError("loaded Step-5 tokenizer differs from native serialized model/regex/decoder")
    return dict(serialized_components_equal=list(fields), heuristic_regex_patch=False)


def verify_native_encodings(tokenizer, source_snapshot, probes):
    """Compare ordinary text and full rendered prompts to the raw Rust backend."""
    from tokenizers import Tokenizer
    raw = Tokenizer.from_file(str(Path(source_snapshot) / "tokenizer.json"))
    count = 0
    for probe in probes:
        for text in (probe["raw_text"], probe["rendered_text"]):
            if tokenizer.encode(text, add_special_tokens=False) != raw.encode(text, add_special_tokens=False).ids:
                raise ValueError("Step-5 native encoding differs from raw checkpoint tokenizer.json")
            count += 1
    return dict(raw_checkpoint_backend_equal=True, raw_checkpoint_encoding_probes=count)


def prepare_corpus(source_snapshot, output, corpus=DEFAULT_CORPUS):
    import torch
    from safetensors.torch import save_file
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python/reference"))
    from checkpoint_tokenizer import load_tokenizer

    source_snapshot = Path(source_snapshot).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise ValueError(f"output exists; use a new directory: {output}")
    config = json.loads((source_snapshot / "config.json").read_text())
    if (source_snapshot.name != SOURCE_REVISION or config.get("model_type") != "step5_vl"
            or config.get("text_config", {}).get("num_hidden_layers") != 92
            or config.get("quantization_config") or config.get("text_config", {}).get("quantization_config")):
        raise ValueError("source snapshot is not the pinned Step-5 Preview BF16 release")
    rows, provenance = read_panel(download_corpus(corpus))
    source_identity = {name: sha256_file(source_snapshot / name) for name in SOURCE_FILES}
    # The shared fidelity loader retains the serialized ByteLevel BPE backend.
    # Current AutoTokenizer maps the source's historical LlamaTokenizerFast
    # label to a new SentencePiece-oriented implementation, changing its regex.
    tokenizer = load_tokenizer(source_snapshot, chat=True)
    if (not isinstance(tokenizer.chat_template, str)
            or hashlib.sha256(tokenizer.chat_template.encode()).hexdigest() != source_identity["chat_template.jinja"]):
        raise ValueError("loaded Step-5 chat template differs from checkpoint bytes")
    tokenizer_identity = verify_native_tokenizer(tokenizer, source_snapshot)
    probes = tokenizer_probes(tokenizer)
    tokenizer_identity.update(verify_native_encodings(tokenizer, source_snapshot, probes))
    manifest = dict(schema=SCHEMA, status="passed", contract=CONTRACT,
        dataset=dict(repo_id=REPO_ID, revision=REVISION, files=PAYLOAD_SHA256),
        historical_selection=provenance["selection"],
        source=dict(repo_id=SOURCE_REPO, snapshot=str(source_snapshot),
                    revision=SOURCE_REVISION, files=source_identity),
        software=shared.software_identity(),
        tokenizer=dict(class_name=type(tokenizer).__name__,
            native_class_name=type(tokenizer).__name__, native_template_encoding_equal=True,
            **tokenizer_identity,
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
