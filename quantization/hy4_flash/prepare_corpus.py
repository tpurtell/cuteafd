"""Prepare the pinned agentic and raw ExLlamaV3 panel with the native Hy4 tokenizer."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from kimi_k3 import prepare_corpus as panel
from kimi_k3.uniform_k2.storage import atomic_json, checkpoint_io, fsync_directory

SOURCE_REVISION = "d88c9dfd96ba4f2e6b92adba0b8e8f1c8696f959"
SCHEMA = "cuteafd-hy4-flash-corpus-v2"
PANEL_REVISION = "54683731eb01c7c2fa5b801a0888e029a0e1b1c7"
PANEL_SUMS = "23f07f7dbe7b8248a09891b6ae2475371f4653e499a9b8f930e267ff3d5c6d57"
DATASET = dict(repo_id="wrldsuksgo2mars/agentic-quant-panel-v2", revision=PANEL_REVISION,
               sha256sums_sha256=PANEL_SUMS)
PACKER = "reproduction/pack_exl3_standard.py"
ASSETS = ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
CHAT = dict(tokenize=True, add_generation_prompt=True, return_dict=False,
            padding=False, truncation=False, reasoning_effort="high", date_string="2026-11-18")
CONTRACT = dict(renderer="native-chat-template",
                raw_calibration="exl3-standard-pack-v1",
                batch_size=1, packing=False, order="dataset", **CHAT)


def long_witness(tokenizer, chat=CHAT, cutoff=2048):
    for units in (48, 72, 96, 128):
        length = (units * cutoff + 2047) // 2048
        prompt = "\n".join(
            f"Observation {i}: The airship records the position and colour of every star. "
            "The archivist compares the last recorded coordinate with the first."
            for i in range(length)
        )
        ids = tokenizer.apply_chat_template([dict(role="user", content=prompt)], **chat)
        if len(ids) > cutoff:
            return dict(input_ids=ids, tokens=len(ids), hessian_eligible=False,
                        input_ids_sha256=panel.ids_sha256(ids))
    raise ValueError("long context witness never crossed the sparse selector cutoff")


def load_token_batches(root, split="calibration"):
    yield from panel.load_token_batches(root, split, schema=SCHEMA, contract=CONTRACT, dataset=DATASET)


def read_v2(root):
    root = Path(root)
    if root.name != PANEL_REVISION or panel.sha256_file(root / "SHA256SUMS") != PANEL_SUMS:
        raise ValueError("require the pinned v2 snapshot and checksum inventory")
    hashes = {name: digest for digest, name in (
        line.split(maxsplit=1) for line in (root / "SHA256SUMS").read_text().splitlines())}
    names = ("panel.json", "manifest.json", *(s + ".jsonl" for s in panel.SPLITS),
             *("data/" + s + ".parquet" for s in (*panel.SPLITS, "exl3_standard")),
             "exl3_standard/manifest.json", "exl3_standard/segments.jsonl",
             "exl3_standard/upstream/tokenizer.py", PACKER)
    if any(hashes[name] != panel.PAYLOAD_SHA256[name]
           for name in ("manifest.json", *(s + ".jsonl" for s in panel.SPLITS))):
        raise ValueError("v2 agentic provenance differs from the released v1 panel")
    return panel.read_panel(root, payload_sha256={name: hashes[name] for name in names})


def append_standard(root, source, stage, ids, offsets, report, tokenizer_sha256):
    import torch
    from safetensors.torch import load_file

    packed = stage / "exl3_standard.safetensors"
    subprocess.run([sys.executable, root / PACKER, root, source, packed], check=True)
    receipt = json.loads(packed.with_suffix(".json").read_text())
    values = load_file(str(packed))["input_ids"]
    digest = hashlib.sha256(values.numpy().tobytes()).hexdigest()
    if (values.shape != (250, 2048) or values.dtype != torch.int64
            or receipt["contract"] != "exl3-standard-pack-v1"
            or receipt["tokenizer_sha256"] != tokenizer_sha256 or receipt["input_ids_sha256"] != digest):
        raise ValueError("ExLlamaV3 calibration pack contract or checksum differs")
    # Hash the exact concatenated stream using the shared loader's canonical format.
    stream = hashlib.sha256()
    for i, row in enumerate(report["records"]):
        stream.update(panel.canonical(dict(id=row["id"], input_ids=ids[offsets[i]:offsets[i+1]].tolist())) + b"\n")
    for index, (row, provenance) in enumerate(zip(values, receipt["rows"], strict=True)):
        tokens = row.tolist()
        row_id = f"exl3_standard/{index:03d}"
        if panel.ids_sha256(tokens) != provenance["token_ids_sha256"]:
            raise ValueError(f"standard panel row digest differs: {row_id}")
        ids.extend(tokens)
        offsets.append(len(ids))
        report["records"].append(dict(id=row_id, tokens=len(tokens),
                                      input_ids_sha256=provenance["token_ids_sha256"],
                                      source=provenance["source"], kind=provenance["kind"]))
        stream.update(panel.canonical(dict(id=row_id, input_ids=tokens)) + b"\n")
    report.update(tokens=len(ids), count=len(offsets)-1, longest=max(report["longest"], values.shape[1]),
                  token_stream_sha256=stream.hexdigest())
    packed.unlink()  # The combined split now owns all token IDs.
    return dict(contract=receipt["contract"], count=values.shape[0], tokens=values.numel(),
                input_ids_sha256=digest, receipt="exl3_standard.json")


def prepare(source, corpus, output, *, source_repo="tencent/Hy4-flash",
            source_revision=SOURCE_REVISION, schema=SCHEMA, contract=CONTRACT, chat=CHAT,
            witness_cutoff=2048, assets=ASSETS, expected_assets=None, software=None,
            diagnostics=None):
    import torch
    from transformers import PreTrainedTokenizerFast

    source, output = Path(source).absolute(), Path(output).absolute()
    if source.name != source_revision or output.exists():
        raise ValueError("require the pinned source and a fresh corpus directory")
    identity = {name: panel.sha256_file(source / name) for name in assets}
    if expected_assets is not None and identity != expected_assets:
        raise ValueError("pinned source tokenizer asset hashes differ")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(source, local_files_only=True)
    if (not tokenizer.is_fast or not isinstance(tokenizer.chat_template, str)
            or hashlib.sha256(tokenizer.chat_template.encode()).hexdigest() != identity["chat_template.jinja"]):
        raise ValueError("native tokenizer did not load the pinned chat template")
    corpus = Path(corpus)
    rows, provenance = read_v2(corpus)
    manifest = dict(schema=schema, status="passed", contract=contract,
        dataset=DATASET, components={},
        historical_selection=provenance["selection"],
        source=dict(repo_id=source_repo, revision=source_revision, files=identity),
        software={**panel.software_identity(), **(software or {})}, tokenizer=dict(vocab_size=len(tokenizer)), splits={})
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=output.name + ".partial-", dir=output.parent))
    try:
        for split in panel.SPLITS:
            ids, offsets, report = panel.tokenize_rows(rows[split], tokenizer, chat_kwargs=chat)
            if split == "calibration":
                manifest["components"]["agentic"] = dict(count=report["count"], tokens=report["tokens"],
                    token_stream_sha256=report["token_stream_sha256"])
                manifest["components"]["exl3_standard"] = append_standard(
                    corpus, source, stage, ids, offsets, report, identity["tokenizer.json"])
            path = stage / (split + ".safetensors")
            receipt = checkpoint_io().write_safetensors(path,
                dict(input_ids=torch.frombuffer(ids, dtype=torch.int64).clone(),
                     offsets=torch.tensor(offsets, dtype=torch.int64)),
                metadata=dict(schema=schema, split=split), hash_algorithm="sha256")
            manifest["splits"][split] = dict(report, file=path.name, sha256=receipt["sha256"])
        witness = long_witness(tokenizer, chat, witness_cutoff)
        atomic_json(stage / "long-context-smoke.json", witness)
        manifest["long_context_smoke"] = dict(file="long-context-smoke.json", tokens=witness["tokens"],
            sha256=panel.sha256_file(stage / "long-context-smoke.json"), hessian_eligible=False)
        if identity != {name: panel.sha256_file(source / name) for name in assets}:
            raise ValueError("source tokenizer changed during preparation")
        if diagnostics is not None:
            manifest["diagnostics"] = diagnostics(manifest["splits"], provenance)
        atomic_json(stage / "manifest.json", manifest)
        for split in panel.SPLITS:
            panel.load_token_records(stage, split, schema=schema, contract=contract, dataset=DATASET)
        stage.rename(output)
        fsync_directory(output.parent)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return manifest


def run_cli(prepare_fn):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("source", "corpus", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    manifest = prepare_fn(args.source, args.corpus, args.output)
    print(json.dumps({"splits": {name: {key: value[key] for key in ("count", "tokens", "longest")}
                                for name, value in manifest["splits"].items()},
                      "long_context_smoke": manifest["long_context_smoke"]}), flush=True)


def main():
    run_cli(prepare)


if __name__ == "__main__":
    main()
