"""Build ERNIE 5.5's prompt-ID stream and separate long-context parity witnesses."""
from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
import shutil
import tempfile

from hy4_flash.prepare_corpus import DATASET, append_standard, read_v2
from kimi_k3 import prepare_corpus as panel
from kimi_k3.uniform_k2.storage import atomic_json, checkpoint_io, fsync_directory

SOURCE_REPO = "baidu/ERNIE-5.5-Flash-Preview-BF16"
SOURCE_REVISION = "33a9925b17506c5ff57accac2b31a6520c951c4e"
SOURCE_FILES = {
    "encoding_ernie.py": "0b353d1eb330574cab7704c7a09323fefb0b1b0b594db151f96b67848e3d1912",
    "decoding_ernie.py": "0ff13c80fcc8c10237c259d8bcd04e7b854ee940605bd84421c9eb89e7092b97",
    "tokenization_ernie.py": "50dde281b9913278cee68ba587398ebb500343447687f67e4ffe06df83098ec2",
    "tokenizer.json": "20e7c81f36fdfb90db664583fc6426d5c2ac6d6bcaf631c910941a8d8f30f2d9",
    "tokenizer_config.json": "3643352cfb6ea8efbd6dd2b1240570e0527ea31c36bc17f8bfdee6970fd1384b",
    "config.json": "8c32ad48cecf14cb558ce8db3c00720031671af405bc6d91da9499935331a51f",
    "generation_config.json": "094c1db050f9ed46b7f54628b5bffea633a65ee1a50a6ba292e6c005278d956e",
}
SCHEMA = "cuteafd-ernie55-flash-corpus-v1"
CHAT = dict(enable_thinking=True, reasoning_effort="high", add_generation_prompt=True)
CONTRACT = dict(renderer="encoding_ernie.render_chat+ErnieChatTokenizer.encode_segments",
                tools="published-xlam-demonstrations-v1", date="2026-11-26", batch_size=1,
                raw_calibration="exl3-standard-pack-v1", padding=False, truncation=False,
                packing=False, sorting=False, **CHAT)


def verify_tokenizer_assets(source):
    if (source.name != SOURCE_REVISION
            or {f: panel.sha256_file(source / f) for f in SOURCE_FILES} != SOURCE_FILES
            or (source / "added_tokens.json").exists()):
        raise ValueError("ERNIE tokenizer assets differ from the pinned snapshot")


def load_tokenizer(source):
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    verify_tokenizer_assets(source)
    cls = get_class_from_dynamic_module("tokenization_ernie.ErnieChatTokenizer", str(source),
                                        local_files_only=True)
    tokenizer = cls.from_pretrained(source, local_files_only=True)
    renderer = cls.render_chat.__globals__["render_chat"]
    if (panel.sha256_file(inspect.getfile(cls)) != SOURCE_FILES["tokenization_ernie.py"]
            or panel.sha256_file(inspect.getfile(renderer)) != SOURCE_FILES["encoding_ernie.py"]
            or len(tokenizer) != 201216):
        raise ValueError("ERNIE failed to load its checkpoint-native segmented tokenizer")
    return tokenizer


def tool_conversation(prompt):
    """Recover the published xLAM demos in order, including every declared tool and call."""
    tools, messages = [], []
    for demo in prompt.split("\n\n---\n\n"):
        if not demo.startswith("Tool schemas:\n"):
            raise ValueError("expected xLAM tool schemas")
        schema, tail = demo[len("Tool schemas:\n"):].split("\n\nUser query:\n")
        query, expected = tail.split("\n\nExpected tool calls:\n")
        schema, calls = json.loads(json.loads(schema)), json.loads(json.loads(expected))
        if not isinstance(schema, list) or not schema or not isinstance(calls, list):
            raise ValueError("missing xLAM tool declarations or calls")
        names = {s["name"] for s in schema}
        if any(c["name"] not in names or not isinstance(c["arguments"], dict) for c in calls):
            raise ValueError("tool call differs from its declared schema")
        # Tool names can repeat between demos with different schemas. Preserve each occurrence.
        tools.extend(dict(type="function", function=s) for s in schema)
        messages.extend([dict(role="user", content=query), dict(role="assistant", content="",
                        tool_calls=[dict(type="function", function=c) for c in calls])])
    return messages, tools


def encode_row(row, family, tokenizer):
    if family == "tool_call":
        messages, tools = tool_conversation(row["prompt"])
    else:
        messages, tools = [dict(role="user", content=row["prompt"])], None
    rendered = tokenizer.render_chat(messages, tools=tools, **CHAT)
    if tools:
        controls = "".join(segment.text for segment in rendered.segments if segment.allow_special)
        if ("<ernie:tool_list>" not in controls
                or controls.count("<ernie:tool_call>") != sum(bool(m.get("tool_calls")) for m in messages)):
            raise ValueError(f"native tool declarations/calls absent: {row['id']}")
    return row, tokenizer.encode_segments(rendered.segments), rendered.text


def write_witnesses(stage, tokenizer, heldout, original):
    """Make eight single long user messages from held-out prose for parity checks only."""
    import torch

    passages = [(r["id"], r["prompt"]) for r, m in zip(heldout, original, strict=True)
                if m["domain"] == "wiki"]
    if not passages:
        raise ValueError("long-context witnesses need held-out wiki passages")
    tensors, records = {}, []
    passage_lengths = {row_id: len(tokenizer.encode_text(text)) for row_id, text in passages}
    for i, target in enumerate((16384, 20480, 24576, 32768, 40960, 49152, 57344, 65536)):
        pieces, size = [], 0
        while size < target:
            row_id, passage = passages[(i + len(pieces)) % len(passages)]
            pieces.append((row_id, passage))
            size += passage_lengths[row_id] + 1
        def encode():
            text = "\n\n".join(s for _, s in pieces)
            return tokenizer.encode_segments(tokenizer.render_chat([dict(role="user", content=text)],
                                                                    **CHAT).segments)
        ids = encode()
        while len(ids) < 16384:
            pieces.append(passages[(i + len(pieces)) % len(passages)])
            ids = encode()
        while len(ids) > 65536:
            pieces.pop()
            ids = encode()
        name = f"witness_{i:02}"
        tensors[name] = torch.tensor(ids, dtype=torch.int64)
        records.append(dict(id=name, tokens=len(ids), input_ids_sha256=panel.ids_sha256(ids),
                            source_ids=[r for r, _ in pieces]))
    path = stage / "long-context-witnesses.safetensors"
    receipt = checkpoint_io().write_safetensors(path, tensors, hash_algorithm="sha256")
    return dict(file=path.name, sha256=receipt["sha256"], records=records, hessian_eligible=False)


def load_witnesses(root):
    """Read parity inputs only. The shared calibration loader rejects this split."""
    import torch
    from safetensors.torch import load_file

    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if (manifest["schema"] != SCHEMA or manifest["status"] != "passed"
            or manifest["contract"] != CONTRACT or manifest["dataset"] != DATASET):
        raise ValueError("long-context witness corpus differs")
    report = manifest["long_context_witnesses"]
    file = root / "long-context-witnesses.safetensors"
    if (report["hessian_eligible"] is not False or report["file"] != file.name
            or panel.sha256_file(file) != report["sha256"]):
        raise ValueError("long-context witness receipt differs")
    tensors = load_file(str(file))
    expected = [f"witness_{i:02}" for i in range(8)]
    if set(tensors) != set(expected) or [r["id"] for r in report["records"]] != expected:
        raise ValueError("long-context witness inventory differs")
    for record in report["records"]:
        ids = tensors[record["id"]]
        if (ids.dtype != torch.int64 or ids.ndim != 1 or not 16384 <= ids.numel() <= 65536
                or record["tokens"] != ids.numel()
                or panel.ids_sha256(ids.tolist()) != record["input_ids_sha256"]):
            raise ValueError("long-context witness IDs differ")
        yield ids[None, :]


def load_token_batches(root, split="calibration"):
    yield from panel.load_token_batches(root, split, schema=SCHEMA, contract=CONTRACT, dataset=DATASET)


def prepare(source, corpus, output):
    import torch

    source, corpus, output = map(Path, (source, corpus, output))
    if output.exists():
        raise FileExistsError(output)
    tokenizer = load_tokenizer(source)
    rows, provenance = read_v2(corpus)
    manifest = dict(schema=SCHEMA, status="passed", contract=CONTRACT, dataset=DATASET,
                    source=dict(repo_id=SOURCE_REPO, revision=SOURCE_REVISION, files=SOURCE_FILES),
                    software=panel.software_identity(), tokenizer=dict(vocab_size=len(tokenizer)),
                    screening_calibration_identity_subset=provenance["screening_calibration_identity_subset"],
                    components={}, splits={})
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=output.name + ".partial-", dir=output.parent))
    try:
        for split in panel.SPLITS:
            original = provenance["splits"][split]["records"]
            rendered_rows = (encode_row(row, m["family"], tokenizer)
                             for row, m in zip(rows[split], original, strict=True))
            ids, offsets, report = panel.collect_rendered_rows(rendered_rows, len(tokenizer))
            report["tool_call_records"] = sum(m["family"] == "tool_call" for m in original)
            if split == "calibration":
                manifest["components"]["agentic"] = {key: report[key] for key in
                                                     ("count", "tokens", "token_stream_sha256")}
                manifest["components"]["exl3_standard"] = append_standard(
                    corpus, source, stage, ids, offsets, report, SOURCE_FILES["tokenizer.json"])
            path = stage / (split + ".safetensors")
            receipt = checkpoint_io().write_safetensors(path,
                dict(input_ids=torch.frombuffer(ids, dtype=torch.int64), offsets=torch.tensor(offsets)),
                metadata=dict(schema=SCHEMA, split=split), hash_algorithm="sha256")
            manifest["splits"][split] = dict(report, file=path.name, sha256=receipt["sha256"])
        manifest["long_context_witnesses"] = write_witnesses(
            stage, tokenizer, rows["heldout"], provenance["splits"]["heldout"]["records"])
        verify_tokenizer_assets(source)
        atomic_json(stage / "manifest.json", manifest)
        for split in panel.SPLITS:
            panel.load_token_records(stage, split, schema=SCHEMA, contract=CONTRACT, dataset=DATASET)
        for _ in load_witnesses(stage):
            pass
        stage.rename(output)
        fsync_directory(output.parent)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("source", "corpus", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    result = prepare(**vars(parser.parse_args()))
    print(json.dumps(result["components"], indent=2))
