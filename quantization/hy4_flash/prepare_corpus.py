"""Prepare the pinned agentic, ExLlamaV3 standard, and long-context panel with the native Hy4 tokenizer."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import panel_corpus
from kimi_k3 import prepare_corpus as panel

SOURCE_REVISION = "d88c9dfd96ba4f2e6b92adba0b8e8f1c8696f959"
SCHEMA = "cuteafd-hy4-flash-corpus-v2"
PANEL_REVISION = panel_corpus.DEFAULT_PANEL_REVISION
PANEL_SUMS = panel_corpus.DEFAULT_PANEL_SHA256SUMS_SHA256
DATASET = dict(panel_corpus.DEFAULT_DATASET)
PACKER = panel_corpus.EXL3_STANDARD_PACKER
ASSETS = ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
CHAT = dict(tokenize=True, add_generation_prompt=True, return_dict=False,
            padding=False, truncation=False, reasoning_effort="high", date_string="2026-11-18")
CONTRACT = dict(renderer="native-chat-template",
                raw_calibration=panel_corpus.EXL3_STANDARD_CONTRACT,
                batch_size=1, packing=False, order="dataset", **CHAT)


def long_witness(tokenizer, chat=CHAT):
    for length in (48, 72, 96, 128):
        prompt = "\n".join(
            f"Observation {i}: The airship records the position and colour of every star. "
            "The archivist compares the last recorded coordinate with the first."
            for i in range(length)
        )
        ids = tokenizer.apply_chat_template([dict(role="user", content=prompt)], **chat)
        if len(ids) > 2048:
            return dict(input_ids=ids, tokens=len(ids), hessian_eligible=False,
                        input_ids_sha256=panel.ids_sha256(ids))
    raise ValueError("long context witness never crossed the sparse selector cutoff")


def load_token_batches(root, split="calibration", *, schema=SCHEMA, contract=CONTRACT,
                       dataset=DATASET, require_components=None, full=None):
    yield from panel_corpus.load_token_batches(
        root, split, schema=schema, contract=contract, dataset=dataset,
        require_components=require_components, full=full,
    )


def read_v2(root, *, panel_revision=None, panel_sha256sums=None, components=None, full=True):
    loaded = panel_corpus.read_panel_v2(
        root,
        panel_revision=panel_revision,
        panel_sha256sums=panel_sha256sums,
        components=components,
        full=full,
    )
    return loaded["agentic_rows"], loaded["agentic_provenance"]


def prepare(source, corpus, output, *, source_repo="tencent/Hy4-flash",
            source_revision=SOURCE_REVISION, schema=SCHEMA, contract=None, chat=CHAT,
            components=None, full=True, panel_revision=None, panel_sha256sums=None):
    active_contract = CONTRACT if contract is None and components is None and panel_revision is None else contract
    return panel_corpus.prepare_panel_corpus(
        source,
        corpus,
        output,
        source_repo=source_repo,
        source_revision=source_revision,
        chat_kwargs=chat,
        schema=schema,
        contract=active_contract,
        assets=ASSETS,
        components=components,
        full=full,
        panel_revision=panel_revision,
        panel_sha256sums=panel_sha256sums,
        long_witness_fn=long_witness,
    )


def run_cli(prepare_fn):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("source", "corpus", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--panel-revision", default=None)
    parser.add_argument("--components", default=None)
    parser.add_argument("--subset", action="store_true")
    args = parser.parse_args()
    comp_list = [c.strip() for c in args.components.split(",") if c.strip()] if args.components else None
    manifest = prepare_fn(
        args.source,
        args.corpus,
        args.output,
        components=comp_list,
        full=not args.subset,
        panel_revision=args.panel_revision,
    )
    print(json.dumps({"dataset": manifest["dataset"],
                      "component_order": manifest["component_order"],
                      "splits": {name: {key: value[key] for key in ("count", "tokens", "longest")}
                                 for name, value in manifest["splits"].items()},
                      "long_context_smoke": manifest["long_context_smoke"]}), flush=True)


def main():
    run_cli(prepare)


if __name__ == "__main__":
    main()
