"""Prepare GLM Flash's max-effort chat panel and the ordered raw EXL3 calibration rows."""
from pathlib import Path
import json

from hy4_flash import prepare_corpus as v2
from kimi_k3 import prepare_corpus as panel

SCHEMA = "cuteafd-glm5-flash-mul1-corpus-v2"
SOURCE_REPO = "zai-org/GLM-5.3-Flash-BF16"
SOURCE_REVISION = "a5b45eb41df6402735dedc900be14a42e8d5e538"
SOURCE_FILES = {
    "chat_template.jinja": "0c4099f3382d6c92700dfb99725025360966fd73032f0ecf32377c0d9e6309c5",
    "config.json": "33e63ec7fe607658be712bd6dd3c16c6549960d8e7f0483d34b939881b55f943",
    "generation_config.json": "230c30609ecbbb9e6583bedde8e7bdda0c6eb8fe5fad0eaeb3d1b293d751cb4f",
    "tokenizer.json": "19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d",
    "tokenizer_config.json": "98b1271574f41abf89427ae2dda030d94dc9478f0edc5a8bd240db213c6fd5fc",
}
CHAT = dict(v2.CHAT, reasoning_effort="max", date_string="2026-11-23")
CONTRACT = dict(v2.CONTRACT, **CHAT)
INDEX_TOPK = 2048
INDEX_KPOOL = 4
SELECT_GROUPS = INDEX_TOPK // INDEX_KPOOL
# A partial tail is always selected in addition to the complete pooled keys.
DSA_CUTOFF = SELECT_GROUPS * INDEX_KPOOL + INDEX_KPOOL - 1
WITNESS_MIN_TOKENS = 8192


def corpus_diagnostics(splits, provenance):
    calibration = splits["calibration"]["records"]
    sparse = [dict(id=row["id"], tokens=row["tokens"]) for row in calibration
              if row["tokens"] > DSA_CUTOFF]
    selection = provenance["splits"]
    def values(split, key):
        return {row[key] for row in selection[split]["records"]}

    overlaps = {}
    for a, b in (("screening", "calibration"), ("screening", "heldout"),
                 ("calibration", "heldout")):
        overlaps[f"{a}_and_{b}"] = {
            key: len(values(a, key) & values(b, key))
            for key in ("group", "prompt_sha256")
        }
    return dict(dsa=dict(selected_groups=SELECT_GROUPS, group_tokens=INDEX_KPOOL,
                         max_tokens_before_first_group_exclusion=DSA_CUTOFF,
                         sparse_calibration_count=len(sparse), sparse_calibration_rows=sparse),
                split_overlap=overlaps)


def prepare(source, corpus, output):
    import transformers

    repo = Path(__file__).resolve().parents[3]
    pinned = repo / "third_party/transformers"
    if not Path(transformers.__file__).resolve().is_relative_to(pinned):
        raise ValueError("prepare with the engine's pinned Transformers source on PYTHONPATH")
    lock = json.loads((repo / "third_party/transformers.lock.json").read_text())
    return v2.prepare(source, corpus, output, source_repo=SOURCE_REPO,
                      source_revision=SOURCE_REVISION, schema=SCHEMA, contract=CONTRACT,
                      chat=CHAT, witness_cutoff=WITNESS_MIN_TOKENS, assets=SOURCE_FILES,
                      expected_assets=SOURCE_FILES, diagnostics=corpus_diagnostics,
                      software={"transformers_revision": lock["revision"]})


def load_token_batches(root, split="calibration"):
    yield from panel.load_token_batches(root, split, schema=SCHEMA, contract=CONTRACT, dataset=v2.DATASET)


if __name__ == "__main__":
    v2.run_cli(prepare)
