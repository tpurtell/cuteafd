"""Pack and attest the v2 agentic and standard calibration data for full Tencent Hy4."""
from hy4_flash import prepare_corpus as family
from kimi_k3 import prepare_corpus as panel
from .source import BASE_REPO, BASE_REVISION

SCHEMA = "cuteafd-hy4-corpus-v2"
DATASET = family.DATASET
CHAT = dict(family.CHAT, date_string="2026-11-22")
CONTRACT = dict(family.CONTRACT, **CHAT)


def prepare(source, corpus, output):
    return family.prepare(source, corpus, output, source_repo=BASE_REPO, source_revision=BASE_REVISION,
                          schema=SCHEMA, contract=CONTRACT, chat=CHAT)


def load_token_batches(root, split="calibration"):
    yield from panel.load_token_batches(root, split, schema=SCHEMA, contract=CONTRACT, dataset=DATASET)


if __name__ == "__main__":
    family.run_cli(prepare)
