"""Pack and attest the v2 agentic, standard, and long-context calibration data for full Tencent Hy4."""
from hy4_flash import prepare_corpus as family
from .source import BASE_REPO, BASE_REVISION

SCHEMA = "cuteafd-hy4-corpus-v2"
DATASET = family.DATASET
CHAT = dict(family.CHAT, date_string="2026-11-22")
CONTRACT = dict(family.CONTRACT, **CHAT)


def prepare(source, corpus, output, *, components=None, full=True, panel_revision=None, panel_sha256sums=None):
    active_contract = CONTRACT if components is None and panel_revision is None else None
    return family.prepare(source, corpus, output, source_repo=BASE_REPO, source_revision=BASE_REVISION,
                          schema=SCHEMA, contract=active_contract, chat=CHAT,
                          components=components, full=full, panel_revision=panel_revision,
                          panel_sha256sums=panel_sha256sums)


def load_token_batches(root, split="calibration", *, require_components=None, full=None):
    yield from family.load_token_batches(
        root, split, schema=SCHEMA, contract=CONTRACT, dataset=DATASET,
        require_components=require_components, full=full,
    )


if __name__ == "__main__":
    family.run_cli(prepare)
