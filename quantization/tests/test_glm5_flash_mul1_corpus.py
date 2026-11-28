"""GLM Flash MUL1 uses a new source, panel, and pooled sparse selector witness."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from glm5_flash import prepare_corpus as previous
from glm5_flash.mul1_mc import corpus
from hy4_flash import prepare_corpus as v2


def test_previous_panel_is_rejected(tmp_path):
    manifest = dict(status="passed", schema=previous.SCHEMA, contract=previous.CONTRACT,
                    dataset=dict(repo_id=previous.REPO_ID, revision=previous.REVISION))
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="contract differs"):
        list(corpus.load_token_batches(tmp_path))


def test_sparse_witness_crosses_pooled_cutoff():
    class Tokenizer:
        def apply_chat_template(self, messages, **chat):
            assert chat["reasoning_effort"] == "max"
            return [1] * len(messages[0]["content"].split())

    witness = v2.long_witness(Tokenizer(), corpus.CHAT, corpus.WITNESS_MIN_TOKENS)
    assert witness["tokens"] // 4 > 2048
    assert witness["hessian_eligible"] is False


def test_group_exclusion_begins_at_513_complete_pools():
    cal = [dict(id="tail", tokens=2051), dict(id="extra_pool", tokens=2052)]
    shared = dict(group="g1", prompt_sha256="same")
    heldout = dict(group="g2", prompt_sha256="independent")
    provenance = dict(splits={name: dict(records=rows) for name, rows in
                              (("calibration", [shared]), ("screening", [shared]),
                               ("heldout", [heldout]))})
    report = corpus.corpus_diagnostics(dict(calibration=dict(records=cal)), provenance)
    assert report["dsa"]["selected_groups"] == 512
    assert report["dsa"]["sparse_calibration_rows"] == [cal[1]]
    assert report["split_overlap"]["screening_and_calibration"] == dict(group=1, prompt_sha256=1)
    assert report["split_overlap"]["calibration_and_heldout"] == dict(group=0, prompt_sha256=0)
