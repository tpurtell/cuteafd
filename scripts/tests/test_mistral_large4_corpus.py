"""Native user/control boundaries and immutable prepared-corpus validation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from safetensors.numpy import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "quantization"))
from mistral_large4 import prepare_corpus as corpus


def token_root(tmp_path, *, ids=(1, 33, 4), offsets=(0, 3)):
    path = tmp_path / "calibration.safetensors"
    save_file(dict(input_ids=np.asarray(ids, dtype=np.int64),
                   offsets=np.asarray(offsets, dtype=np.int64)), str(path))
    records = [dict(id="row-a", tokens=len(ids),
                    input_ids_sha256=corpus.shared.ids_sha256(ids))]
    stream = corpus.canonical(dict(id="row-a", input_ids=list(ids))) + b"\n"
    manifest = dict(schema=corpus.SCHEMA, status="passed", contract=corpus.CONTRACT,
        dataset=dict(repo_id=corpus.REPO_ID, revision=corpus.REVISION, files=corpus.PAYLOAD_SHA256),
        source=dict(repo_id=corpus.SOURCE_REPO, revision=corpus.SOURCE_REVISION),
        tokenizer=dict(vocab_size=204800), splits=dict(calibration=dict(
            file=path.name, sha256=corpus.sha256_file(path), records=records,
            count=1, tokens=len(ids), shortest=len(ids), longest=len(ids),
            token_stream_sha256=hashlib.sha256(stream).hexdigest())))
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def test_token_records_roundtrip_without_torch(tmp_path):
    token_root(tmp_path)
    assert corpus.load_token_records(tmp_path) == [dict(id="row-a", input_ids=[1, 33, 4])]


@pytest.mark.parametrize("mutation", ["source", "contract", "stream", "row"])
def test_token_records_reject_tampered_identity(tmp_path, mutation):
    manifest = token_root(tmp_path)
    if mutation == "source":
        manifest["source"]["revision"] = "0" * 40
    elif mutation == "contract":
        manifest["contract"] = {**manifest["contract"], "reasoning_effort": "max"}
    elif mutation == "stream":
        manifest["splits"]["calibration"]["token_stream_sha256"] = "0" * 64
    else:
        manifest["splits"]["calibration"]["records"][0]["input_ids_sha256"] = "0" * 64
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        corpus.load_token_records(tmp_path)


@pytest.mark.parametrize("ids,offsets", [((1, 204800, 4), (0, 3)),
                                         ((1, -1, 4), (0, 3)),
                                         ((1, 33, 4), (0, 2)),
                                         ((1, 33, 4), (0, 0, 3))])
def test_token_records_reject_invalid_ids_and_offsets(tmp_path, ids, offsets):
    token_root(tmp_path, ids=ids, offsets=offsets)
    with pytest.raises(ValueError):
        corpus.load_token_records(tmp_path)


def test_pilot_preserves_complete_records_in_order(monkeypatch):
    rows = [dict(id=str(i), input_ids=list(range(length)))
            for i, length in enumerate((12, 5, 4, 6, 3))]
    monkeypatch.setattr(corpus, "load_token_records", lambda *args: rows)
    assert corpus.pilot_records("unused", max_records=3, max_tokens=10, max_length=6) == [rows[1], rows[2]]
    with pytest.raises(ValueError, match="complete prompts"):
        corpus.pilot_records("unused", max_records=1, max_tokens=2, max_length=2)


def test_missing_panel_is_never_downloaded(tmp_path):
    with pytest.raises(ValueError, match="SparkNest"):
        corpus.download_corpus(tmp_path)


@pytest.fixture
def native():
    pytest.importorskip("mistral_common")
    pytest.importorskip("jinja2")
    pytest.importorskip("tokenizers")
    snapshot = (Path("/mnt/sparknest/hf-home/hub") /
        "models--mistralai--Mistral-Large-4-1T-A52B/snapshots" / corpus.SOURCE_REVISION)
    if not (snapshot / "tekken.json").is_file():
        pytest.skip("pinned native tokenizer is not installed")
    return corpus.NativeTokenizer(snapshot)


def test_native_literal_specials_never_become_control_tokens(native):
    probes = corpus.tokenizer_probes(native)
    special = probes[2]
    assert special["input_ids"] != special["rendered_text_input_ids"]
    assert special["input_ids"].count(1) == 1
    assert special["input_ids"].count(3) == 1
    assert special["input_ids"].count(4) == 1
    assert special["input_ids"][-1] == 4
    assert all(probes[i]["rendered_reencoding_equal"] for i in (0, 1, 3, 4))


def test_native_rejects_unreviewed_effort_and_conversation(native):
    with pytest.raises(ValueError, match="options"):
        native.apply_chat_template([dict(role="user", content="hello")],
                                   **{**corpus.CHAT_KWARGS, "reasoning_effort": "max"})
    with pytest.raises(ValueError, match="one literal user"):
        native.apply_chat_template([dict(role="assistant", content="hello")], **corpus.CHAT_KWARGS)
