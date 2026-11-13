import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inkling2 import prepare_corpus as corpus


class NativeTokenizer:
    def __len__(self):
        return 0x110000

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return list(map(ord, text))

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {**corpus.CHAT_KWARGS, "tokenize": kwargs["tokenize"]}
        assert len(messages) == 1 and messages[0]["role"] == "user"
        rendered = ("<|message_system|><|content_text|>Thinking effort level: 0.99<|end_message|>"
                    "<|message_user|><|content_text|>" + messages[0]["content"] + "<|end_message|>")
        return self.encode(rendered, add_special_tokens=False) if kwargs["tokenize"] else rendered


def test_native_rows_preserve_order_and_complete_text_without_packing():
    tokenizer = NativeTokenizer()
    rows = [dict(id="long", prompt="a" * 1500), dict(id="short", prompt="b")]
    flat, offsets, report = corpus.tokenize_rows(rows, tokenizer)
    assert [r["id"] for r in report["records"]] == ["long", "short"]
    assert report["records"][0]["tokens"] > 1500
    assert offsets == [0, report["records"][0]["tokens"], len(flat)]
    assert corpus.CONTRACT["packing"] is False and corpus.CONTRACT["truncation"] is False
    assert corpus.CHAT_KWARGS["date_string"] == "2026-11-13"
    assert len(corpus.tokenizer_probes(tokenizer)) == 5


def test_native_probe_rejects_render_id_disagreement():
    class Broken(NativeTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            result = super().apply_chat_template(messages, **kwargs)
            return result + [1] if kwargs["tokenize"] else result
    with pytest.raises(ValueError, match="tokenization differs"):
        corpus.tokenizer_probes(Broken())


def test_long_window_smoke_is_heldout_only_and_checked(tmp_path):
    tokenizer = NativeTokenizer()
    row = corpus.long_context_smoke(tokenizer)
    assert len(row["input_ids"]) == row["tokens"] > 1024
    assert row["hessian_eligible"] is False
    path = tmp_path / "long-context-smoke.json"
    path.write_bytes(corpus.canonical(row))
    manifest = dict(schema=corpus.SCHEMA, contract=corpus.CONTRACT, tokenizer=dict(vocab_size=len(tokenizer)),
        long_context_smoke=dict(file=path.name, sha256=corpus.sha256_file(path), hessian_eligible=False))
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    assert corpus.load_long_context_smoke(tmp_path) == row
    with pytest.raises(ValueError, match="unknown split"):
        corpus.load_token_records(tmp_path, "long_context_smoke")
    row["input_ids"][0] += 1
    path.write_bytes(corpus.canonical(row))
    manifest["long_context_smoke"]["sha256"] = corpus.sha256_file(path)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="token identity"):
        corpus.load_long_context_smoke(tmp_path)


def test_saved_standard_rows_load_as_individual_all_ones_batches(tmp_path):
    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file
    tokenizer = NativeTokenizer()
    flat, offsets, report = corpus.tokenize_rows([dict(id="first", prompt="hello"), dict(id="second", prompt="world")], tokenizer)
    path = tmp_path / "calibration.safetensors"
    save_file(dict(input_ids=torch.tensor(list(flat)), offsets=torch.tensor(offsets)), str(path))
    report.update(file=path.name, sha256=corpus.sha256_file(path))
    manifest = dict(schema=corpus.SCHEMA, status="passed", contract=corpus.CONTRACT,
        dataset=dict(repo_id=corpus.REPO_ID, revision=corpus.REVISION, files=corpus.PAYLOAD_SHA256),
        tokenizer=dict(vocab_size=len(tokenizer)), splits=dict(calibration=report))
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    records = corpus.load_token_records(tmp_path)
    batches = list(corpus.load_token_batches(tmp_path))
    assert [r["id"] for r in records] == ["first", "second"]
    assert len(batches) == 2
    for record, batch in zip(records, batches, strict=True):
        assert batch["input_ids"].tolist() == [record["input_ids"]]
        assert torch.equal(batch["attention_mask"], torch.ones_like(batch["input_ids"]))
