"""Provisional foreign-text panels preserve text and never fabricate generations."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("retokenize_builder", ROOT / "scripts/bench/fidelity-set.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class Source:
    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(i) for i in ids)


class Target(Source):
    chat_template = "test template"
    lossy = False

    def apply_chat_template(self, messages, **kwargs):
        return "".join("[" + m["role"] + "]" + m["content"] + "[/]" for m in messages)

    def __call__(self, text, **kwargs):
        return dict(input_ids=[ord(c) for c in text], offset_mapping=[(i, i + 1) for i in range(len(text))])

    def decode(self, ids, **kwargs):
        return super().decode(ids, **kwargs) + ("loss" if self.lossy else "")


def window():
    text = "<|im_start|>user\n" + "x" * 600 + "<|im_end|>\n<|im_start|>assistant\n" + "answer" * 100
    begin = text.index("answer")
    return dict(id="a00", block="A", tokens=[ord(c) for c in text],
                roles=["ctx"] * begin + ["gen"] * (len(text) - begin), score_from=len(text) - 512,
                bucket="0-2K")


def test_roundtrip_and_mask_mapping_at_chat_boundaries():
    result = builder.render_retokenized_window(window(), Source(), Target())
    text = Source().decode(result["tokens"])
    assert text == "[user]" + "x" * 600 + "[/][assistant]" + "answer" * 100 + "[/]"  # Mock ignores continuation.
    start = text.index("answer")
    assert result["roles"][start - 1] == "ctx"
    assert result["roles"][start:start + 600] == ["gen"] * 600
    assert result["roles"][-3:] == ["ctx"] * 3
    assert len(result["tokens"]) - result["score_from"] == 512


def test_lossy_tokenizer_and_media_are_refused(tmp_path):
    target = Target()
    target.lossy = True
    with pytest.raises(ValueError, match="lossy"):
        builder.render_retokenized_window(window(), Source(), target)
    w = window()
    manifest = dict(schema=builder.SET_SCHEMA, family="qwen4", checkpoint="source",
                    generation_arm=dict(checkpoint="source"), windows=[w], quick_windows=["a00"],
                    tokenizer_sha256="d" * 64)
    manifest["set_sha256"] = builder.set_hash(manifest)
    path = tmp_path / "windows.json"
    path.write_bytes(builder.canonical(manifest))
    kwargs = dict(family="step5", model="root", version="v1", arm=dict(tokenizer_sha256="a" * 64),
                  source_sha256=builder.hashlib.sha256(path.read_bytes()).hexdigest(),
                  source_tokenizer_sha256="d" * 64, template_sha256="b" * 64)
    result = builder.retokenize_text_set(path, Source(), Target(), **kwargs)
    assert "not Step-5's own greedy path" in result["retokenized_text_source"]["caveat"]
    assert result["generation_arm"] == manifest["generation_arm"]
    with pytest.raises(ValueError, match="source hash"):
        builder.retokenize_text_set(path, Source(), Target(), **{**kwargs, "source_sha256": "c" * 64})
    with pytest.raises(ValueError, match="source tokenizer"):
        builder.retokenize_text_set(path, Source(), Target(), **{**kwargs, "source_tokenizer_sha256": "e" * 64})
    manifest["windows"][0]["media"] = {"kind": "image"}
    manifest["set_sha256"] = builder.set_hash(manifest)
    path.write_bytes(builder.canonical(manifest))
    with pytest.raises(ValueError, match="media"):
        builder.retokenize_text_set(path, Source(), Target(),
            **{**kwargs, "source_sha256": builder.hashlib.sha256(path.read_bytes()).hexdigest()})


def test_tokens_crossing_role_boundary_are_context():
    class Merged(Target):
        def __call__(self, text, **kwargs):
            encoded = super().__call__(text, **kwargs)
            first = text.index("answer") - 1
            # Model a boundary-straddling token without changing decode text.
            encoded["offset_mapping"][first] = (first, first + 2)
            return encoded
    result = builder.render_retokenized_window(window(), Source(), Merged())
    first = Source().decode(result["tokens"]).index("answer")
    assert result["roles"][first - 1:first + 1] == ["ctx", "gen"]

def test_legacy_extension_is_pinned_context_and_minimal():
    w = dict(id="legacy", block="E", tokens=[ord("x")] * 500,
             roles=["ctx"] * 500, score_from=1, bucket="0-2K")
    identity = dict(path="PLAN.md", revision="a" * 40, sha256="b" * 64)
    result = builder.render_retokenized_window(w, Source(), Target(),
        legacy_source=("x" * 600, identity))
    assert len(result["tokens"]) == 513
    assert result["roles"] == ["ctx"] * 513
    assert result["score_from"] == 1
    assert result["tokenizer_length_extension"]["original_span"] == [0, 500]
    assert result["tokenizer_length_extension"]["extended_span"] == [0, 513]
    assert result["tokenizer_length_extension"]["sha256"] == "b" * 64
    with pytest.raises(ValueError, match="does not match"):
        builder.render_retokenized_window(w, Source(), Target(), legacy_source=("wrong", identity))
    with pytest.raises(ValueError, match="cannot supply"):
        builder.render_retokenized_window(w, Source(), Target())
