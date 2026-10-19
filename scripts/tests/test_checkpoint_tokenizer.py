"""Native checkpoint tokenization/rendering is pinned, not converted to a fast tokenizer."""
import hashlib
import json
from pathlib import Path
import sys

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python/reference"))
from checkpoint_tokenizer import load_tokenizer, tokenizer_identity
from fidelity_generation import ChatRenderer, token_ids
from fidelity_windows import verify_snapshot


@pytest.fixture
def native(tmp_path):
    snapshot = tmp_path / ("a" * 40)
    snapshot.mkdir()
    (snapshot / "config.json").write_text('{"model_type":"fixture"}')
    (snapshot / "tokenizer_config.json").write_text(json.dumps({
        "tokenizer_class": "FixtureTokenizer", "auto_map": {
            "AutoTokenizer": ["tokenization_fixture.FixtureTokenizer", None]}}))
    (snapshot / "vocab.model").write_text("fixture byte vocabulary")
    (snapshot / "render_fixture.py").write_text('''def render(messages, add_generation_prompt):
    return "".join(m["role"] + ":" + (m.get("reasoning_content") or "") +
                   (m.get("content") or "") + "|" for m in messages) + (
                       "assistant:" if add_generation_prompt else "")
''')
    (snapshot / "tokenization_fixture.py").write_text('''from transformers import PreTrainedTokenizer
from .render_fixture import render

class FixtureTokenizer(PreTrainedTokenizer):
    vocab_files_names = {"vocab_file": "vocab.model"}
    def __init__(self, vocab_file, **kwargs):
        self.resource = open(vocab_file).read()
        super().__init__(**kwargs)
    @property
    def vocab_size(self):
        return 256
    def get_vocab(self):
        return {chr(i): i for i in range(256)}
    def _tokenize(self, text):
        return list(text)
    def _convert_token_to_id(self, token):
        return ord(token)
    def _convert_id_to_token(self, index):
        return chr(index)
    def convert_tokens_to_string(self, tokens):
        return "".join(tokens)
    def apply_chat_template(self, messages, tools=None, tokenize=False, add_generation_prompt=False):
        text = render(messages, add_generation_prompt)
        return self.encode(text, add_special_tokens=False) if tokenize else text
''')
    return snapshot


def manifest(snapshot):
    identity = tokenizer_identity(snapshot)
    arm = {**identity, "snapshot_revision": snapshot.name,
           "config_sha256": hashlib.sha256((snapshot / "config.json").read_bytes()).hexdigest()}
    return {**identity, "generation_arm": arm, "reference_snapshot": arm}


def test_fast_checkpoint_identity_and_token_ids_unchanged(tmp_path):
    tokenizer = Tokenizer(models.WordLevel({"hello": 0, "world": 1, "[UNK]": 2}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    path = tmp_path / "tokenizer.json"
    tokenizer.save(str(path))
    assert tokenizer_identity(tmp_path) == {"tokenizer_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    assert load_tokenizer(path).encode("hello world").ids == tokenizer.encode("hello world").ids


def test_native_source_closure_and_snapshot_hashes(native):
    pin = tokenizer_identity(native)
    assert set(pin["tokenizer_files"]) == {
        "tokenization_fixture.py", "render_fixture.py", "tokenizer_config.json", "vocab.model"}
    panel = manifest(native)
    assert verify_snapshot(panel, native)["tokenizer_files"] == pin["tokenizer_files"]
    # A caller cannot attest only a combined hash and omit the resource inventory.
    panel["generation_arm"] = {k: v for k, v in panel["generation_arm"].items() if k != "tokenizer_files"}
    with pytest.raises(ValueError, match="source/resource hashes differ"):
        verify_snapshot(panel, native)


@pytest.mark.parametrize("name", ["tokenization_fixture.py", "render_fixture.py", "tokenizer_config.json", "vocab.model"])
def test_native_source_and_resource_mutation_rejected(native, name):
    panel = manifest(native)
    path = native / name
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="snapshot tokenizer differs"):
        verify_snapshot(panel, native)


def test_native_fixture_roundtrips_shipped_python_chat(native):
    pytest.importorskip("transformers")
    tokenizer = load_tokenizer(native / "vocab.model", chat=True)
    renderer = ChatRenderer(tokenizer)
    body = {"messages": [{"role": "user", "content": "fixture"}]}
    assistant = {"role": "assistant", "reasoning_content": "reason;", "content": "answer"}
    ids = renderer.prompt(body) + renderer.completion(body, assistant)
    assert tokenizer.decode(ids, clean_up_tokenization_spaces=False) == renderer.render(body, assistant)
    assert ids == tokenizer.apply_chat_template(body["messages"] + [assistant], tokenize=True)
    assert token_ids(tokenizer, "fixture") == tokenizer.encode("fixture")
    assert renderer.identity["tokenizer_files"] == tokenizer_identity(native)["tokenizer_files"]
    assert renderer.identity["chat_template_kind"] == "checkpoint_python"


def test_official_checkpoint_encoder_is_used_without_a_jinja_template(tmp_path):
    pytest.importorskip("transformers")
    from tokenizers import decoders, trainers
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator(["system user assistant reason answer"],
        trainers.BpeTrainer(vocab_size=300, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                            special_tokens=["[UNK]"]))
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "config.json").write_text('{"model_type":"deepseek_v41"}')
    (tmp_path / "tokenizer_config.json").write_text('{"tokenizer_class":"PreTrainedTokenizerFast"}')
    encoding = tmp_path / "encoding/encoding.py"
    encoding.parent.mkdir()
    encoding.write_text('''def encode_messages(messages, *, thinking_mode, reasoning_effort):
    assert thinking_mode == "thinking" and reasoning_effort == "high"
    assert messages[0]["tools"][0]["function"]["name"] == "read_file"
    return "".join(m["role"] + ":" + (m.get("reasoning_content") or "") +
                   (m.get("content") or "") + "|" for m in messages) + (
                       "assistant:" if messages[-1]["role"] != "assistant" else "")
''')
    renderer = ChatRenderer(load_tokenizer(tmp_path, chat=True))
    body = {"messages": [{"role": "system", "content": "system"}, {"role": "user", "content": "user"}],
            "tools": [{"type": "function", "function": {"name": "read_file"}}]}
    assistant = {"role": "assistant", "reasoning_content": "reason", "content": "answer"}
    ids = renderer.prompt(body) + renderer.completion(body, assistant)
    assert renderer.tokenizer.decode(ids, clean_up_tokenization_spaces=False) == renderer.render(body, assistant)
    assert "tools" not in body["messages"][0]
    assert renderer.identity == {"chat_template_kind": "checkpoint_encoder",
        "chat_template_sha256": hashlib.sha256(encoding.read_bytes()).hexdigest(),
        "chat_encoder_file": "encoding/encoding.py"}
    panel = manifest(tmp_path)
    panel["generation_arm"].update(renderer.identity)
    verify_snapshot(panel, tmp_path)
    encoding.write_text(encoding.read_text() + "\n")
    with pytest.raises(ValueError, match="checkpoint chat encoder differs"):
        verify_snapshot(panel, tmp_path)
    encoding.unlink()
    with pytest.raises(ValueError, match="official encoding/encoding.py"):
        load_tokenizer(tmp_path, chat=True)


def test_native_mapping_cannot_load_another_repository(native):
    path = native / "tokenizer_config.json"
    path.write_text(json.dumps({"auto_map": {"AutoTokenizer": ["other/repo--module.Tokenizer", None]}}))
    with pytest.raises(ValueError, match="local AutoTokenizer"):
        tokenizer_identity(native)
