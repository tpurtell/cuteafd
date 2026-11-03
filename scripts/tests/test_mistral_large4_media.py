"""Checkpoint-derived Pixtral row layout and official feature injection on CPU."""
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python/reference"))
import mistral_large4_media as media


@pytest.fixture
def snapshot(tmp_path):
    path = tmp_path / "snapshot"
    path.mkdir()
    documents = {
        "config.json": {"model_type": "mistral_large4", "image_token_index": 5,
            "vision_config": {"patch_size": 14}, "spatial_merge_size": 2,
            "text_config": {"vocab_size": 32}},
        "params.json": {"vision_encoder": {"image_token_id": 5, "image_break_token_id": 6, "image_end_token_id": 7}},
        "processor_config.json": {"spatial_merge_size": 2, "image_processor": {
            "image_processor_type": "PixtralImageProcessor", "patch_size": 14,
            "size": {"longest_edge": 1540}, "image_mean": [0.5] * 3, "image_std": [0.5] * 3,
            "resample": 3, "do_resize": True, "do_normalize": True, "do_rescale": True,
            "rescale_factor": 1 / 255, "do_convert_rgb": True}},
        "tokenizer.json": {}, "consolidated.safetensors.index.json": {"weight_map": {}},
    }
    for name, value in documents.items():
        (path / name).write_text(json.dumps(value))
    return path


def panel(snapshot):
    return {"image_processor": media.processor_contract(snapshot), "windows": [{
        "tokens": [1, 5, 5, 6, 5, 5, 7, 2, 3], "roles": ["ctx"] * 9, "score_from": 7,
        "media": [{"kind": "image", "start": 1, "len": 6, "grid": [1, 4, 4],
            "key": "ab" * 32, "fixture": {"path": "image.png", "sha256": "cd" * 32}}]}]}


def test_checkpoint_contract_is_authoritative_and_reserved_tokens_bound(snapshot):
    manifest = panel(snapshot)
    media.validate_spans(manifest, snapshot)
    for mutation in ("contract", "marker", "unbound", "extent"):
        bad = copy.deepcopy(manifest)
        window = bad["windows"][0]
        if mutation == "contract":
            bad["image_processor"]["merge"] = 1
        elif mutation == "marker":
            window["tokens"][3] = 7
        elif mutation == "unbound":
            window["tokens"][0] = 5
        else:
            window["media"][0]["len"] = 5
        with pytest.raises(ValueError):
            media.validate_spans(bad, snapshot)
    config = json.loads((snapshot / "config.json").read_text())
    config["image_token_index"] = 8
    (snapshot / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="marker ids"):
        media.processor_contract(snapshot)


def test_media_resume_provenance_revalidates_pinned_bytes(snapshot, tmp_path, monkeypatch):
    manifest = panel(snapshot)
    image = tmp_path / "image.png"
    image.write_bytes(b"pinned image bytes")
    manifest["windows"][0]["media"][0]["fixture"]["sha256"] = hashlib.sha256(image.read_bytes()).hexdigest()
    monkeypatch.setattr(media, "snapshot_identity", lambda p: {"snapshot": "verified"})
    args = types.SimpleNamespace(snapshot=snapshot, media_root=tmp_path, windows=tmp_path / "set.json", tower_dtype="bf16")
    first = media.provenance(args, manifest)
    args.tower_dtype = "fp32"
    assert media.provenance(args, manifest) != first
    assert all(len(digest) == 64 for digest in first["sources"].values())
    image.write_bytes(b"mutated")
    with pytest.raises(ValueError, match="pinned media identity"):
        media.provenance(args, manifest)


def test_shared_layer_major_injects_only_span_rows():
    torch = pytest.importorskip("torch")
    import layer_major
    family = types.SimpleNamespace(embed=lambda tokens: torch.arange(len(tokens) * 3).reshape(1, len(tokens), 3).float())
    window = {"tokens": list(range(9)), "media": [{"start": 1, "len": 6, "key": "image"}]}
    original = family.embed(window["tokens"])
    features = {"image": torch.full((6, 3), 42, dtype=torch.bfloat16)}
    actual = layer_major.embed_window(family, window, features)
    assert torch.equal(actual[0, 1:7], features["image"].float())
    assert torch.equal(actual[:, :1], original[:, :1])
    assert torch.equal(actual[:, 7:], original[:, 7:])
    assert torch.equal(layer_major.embed_window(family, {"tokens": list(range(9))}), original)
    with pytest.raises(ValueError, match="missing official"):
        layer_major.embed_window(family, window)
    with pytest.raises(ValueError, match="extent"):
        layer_major.embed_window(family, window, {"image": torch.zeros(5, 3)})


def test_official_processor_and_tiny_tower_preserve_marker_rows(snapshot, tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers.models.pixtral.modeling_pixtral")
    from PIL import Image
    from families.mistral_large4 import model
    tower, projector = model.load_vision_official({"vision_config": {
        "hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 1,
        "num_attention_heads": 4, "patch_size": 14, "image_size": 56},
        "text_config": {"hidden_size": 48}, "spatial_merge_size": 2})
    for module in (tower, projector):
        for parameter in module.parameters():
            parameter.data = parameter.data.bfloat16()
        module.eval()
    image = Image.new("RGB", (53, 31), (20, 40, 60))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    data = buffer.getvalue()
    (tmp_path / "image.png").write_bytes(data)
    span = panel(snapshot)["windows"][0]["media"][0]
    span["fixture"]["sha256"] = hashlib.sha256(data).hexdigest()
    markers = [torch.arange(48).bfloat16(), -torch.arange(48).bfloat16()]
    processor = media.processor(snapshot)
    prepared = processor(images=image, return_tensors="pt")
    assert prepared["image_sizes"].tolist() == [[56, 56]]
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        actual = media.encode_span((tower, projector, markers), processor, tmp_path, span)
        with torch.inference_mode(), torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            expected = projector(tower(prepared["pixel_values"].bfloat16(),
                image_sizes=prepared["image_sizes"]).last_hidden_state.squeeze(0), prepared["image_sizes"])
    finally:
        torch.set_num_threads(previous)
    assert actual.shape == (6, 48)
    assert torch.equal(actual.reshape(2, 3, 48)[:, :2].reshape(4, 48), expected)
    assert torch.equal(actual[2], markers[0]) and torch.equal(actual[5], markers[1])
    span["grid"] = [1, 2, 4]
    with pytest.raises(ValueError, match="processor grid"):
        media.encode_span((tower, projector, markers), processor, tmp_path, span)


def test_identity_matches_serving_schema_and_source_mutation_fails(snapshot, monkeypatch):
    pytest.importorskip("transformers.models.pixtral.modeling_pixtral")
    identity = media.snapshot_identity(snapshot)
    assert set(identity) == {"snapshot_revision", "transformers", "modeling_sha256", "vision_modeling_sha256",
        "image_processing_sha256", "config_sha256", "tokenizer_sha256", "preprocessor_sha256", "index_sha256"}
    assert identity["config_sha256"] == hashlib.sha256((snapshot / "config.json").read_bytes()).hexdigest()
    pins = dict(media.model.SOURCE_PINS)
    pins["models/pixtral/image_processing_pil_pixtral.py"] = "0" * 64
    monkeypatch.setattr(media.model, "SOURCE_PINS", pins)
    with pytest.raises(ValueError, match="source differs"):
        media.processor(snapshot)
