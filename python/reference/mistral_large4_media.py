"""Verified official Pixtral features, including checkpoint-native row markers.

This exports the tower/projector only; it does not qualify the language model.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys

from families.mistral_large4 import model
from fidelity_media import image_extent, publish_immutable, read_fixture, write_features


def processor_contract(snapshot):
    """Derive the row layout from checkpoint files, never a probe's manifest."""
    snapshot = Path(snapshot)
    config = json.loads((snapshot / "config.json").read_bytes())
    document = json.loads((snapshot / "processor_config.json").read_bytes())
    processor = document["image_processor"]
    vision = json.loads((snapshot / "params.json").read_bytes())["vision_encoder"]
    if (config.get("model_type") != "mistral_large4"
            or processor.get("image_processor_type") != "PixtralImageProcessor"
            or processor.get("patch_size") != config["vision_config"]["patch_size"]
            or document.get("spatial_merge_size") != config.get("spatial_merge_size")):
        raise ValueError("checkpoint Pixtral processor and model geometry disagree")
    ids = [vision.get(k) for k in ("image_token_id", "image_break_token_id", "image_end_token_id")]
    vocabulary = config["text_config"]["vocab_size"]
    if (any(type(n) is not int or not 0 <= n < vocabulary for n in ids)
            or len(set(ids)) != 3 or config.get("image_token_index") != ids[0]):
        raise ValueError("checkpoint image marker ids disagree")
    merge = document["spatial_merge_size"]
    if type(merge) is not int or merge != 2 or processor["patch_size"] != 14:
        raise ValueError("unsupported checkpoint Pixtral merge/patch geometry")
    return {"merge": merge, "row_markers": 1, "image_token_id": ids[0],
            "row_break_token_id": ids[1], "end_token_id": ids[2]}


def provenance(args, manifest):
    """Revalidate fixture bytes and bind media arithmetic before any resume."""
    root = args.media_root or args.windows.parent
    fixtures = {}
    for window in manifest["windows"]:
        for span in window.get("media", []):
            read_fixture(root, span)
            fixtures[span["fixture"]["path"]] = span["fixture"]["sha256"]
    return {"snapshot": snapshot_identity(args.snapshot), "tower_dtype": args.tower_dtype,
        "fixtures": fixtures, "sources": {
            "adapter": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "qualifier": hashlib.sha256(Path(qualifier().__file__).read_bytes()).hexdigest(),
            "runner": hashlib.sha256(Path(__file__).with_name("layer_major.py").read_bytes()).hexdigest()}}


def snapshot_identity(snapshot):
    """Exactly the serving probe's identity, after verifying official imports."""
    model.source_identity()
    pins = model.SOURCE_PINS
    identity = {"snapshot_revision": snapshot.name, "transformers": model.TRANSFORMERS_VERSION,
        "modeling_sha256": pins["models/mistral_large4/modeling_mistral_large4.py"],
        "vision_modeling_sha256": pins["models/pixtral/modeling_pixtral.py"],
        "image_processing_sha256": pins["models/pixtral/image_processing_pil_pixtral.py"]}
    for key, name in (("config", "config.json"), ("tokenizer", "tokenizer.json"),
                      ("preprocessor", "processor_config.json"), ("index", "consolidated.safetensors.index.json")):
        identity[key + "_sha256"] = hashlib.sha256((snapshot / name).read_bytes()).hexdigest()
    return identity


def validate_spans(manifest, snapshot):
    from fidelity_media import validate_media
    contract = processor_contract(snapshot)
    if manifest.get("image_processor") != contract:
        raise ValueError("manifest image processor differs from checkpoint contract")
    reserved = {contract[k] for k in ("image_token_id", "row_break_token_id", "end_token_id")}
    for window in manifest["windows"]:
        spans, tokens = window.get("media", []), window["tokens"]
        validate_media(spans, tokens, window["roles"], window["score_from"], processor=contract)
        covered = {i for span in spans for i in range(span["start"], span["start"] + span["len"])}
        if {i for i, token in enumerate(tokens) if token in reserved} != covered:
            raise ValueError("unbound checkpoint image marker outside image span")
    return contract


def qualifier():
    """Reuse the existing consolidated-to-official weight mapping."""
    path = Path(__file__).resolve().parents[1] / "tools/qualify/mistral_large4/qualify-vision.py"
    spec = importlib.util.spec_from_file_location("_ml4_media_qualifier", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextmanager
def official_tower_arithmetic():
    import torch
    import torch.nn.functional as functional
    invariant = sys.modules.get("shape_invariant")
    linear, einsum = functional.linear, torch.einsum
    if invariant is not None:
        functional.linear, torch.einsum = invariant._linear, invariant._einsum
    try:
        yield
    finally:
        functional.linear, torch.einsum = linear, einsum


def load_tower(snapshot, dtype="bf16", device="cuda"):
    import torch
    if dtype not in ("bf16", "fp32"):
        raise ValueError("unsupported official tower precision")
    model.source_identity()
    tool = qualifier()
    config = json.loads((snapshot / "config.json").read_bytes())
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        tower, projector, _ = tool.reference(config, tool.read_tower(snapshot),
            torch.bfloat16 if dtype == "bf16" else torch.float32, device=device)
    finally:
        torch.set_default_dtype(previous)
    contract = processor_contract(snapshot)
    marker_bits = tool.marker_rows(snapshot, [contract["row_break_token_id"], contract["end_token_id"]])
    markers = [torch.from_numpy(bits.copy()).view(torch.bfloat16) for bits in marker_bits]
    return tower, projector, markers


def processor(snapshot):
    model.source_identity()
    from transformers.models.pixtral.image_processing_pil_pixtral import PixtralImageProcessorPil
    contract = processor_contract(snapshot)
    config = dict(json.loads((snapshot / "processor_config.json").read_bytes())["image_processor"])
    config.pop("image_processor_type", None)
    # Official PixtralProcessor supplies patch * merge to its image processor.
    config["patch_size"] *= contract["merge"]
    return PixtralImageProcessorPil(**config)


def encode_span(models, image_processor, root, span, *, dtype="bf16"):
    import torch
    from PIL import Image
    tower, projector, markers = models
    device = next(tower.parameters()).device
    with Image.open(io.BytesIO(read_fixture(root, span))) as image:
        prepared = image_processor(images=image.convert("RGB"), return_tensors="pt")
    sizes = torch.as_tensor(prepared["image_sizes"], device=device)
    if sizes.shape != (1, 2) or [1, *(sizes[0].cpu() // 14).tolist()] != span["grid"]:
        raise ValueError("official Pixtral processor grid differs from engine span")
    pixels = prepared["pixel_values"].to(device=device,
        dtype=torch.bfloat16 if dtype == "bf16" else torch.float32)
    with official_tower_arithmetic(), torch.inference_mode(), torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        hidden = tower(pixels, image_sizes=sizes).last_hidden_state
        features = projector(hidden.squeeze(0), sizes)
    h, w, extent = image_extent(span["grid"], {"merge": 2, "row_markers": 1})
    if features.ndim != 2 or features.shape[0] != h * w or span["len"] != extent:
        raise ValueError("official Pixtral projector extent differs from image span")
    rows = features.new_empty((h, w + 1, features.shape[-1]))
    rows[:, :w] = features.reshape(h, w, -1)
    rows[:, w] = markers[0].to(device=device, dtype=features.dtype)
    rows[-1, w] = markers[1].to(device=device, dtype=features.dtype)
    rows = rows.reshape(extent, -1)
    if not torch.isfinite(rows).all():
        raise ValueError("nonfinite official Pixtral features")
    return rows.cpu()


def window_features(args, manifest):
    import torch
    from fidelity_windows import canonical
    validate_spans(manifest, args.snapshot)
    identity = snapshot_identity(args.snapshot)
    dtype = getattr(args, "tower_dtype", "bf16")
    models = load_tower(args.snapshot, dtype)
    features, bindings, entries = {}, {}, []
    try:
        image_processor = processor(args.snapshot)
        for window in manifest["windows"]:
            for span in window.get("media", []):
                binding = (span["grid"], span["fixture"])
                if span["key"] in features:
                    if bindings[span["key"]] != binding:
                        raise ValueError("one ML4 media key binds different fixtures/grids")
                    continue
                value = encode_span(models, image_processor, args.media_root or args.windows.parent, span, dtype=dtype)
                features[span["key"]], bindings[span["key"]] = value, binding
                if args.media_features_out and not getattr(args, "_prefix_probe", False):
                    entries.append(write_features(args.media_features_out, span,
                        value.bfloat16().contiguous().view(torch.uint16).numpy(), tower_dtype=dtype, identity=identity))
    finally:
        torch.cuda.synchronize()
        del models
        torch.cuda.empty_cache()
    if args.media_features_out and not getattr(args, "_prefix_probe", False):
        index = {"schema": "cuteafd.media.features.index/1", "family": manifest["family"],
            "checkpoint": manifest["checkpoint"], "tokenizer_sha256": identity["tokenizer_sha256"],
            "set_sha256": manifest["set_sha256"], "features": [entry["key"] for entry in entries]}
        publish_immutable(args.media_features_out / "features.json", canonical(index) + b"\n")
    return features, identity
