"""Pinned official Qwen vision arithmetic and native-id M-RoPE for media goldens."""
from __future__ import annotations

import hashlib
import io
import importlib
import json
from pathlib import Path
import subprocess
import types

from fidelity_media import publish_immutable, read_fixture, validate_media, write_features
from fidelity_windows import canonical

TRANSFORMERS_REVISION = "62d7ebd7de4938e072b7aaeb881593b79dc56835"
MODELING_SHA256 = "2a44aeadb215acbb5c75939fcc97e9f14bccff5a51c232826427594993f6a760"
IMAGE_ID, START_ID, END_ID = 248056, 248053, 248054
WIDTH = 2560
QSA_KEY_ROWS = 2560
PROCESSOR_SOURCES = {
    "models.qwen2_vl.image_processing_pil_qwen2_vl": "f7403c897abd3c0b3ed678a3a208233cb4ab21fc030d45b8e16e6153de6a4e87",
    "image_utils": "24e1b8f65481a87f2d294473b238fe7ac3a5dbdf8fe64c4efac72a3616c66437",
    "image_transforms": "260fd1b2d3b23811f6be81da20777105d522c0213030580e6cf717a34492ef3b",
    "image_processing_backends": "5f3176903638125ee01af36e8c74741a7bab19581f3c13349a775b093cb1bbb0",
    "image_processing_utils": "3328a0e2b38e272e824cad64a50d3e84206dab13d31b96104afa6459e149f7c1",
    "image_processing_base": "c37174598b741995c505ff315c1562e8a7e9e0a216cad53a56615e103af0327a",
    "processing_utils": "53ffe4236eca38eea4142315c0b98bfebc18859812df620b136bac928723d6b5",
    "models.qwen4_exp.configuration_qwen4_exp": "b78132d8cd935437208ee281fa4569b771a63fcb58ebffe84f3e62f5b86235ca",
}


def verify_sources(ref):
    """Check actual imports and their clean checkout before claiming a full revision."""
    source = Path(ref.__file__).resolve()
    root = source.parents[2]
    checkout = root.parent.parent
    try:
        revision = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True, timeout=10).strip()
        changes = subprocess.check_output(["git", "-C", str(checkout), "status", "--porcelain",
                                           "--untracked-files=no"], text=True, timeout=10)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as error:
        raise ValueError("Qwen media requires the pinned Transformers source checkout") from error
    if revision != TRANSFORMERS_REVISION or changes:
        raise ValueError("Qwen media requires the clean pinned Transformers source checkout")
    for name, expected in PROCESSOR_SOURCES.items():
        module = importlib.import_module("transformers." + name)
        actual = Path(module.__file__).resolve()
        if actual != root / (name.replace(".", "/") + ".py") or hashlib.sha256(actual.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Qwen media requires the pinned imported {name} source")
    # The PIL smart-resize implementation is copied from the Torchvision variant;
    # check its source too without importing the optional Torchvision dependency.
    copied = root / "models/qwen2_vl/image_processing_qwen2_vl.py"
    if hashlib.sha256(copied.read_bytes()).hexdigest() != "4e1da45f9e7e157ca08aa88243fcc1495ea6a333e40e0c5035ef17ef2747a97d":
        raise ValueError("Qwen media requires the pinned Qwen image processor source")


def official_reference():
    from transformers.models.qwen4_exp import modeling_qwen4_exp as ref
    digest = hashlib.sha256(Path(ref.__file__).read_bytes()).hexdigest()
    if digest != MODELING_SHA256:
        raise ValueError("Qwen media requires the pinned modeling_qwen4_exp.py")
    verify_sources(ref)
    return ref


def snapshot_identity(snapshot):
    ref = official_reference()
    return {"snapshot_revision": snapshot.name,
            "transformers_revision": TRANSFORMERS_REVISION,
            "modeling_sha256": hashlib.sha256(Path(ref.__file__).read_bytes()).hexdigest(),
            **{key + "_sha256": hashlib.sha256((snapshot / filename).read_bytes()).hexdigest()
               for key, filename in (("config", "config.json"), ("tokenizer", "tokenizer.json"),
                                     ("preprocessor", "preprocessor_config.json"))}}


def validate_window(window):
    """Reject cache keys, other families' ids, unbound placeholders and malformed brackets."""
    tokens, spans = window["tokens"], window.get("media", [])
    validate_media(spans, tokens, window["roles"], window["score_from"])
    covered = set()
    starts, ends = [], []
    for span in spans:
        start, end = span["start"], span["start"] + span["len"]
        if (start == 0 or end >= len(tokens) or tokens[start - 1] != START_ID
                or tokens[end] != END_ID or any(t != IMAGE_ID for t in tokens[start:end])):
            raise ValueError("Qwen image span requires native 248053/248056/248054 ids")
        if span["len"] > 4096:
            raise ValueError("Qwen image span exceeds the 4096-token cap")
        covered.update(range(start, end))
        starts.append(start - 1)
        ends.append(end)
    if ({i for i, t in enumerate(tokens) if t == IMAGE_ID} != covered
            or [i for i, t in enumerate(tokens) if t == START_ID] != starts
            or [i for i, t in enumerate(tokens) if t == END_ID] != ends):
        raise ValueError("Qwen native image tokens differ from the pinned span list")
    return spans


def rope_positions(window, snapshot, *, device="cuda"):
    """Use the official multimodal indexing methods without constructing the full LM."""
    import torch
    ref = official_reference()
    spans = validate_window(window)
    config = json.loads((snapshot / "config.json").read_text())
    if (config.get("image_token_id"), config.get("vision_start_token_id"),
            config.get("vision_end_token_id")) != (IMAGE_ID, START_ID, END_ID):
        raise ValueError("unsupported Qwen native image ids")
    if config["vision_config"]["spatial_merge_size"] != 2:
        raise ValueError("unsupported Qwen spatial merge size")
    owner = types.SimpleNamespace(config=types.SimpleNamespace(
        vision_config=types.SimpleNamespace(spatial_merge_size=2)))
    owner.get_vision_position_ids = types.MethodType(ref.Qwen4ExpModel.get_vision_position_ids, owner)
    ids = torch.tensor([window["tokens"]], dtype=torch.long, device=device)
    modality = torch.zeros_like(ids)
    for span in spans:
        modality[:, span["start"]:span["start"] + span["len"]] = 1
    grids = torch.tensor([s["grid"] for s in spans], dtype=torch.long, device=device).reshape(-1, 3)
    vision_positions, _ = ref.Qwen4ExpModel.get_rope_index(owner, ids, modality, image_grid_thw=grids)
    # Generation's four-row contract: causal/text row indices, then official T/H/W.
    text_positions = torch.arange(ids.shape[1], device=device).view(1, 1, -1)
    return torch.cat([text_positions, vision_positions], dim=0)


def inject_embeddings(embedding, window, features, hc_count):
    """Replace single-width embeddings first; PLE still receives the untouched native ids."""
    import torch
    if embedding.shape != (1, len(window["tokens"]), WIDTH) or hc_count != 4:
        raise ValueError("unsupported Qwen embedding/HC geometry")
    for span in validate_window(window):
        value = features[span["key"]]
        if value.shape != (span["len"], WIDTH) or value.dtype != torch.bfloat16 or not torch.isfinite(value).all():
            raise ValueError("invalid Qwen BF16 feature shape/dtype/values")
        embedding[0, span["start"]:span["start"] + span["len"]].copy_(value.to(embedding.device))
    return embedding.repeat(1, 1, hc_count)


def validate_qsa_key_rows(rows):
    if type(rows) is not int or not 0 < rows <= 16384 or rows % 128:
        raise ValueError("fixed QSA keys must be a positive multiple of 128 <= 16384")
    return rows


def qsa_key_rows(manifest, requested=None):
    """Resolve one sealed extent for admission, prefix probes and eager math."""
    configured = manifest.get("qsa_key_rows", QSA_KEY_ROWS)
    if requested is not None:
        validate_qsa_key_rows(requested)
        if "qsa_key_rows" in manifest and requested != configured:
            raise ValueError(f"QSA CLI rows {requested} differ from configured set rows {configured}")
        configured = requested
    validate_qsa_key_rows(configured)
    observed = max(len(w["tokens"]) for w in manifest["windows"])
    if observed > configured:
        raise ValueError(f"panel exceeds fixed QSA geometry: observed {observed} rows > configured {configured} rows")
    return configured


def configure_qsa_eager(ref, manifest, requested=None):
    rows = qsa_key_rows(manifest, requested)
    current = ref.eager_attention_forward
    installed = getattr(current, "_qsa_key_rows", None)
    if installed is not None and installed != rows:
        raise ValueError(f"QSA factory rows {installed} differ from configured set rows {rows}")
    if installed is None:
        ref.eager_attention_forward = fixed_qsa_eager(current, rows)
    return rows


def fixed_qsa_eager(original, rows=QSA_KEY_ROWS):
    """Fix the PV reduction extent while calling unchanged official eager math."""
    validate_qsa_key_rows(rows)

    def forward(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        if module.training or dropout != 0.0:
            raise ValueError("fixed QSA geometry requires inference without dropout")
        if (query.ndim != 4 or key.ndim != 4 or value.shape != key.shape
                or query.shape[0] != key.shape[0] or query.shape[-1] != key.shape[-1]):
            raise ValueError("invalid QSA query/key/value geometry")
        count = key.shape[-2]
        if not 0 < count <= rows:
            raise ValueError(f"key sequence exceeds fixed QSA geometry: observed {count} rows > configured {rows} rows")
        if (attention_mask is None or attention_mask.ndim != 4
                or not attention_mask.is_floating_point()
                or attention_mask.shape[-1] != count
                or attention_mask.shape[-2] not in (1, query.shape[-2])):
            raise ValueError("fixed QSA geometry requires the full additive causal/indexer mask")
        if count != rows:
            shape = (*key.shape[:-2], rows, key.shape[-1])
            padded_key, padded_value = key.new_zeros(shape), value.new_zeros(shape)
            padded_key[..., :count, :].copy_(key)
            padded_value[..., :count, :].copy_(value)
            mask = attention_mask.new_full((*attention_mask.shape[:-1], rows), -float("inf"))
            mask[..., :count].copy_(attention_mask)
            key, value, attention_mask = padded_key, padded_value, mask
        return original(module, query, key, value, attention_mask,
                        scaling=scaling, dropout=dropout, **kwargs)

    forward._qsa_key_rows = rows
    return forward


def fixed_gdn_forward(original, rows):
    """Diagnostic no-cache prefill geometry; keep the official forward unchanged."""
    if type(rows) is not int or rows <= 0 or rows > 16384 or rows % 64:
        raise ValueError("fixed GDN rows must be a positive multiple of 64 <= 16384")

    def forward(hidden_states, cache_params=None, attention_mask=None, **kwargs):
        import torch
        if cache_params is not None or attention_mask is not None or kwargs:
            raise ValueError("fixed GDN geometry supports only unpadded, unpacked no-cache prefill")
        if hidden_states.ndim != 3 or hidden_states.shape[0] != 1:
            raise ValueError("fixed GDN geometry requires one batch of hidden states")
        count, width = hidden_states.shape[1:]
        if not 0 < count <= rows:
            raise ValueError("sequence exceeds fixed GDN geometry")
        # Pad before projections/conv/softplus, not after the decay was computed.
        padded = hidden_states.new_zeros((1, rows, width))
        padded[:, :count].copy_(hidden_states)
        mask = hidden_states.new_zeros((1, rows), dtype=torch.bool)
        mask[:, :count] = True
        output = original(padded, cache_params=None, attention_mask=mask)
        if output.shape != padded.shape or output.dtype != hidden_states.dtype:
            raise ValueError("official GDN output differs from the fixed geometry/dtype")
        # Never publish the continuation or its recurrent state to a caller.
        return output[:, :count].contiguous()

    return forward


def load_tower(snapshot):
    import torch
    from safetensors import safe_open
    from transformers import AutoConfig
    ref = official_reference()
    config = AutoConfig.from_pretrained(snapshot).vision_config
    if (config.hidden_size, config.out_hidden_size, config.depth, config.num_heads,
            config.spatial_merge_size) != (1152, WIDTH, 27, 16, 2):
        raise ValueError("unsupported official Qwen tower geometry")
    config._attn_implementation = "sdpa"
    default = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        model = ref.Qwen4ExpVisionModel(config).to(torch.bfloat16)
    finally:
        torch.set_default_dtype(default)
    expected = dict(model.named_parameters())
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
    names = {name for name in index if name.startswith("model.visual.")}
    if names != {"model.visual." + name for name in expected} or len(names) != 333:
        raise ValueError("official Qwen tower tensor set differs (expected 333 parameters)")
    with torch.no_grad():
        for shard in sorted({index[name] for name in names}):
            with safe_open(str(snapshot / shard), framework="pt", device="cpu") as reader:
                for name in sorted(n for n in names if index[n] == shard):
                    key = name.removeprefix("model.visual.")
                    value = reader.get_tensor(name)
                    if value.dtype != torch.bfloat16 or value.shape != expected[key].shape:
                        raise ValueError(f"official Qwen tower tensor dtype/shape differs: {name}")
                    expected[key].copy_(value)
    return model.cuda().eval()


def processor(snapshot):
    official_reference()
    from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import Qwen2VLImageProcessorPil
    config = json.loads((snapshot / "preprocessor_config.json").read_text())
    config.pop("image_processor_type", None)
    config.pop("processor_class", None)
    limit = 4096 * config["patch_size"] ** 2 * config["merge_size"] ** 2
    size = config.get("size", {})
    config["max_pixels"] = min(config.get("max_pixels", size.get("longest_edge", limit)), limit)
    return Qwen2VLImageProcessorPil(**config)


def encode_span(model, image_processor, root, span):
    import sys
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from transformers.image_utils import load_image
    with Image.open(io.BytesIO(read_fixture(root, span))) as source:
        # Official load_image applies EXIF orientation, then drops alpha to RGB.
        prepared = image_processor(images=load_image(source), return_tensors="pt")
    if prepared["image_grid_thw"].tolist() != [span["grid"]]:
        raise ValueError("official Qwen PIL processor grid differs from engine span")
    invariant = sys.modules.get("shape_invariant")
    linear, einsum = F.linear, torch.einsum
    if invariant is not None:
        F.linear, torch.einsum = invariant._linear, invariant._einsum
    try:
        with torch.inference_mode(), torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            output = model(prepared["pixel_values"].cuda().to(torch.bfloat16),
                           prepared["image_grid_thw"].cuda(), return_dict=True)
            features = output.pooler_output
    finally:
        F.linear, torch.einsum = linear, einsum
    if features.shape != (span["len"], WIDTH) or features.dtype != torch.bfloat16 or not torch.isfinite(features).all():
        raise ValueError("invalid official Qwen tower feature rows")
    return features.cpu()


def window_features(args, manifest):
    import torch
    root = args.media_root or args.windows.parent
    identity = snapshot_identity(args.snapshot)
    for window in manifest["windows"]:
        validate_window(window)
    model, image_processor = load_tower(args.snapshot), processor(args.snapshot)
    features, bindings, entries = {}, {}, []
    try:
        for window in manifest["windows"]:
            for span in window.get("media", []):
                binding = (span["grid"], span["fixture"])
                if span["key"] in features:
                    if bindings[span["key"]] != binding:
                        raise ValueError("one Qwen media key binds different fixtures/grids")
                    continue
                output = encode_span(model, image_processor, root, span)
                bindings[span["key"]], features[span["key"]] = binding, output
                if args.media_features_out and not getattr(args, "_prefix_probe", False):
                    entries.append(write_features(args.media_features_out, span,
                        output.contiguous().view(torch.uint16).numpy(), tower_dtype="bf16", identity=identity))
    finally:
        del model
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    if args.media_features_out and not getattr(args, "_prefix_probe", False):
        index = {"schema": "cuteafd.media.features.index/1", "family": "qwen4",
                 "checkpoint": manifest["checkpoint"], "tokenizer_sha256": identity["tokenizer_sha256"],
                 "set_sha256": manifest["set_sha256"], "features": [entry["key"] for entry in entries]}
        path = args.media_features_out / "features.json"
        encoded = canonical(index) + b"\n"
        publish_immutable(path, encoded)
    return features, identity
