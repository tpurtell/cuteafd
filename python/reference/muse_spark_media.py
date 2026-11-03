"""Pinned bundled Muse towers at the official public feature boundary, no LM."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import io
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

from fidelity_media import (processor_contract, publish_immutable, read_fixture,
                            read_pcm_fixture, validate_media, write_features)
from families.muse_spark import model as text

SOURCE_PINS = {
    "models/muse_spark/feature_extraction_muse_spark.py": "ea6791c32a840d3bfc62aa61b4193a6865dfa7d05d45068a462580feee76adcf",
    "models/muse_spark/processing_muse_spark.py": "880a4f2328605018ccfbee7a7aef564d32781c308c34df0d145f8ce939e80d78",
    "models/muse_glimmer/image_processing_muse_glimmer.py": "827b93a32e8a4b3c80c35395cc261f798a3af3b17665116b0b30d40906a960a9",
    "image_processing_backends.py": "08ab5a408a11b96bc9401a5dc6254b4d4ecdec4f5502a44f452d16814eb46a49",
}


def audio_contract(snapshot):
    config = json.loads((Path(snapshot) / "config.json").read_bytes())
    processor = json.loads((Path(snapshot) / "processor_config.json").read_bytes())["feature_extractor"]
    audio = config["audio_config"]
    stride = processor.get("samples_per_token")
    if (audio.get("model_type") != "muse_spark_audio" or processor.get("sampling_rate") != 24000
            or type(stride) is not int or stride <= 0 or stride != math.prod(audio["patch_sizes"])):
        raise ValueError("audio processor sampling_rate/samples_per_token disagree with declared PCM tower")
    if "samples_per_token" in audio and audio["samples_per_token"] != stride:
        raise ValueError("processor samples_per_token disagrees with audio_config.samples_per_token")
    return {"sampling_rate": processor["sampling_rate"], "samples_per_token": stride}


def validate_spans(manifest, snapshot):
    from tokenizers import Tokenizer
    snapshot = Path(snapshot)
    config = json.loads((snapshot / "config.json").read_bytes())
    if config.get("model_type") != "muse_spark":
        raise ValueError("Muse media requires its own bundled checkpoint towers")
    image, audio = processor_contract(snapshot), audio_contract(snapshot)
    tokenizer = Tokenizer.from_file(str(snapshot / "tokenizer.json"))
    for window in manifest["windows"]:
        validate_media(window.get("media", []), window["tokens"], window["roles"], window["score_from"],
                       processor=image, audio_processor=audio)
        covered = {"image": set(), "audio": set()}
        for span in window.get("media", []):
            kind = span["kind"]
            start, end = span["start"], span["start"] + span["len"]
            begin, finish = (("<|image_start|>", "<|image_end|>") if kind == "image"
                             else ("<|soft_audio_start|>", "<|soft_audio_end|>"))
            if (start < 1 or end >= len(window["tokens"])
                    or window["tokens"][start - 1] != tokenizer.token_to_id(begin)
                    or window["tokens"][end] != tokenizer.token_to_id(finish)
                    or window["tokens"][start:end] != [config[kind + "_token_id"]] * span["len"]):
                raise ValueError("Muse span differs from checkpoint media markers")
            covered[kind].update(range(start, end))
        for kind, positions in covered.items():
            if {i for i, token in enumerate(window["tokens"]) if token == config[kind + "_token_id"]} != positions:
                raise ValueError("Muse placeholder coverage differs from pinned media spans")


def provenance(args, manifest):
    import transformers
    import torchvision
    from packaging.version import Version
    root = Path(transformers.__file__).resolve().parent
    identity = text.source_identity()
    actual = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in SOURCE_PINS}
    if actual != SOURCE_PINS:
        raise ValueError("Muse media processor source differs from pinned official release")
    image = any(s["kind"] == "image" for w in manifest["windows"] for s in w.get("media", []))
    if image and Version(torchvision.__version__.split("+")[0]) < Version("0.27"):
        raise ValueError("official CPU LANCZOS requires torchvision >=0.27; refusing BICUBIC fallback")
    processor = json.loads((args.snapshot / "processor_config.json").read_bytes())
    if image and processor["image_processor"].get("resample") != 1:
        raise ValueError("Muse official processor requires declared LANCZOS filter")
    return {**identity, "media_sources_sha256": actual, "snapshot_revision": args.snapshot.name,
            "media_adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "torchvision": torchvision.__version__, "processor_device": "cpu", "resample": "LANCZOS",
            "public_boundary": "MuseSparkModel.get_image_features/get_audio_features",
            **{name + "_sha256": hashlib.sha256((args.snapshot / filename).read_bytes()).hexdigest()
               for name, filename in (("config", "config.json"), ("processor", "processor_config.json"),
                                      ("tokenizer", "tokenizer.json"), ("index", "model.safetensors.index.json"))}}


@contextmanager
def official_arithmetic():
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


def load_modules(snapshot, kind, dtype):
    import torch
    from transformers.models.muse_spark.configuration_muse_spark import MuseSparkConfig
    from transformers.models.muse_spark.modeling_muse_spark import (
        MuseSparkVisionModel, MuseSparkVisionAdapter, MuseSparkAudioModel)
    config = MuseSparkConfig.from_pretrained(snapshot, local_files_only=True)
    config.vision_config._attn_implementation = "sdpa"
    if config.audio_config.samples_per_token != audio_contract(snapshot)["samples_per_token"]:
        raise ValueError("official audio samples_per_token differs from the checkpoint processor")
    with torch.device("meta"):
        if kind == "image":
            modules = [(MuseSparkVisionModel(config.vision_config), "model.vision_tower."),
                       (MuseSparkVisionAdapter(config), "model.vision_adapter."),
                       (torch.nn.Linear(config.projector_hidden_size, config.text_config.hidden_size, bias=False),
                        "model.vision_projection.")]
        else:
            modules = [(MuseSparkAudioModel(config.audio_config), "model.audio_tower.")]
    checkpoint = text.Checkpoint(snapshot)
    try:
        for module, prefix in modules:
            expected = module.state_dict()
            names = {name.removeprefix(prefix) for name in checkpoint.index if name.startswith(prefix)}
            if names != set(expected):
                raise ValueError(f"official media module tensor inventory differs: {prefix}")
            state = {}
            for name, parameter in expected.items():
                value = checkpoint.get(prefix + name)
                if value.dtype != torch.bfloat16 or value.shape != parameter.shape:
                    raise ValueError(f"official media tensor shape/dtype differs: {prefix}{name}")
                state[name] = value.to(device="cuda", dtype=dtype)
            module.load_state_dict(state, strict=True, assign=True)
            module.eval()
    finally:
        checkpoint.close()
    if kind == "image":
        tower = modules[0][0]
        inv, _ = tower.rotary_emb.compute_axial_rope_parameters(config.vision_config, device="cuda")
        tower.rotary_emb.inv_freq, tower.rotary_emb.original_inv_freq = inv, inv.clone()
    return [module for module, _ in modules], config


def encode_span(modules, config, snapshot, root, span, dtype):
    import torch
    from transformers.models.muse_spark.modeling_muse_spark import MuseSparkModel, MuseSparkRMSNorm
    config.return_dict = True
    norm = MuseSparkRMSNorm(eps=config.text_config.rms_norm_eps, with_scale=False)
    if span["kind"] == "image":
        from PIL import Image
        from transformers.image_utils import load_image
        from transformers.models.muse_glimmer.image_processing_muse_glimmer import MuseGlimmerImageProcessor
        parameters = json.loads((snapshot / "processor_config.json").read_bytes())["image_processor"]
        processor = MuseGlimmerImageProcessor(**{k: v for k, v in parameters.items() if k != "image_processor_type"})
        with Image.open(io.BytesIO(read_fixture(root, span))) as image:
            prepared = processor(images=load_image(image), return_tensors="pt", device="cpu")
        if prepared["image_grid_thw"].tolist() != [span["grid"]]:
            raise ValueError("official image processor grid differs from pinned span")
        owner = SimpleNamespace(config=config, vision_tower=modules[0], vision_adapter=modules[1],
                                vision_projection=modules[2], perception_emb_norm=norm)
        result = MuseSparkModel.get_image_features(owner,
            pixel_values=prepared["pixel_values"].to("cuda", dtype),
            image_grid_thw=prepared["image_grid_thw"].to("cuda"), return_dict=True)
        value = torch.cat(result.pooler_output)
    else:
        from transformers.models.muse_spark.feature_extraction_muse_spark import MuseSparkFeatureExtractor
        parameters = json.loads((snapshot / "processor_config.json").read_bytes())["feature_extractor"]
        processor = MuseSparkFeatureExtractor(**{k: v for k, v in parameters.items() if k != "feature_extractor_type"})
        prepared = processor(read_pcm_fixture(root, span), sampling_rate=parameters["sampling_rate"], return_tensors="pt")
        owner = SimpleNamespace(config=config, audio_tower=modules[0],
                                get_input_embeddings=lambda: SimpleNamespace(embed_norm=norm))
        result = MuseSparkModel.get_audio_features(owner,
            input_values=prepared["input_values"].to("cuda", dtype),
            input_values_mask=prepared["input_values_mask"].to("cuda"), return_dict=True)
        value = result.pooler_output
    value = value.float().cpu()
    if value.shape != (span["len"], config.text_config.hidden_size) or not torch.isfinite(value).all():
        raise ValueError("official Muse public features have invalid shape/values")
    return value


def window_features(args, manifest):
    import torch
    from fidelity_windows import canonical
    validate_spans(manifest, args.snapshot)
    identity = provenance(args, manifest)
    root, dtype_name = args.media_root or args.windows.parent, getattr(args, "tower_dtype", "bf16")
    if dtype_name not in ("bf16", "fp32"):
        raise ValueError("unsupported Muse tower precision")
    dtype = torch.bfloat16 if dtype_name == "bf16" else torch.float32
    features, bindings, entries = {}, {}, []
    with official_arithmetic(), torch.inference_mode():
        for kind in ("image", "audio"):
            spans = [s for w in manifest["windows"] for s in w.get("media", []) if s["kind"] == kind]
            if not spans:
                continue
            modules, config = load_modules(args.snapshot, kind, dtype)
            try:
                for span in spans:
                    binding = canonical({key: value for key, value in span.items() if key != "start"})
                    if span["key"] in features:
                        if bindings[span["key"]] != binding:
                            raise ValueError("one Muse key binds different media fixtures/geometry")
                        continue
                    value = encode_span(modules, config, args.snapshot, root, span, dtype)
                    features[span["key"]], bindings[span["key"]] = value, binding
                    if args.media_features_out and not getattr(args, "_prefix_probe", False):
                        entries.append(write_features(args.media_features_out, span,
                            value.bfloat16().contiguous().view(torch.uint16).numpy(), tower_dtype=dtype_name, identity=identity))
            finally:
                torch.cuda.synchronize()
                del modules
                torch.cuda.empty_cache()
    if args.media_features_out and not getattr(args, "_prefix_probe", False):
        index = {"schema": "cuteafd.media.features.index/1", "family": manifest["family"],
                 "checkpoint": manifest["checkpoint"], "tokenizer_sha256": identity["tokenizer_sha256"],
                 "set_sha256": manifest["set_sha256"], "features": [entry["key"] for entry in entries]}
        publish_immutable(args.media_features_out / "features.json", canonical(index) + b"\n")
    return features, identity
