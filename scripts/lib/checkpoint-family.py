#!/usr/bin/env python3
"""The launch family table: which family a checkpoint is, and which decoder
layers carry routed experts.

  checkpoint-family.py CONFIG_JSON   -> "FAMILY MODEL_TYPE FIRST_LAYER LAST_LAYER"

FAMILY is the long family id (deepseek_v41, deepseek_v4, glm5, glm5_flash,
mimo_v2, qwen4). The Spark ranks serve routed experts on [FIRST, LAST]
(LAST -1: through the end); MTP/nextn layers past the decoder stay off them for
GLM 5.3 Flash and Qwen 3.8, whose serve paths do not run their MTP heads.

The layer patterns are read as the runtimes' config readers read them
(cuteafd-loader families/*/config.rs, plan/launch.rs): both spellings of a
pattern, each covering exactly every layer, agreeing when both are present.
rust/crates/cuteafd-loader/tests/fixtures/launch-families.json holds the cases
both implementations are checked against.
"""
from __future__ import annotations

import json
import math
import re
import sys

FAMILIES = {
    "deepseek_v41": "deepseek_v41",
    "deepseek_v4": "deepseek_v4",
    "glm_moe_dsa": "glm5",
    "glm5_next": "glm5_flash",
    "mimo_v2_flash": "mimo_v2",
    "mimo_v2": "mimo_v2",
    "qwen4_exp": "qwen4",
}


def _pattern(text: dict, layers: int, spellings: dict) -> list | None:
    """One per-layer pattern in either spelling (key -> {entry: value}); None when absent."""
    found = None
    for key, values in spellings.items():
        raw = text.get(key)
        if raw is None:
            continue
        if not isinstance(raw, list) or len(raw) != layers:
            raise SystemExit(f"{key} must be a list of {layers} entries")
        # Entries match by type too (JSON true is not 1, 1.0 is not 1).
        typed = {(type(entry), entry): value for entry, value in values.items()}
        try:
            decoded = [typed[(type(entry), entry)] for entry in raw]
        except (KeyError, TypeError):
            raise SystemExit(f"{key} has an unknown entry") from None
        if found is not None and found[1] != decoded:
            raise SystemExit(f"{found[0]} and {key} disagree")
        found = (key, decoded)
    return None if found is None else found[1]


def _dense_prefix(text: dict, layers: int) -> int:
    """GLM 5.x: dense layers are a prefix (mlp_layer_types or first_k_dense_replace)."""
    first_k = text.get("first_k_dense_replace")
    dense = _pattern(text, layers, {"mlp_layer_types": {"dense": True, "sparse": False}})
    if dense is None:
        if first_k is None:
            raise SystemExit("config lacks first_k_dense_replace")
        return int(first_k)
    first = dense.index(False) if False in dense else layers
    if any(dense[first:]):
        raise SystemExit("mlp_layer_types: dense layers must be a prefix")
    if first_k is not None and first_k != first:
        raise SystemExit("mlp_layer_types and first_k_dense_replace disagree")
    return first


def describe(config: dict) -> tuple[str, str, int, int]:
    text = config.get("text_config", config)
    kind = config.get("model_type", "?")
    family = FAMILIES.get(kind)
    if family is None:
        raise SystemExit(f"unsupported model_type {kind!r}; known: {', '.join(sorted(FAMILIES))}")
    first, last = 0, -1
    if family in ("glm5", "glm5_flash", "mimo_v2", "qwen4"):
        layers = text.get("num_hidden_layers")
        if not isinstance(layers, int) or layers <= 0:
            raise SystemExit("config lacks num_hidden_layers")
    if family == "glm5":
        first = _dense_prefix(text, layers)
    elif family == "glm5_flash":
        kinds = {"linear_attention": "kda", "deepseek_sparse_attention": "mla"}
        if _pattern(text, layers, {"layer_types": kinds}) is None:
            raise SystemExit("config lacks layer_types")
        # Any per-layer pattern (serve-glmf reads dense[layer]); first_k_dense_replace agrees when present.
        first_k = text.get("first_k_dense_replace")
        dense = _pattern(text, layers, {"mlp_layer_types": {"dense": True, "sparse": False}})
        if dense is None:
            dense = [layer < (first_k or 0) for layer in range(layers)]
        elif first_k is not None and dense != [layer < first_k for layer in range(layers)]:
            raise SystemExit("mlp_layer_types and first_k_dense_replace disagree")
        first = dense.index(False) if False in dense else layers
        last = layers - 1
    elif family == "mimo_v2":
        dense = _pattern(text, layers, {"moe_layer_freq": {0: True, 1: False},
                                        "mlp_layer_types": {"dense": True, "sparse": False}})
        if dense is None:
            dense = [layer == 0 for layer in range(layers)]
        first = dense.index(False) if False in dense else layers
    elif family == "qwen4":
        kinds = {"linear_attention": "gdn", "full_attention": "full"}
        if _pattern(text, layers, {"layer_types": kinds}) is None:
            raise SystemExit("config lacks layer_types")
        last = layers - 1
    return family, kind, first, last


def routed_layers(config: dict) -> int | None:
    """Count actual sparse layers, not the decoder extent or its first index."""
    text = config.get("text_config", config)
    layers = text.get("num_hidden_layers", text.get("n_layers"))
    if layers is None:
        return None
    if type(layers) is not int or layers <= 0:
        raise SystemExit("invalid num_hidden_layers")
    family = FAMILIES.get(config.get("model_type"))
    if family in ("deepseek_v41", "deepseek_v4", "qwen4"):
        return layers
    if family == "glm5":
        first = _dense_prefix(text, layers)
        if not 0 <= first <= layers:
            raise SystemExit("invalid first_k_dense_replace")
        return layers - first
    if family == "glm5_flash":
        # Reuse the runtime-equivalent pattern checks, including agreement.
        describe(config)
    dense = _pattern(text, layers, {"mlp_layer_types": {"dense": True, "sparse": False},
                                    "moe_layer_freq": {0: True, 1: False}})
    if dense is not None:
        return dense.count(False)
    first = text.get("first_k_dense_replace", 0) if family == "glm5_flash" else 1
    if type(first) is not int or not 0 <= first <= layers:
        raise SystemExit("invalid first_k_dense_replace")
    return layers - first


def onboard(value: str, config: dict | None = None) -> str:
    accepted = "RTX_EXPERT_LAYERS must be auto, max, all, N or N% (0..100%)"
    count = routed_layers(config) if config is not None else None
    if value in ("auto", "max"):
        return value
    if value == "all":
        return str(count) if count is not None else value
    if re.fullmatch(r"[0-9]+", value):
        number = int(value)
        if count is not None and number > count:
            raise SystemExit(f"{accepted}; checkpoint has {count} routed layers")
        return str(number)
    if re.fullmatch(r"[0-9]+(?:\.[0-9]+)?%", value):
        percent = float(value[:-1])
        if math.isfinite(percent) and percent <= 100:
            return str(math.floor(count * percent / 100 + 0.5)) if count is not None else value
    raise SystemExit(accepted)


def main() -> int:
    if len(sys.argv) in (3, 4) and sys.argv[1] == "--onboard":
        config = None
        if len(sys.argv) == 4:
            with open(sys.argv[3], encoding="utf-8") as handle:
                config = json.load(handle)
        print(onboard(sys.argv[2], config))
        return 0
    if len(sys.argv) != 2:
        print(__doc__.strip().splitlines()[2].strip(), file=sys.stderr)
        return 2
    with open(sys.argv[1], encoding="utf-8") as handle:
        print(*describe(json.load(handle)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
