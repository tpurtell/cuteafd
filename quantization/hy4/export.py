"""Publish full Hy4 through the shared three-view MUL1 writer."""
from pathlib import Path

from hy4_flash import export
from . import recipe


def verify_original_discovery(output, expected_mapping):
    return export.verify_original_discovery(output, expected_mapping, model_type="hy_v4_vl",
                                            reference_revision=recipe.REFERENCE_REVISION)


def model_card(path, engram, *, k6_drafter_bits, selection, identity, view_bytes):
    dataset = identity["dataset"]
    components = identity["components"]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = "\n".join(f"| {name} | {view_bytes[name] / 2**30:.2f} GiB |" for name in export.VIEWS)
    path.write_text(f"""---
library_name: cuteafd
license: other
license_name: tencent-hy-license
license_link: LICENSE
base_model: tencent/Hy4
datasets:
- {dataset['repo_id']}
tags:
- exl3
- mixture-of-experts
---

# Hy4 EXL3 K4 MC

The default index preserves Tencent's core and DFly2 weights in their source dtypes.
All views share the K4 routed experts, router, indexer, embeddings, vision tower
and native DFly2 components. The compact views encode eligible target matrices
at K8 or K6. The K8 view uses a K8 drafter, and the K6 view uses a K{k6_drafter_bits} drafter.

| Core view | Indexed tensor payload |
| --- | ---: |
{rows}

## Quantization recipe

| Setting | Value |
| --- | --- |
| Source | {recipe.BASE_REPO}@{recipe.BASE_REVISION} |
| Codebook | MUL1 (0x83DCD12D), side 1, automatic output scales |
| Precision | K4 routed experts, K8 and K6 compact core |
| Encoder | tpurtell/GPTQModel@{identity['code']['fork']} |
| Calibration | {dataset['repo_id']}@{dataset['revision']} |
| Chat prompts | {components['agentic']['tokens']:,} tokens, high reasoning effort |
| ExLlamaV3 standard | {components['exl3_standard']['tokens']:,} tokens |
| Search | Seed {identity['seed']}, sigma {identity['sigma']} |
| Coverage | All natural routes, then ranks 9–16 and identity padding to a floor of 1,024 |
| Expert combine | Native top-8 routes, ordered FP32 sum |

Calibration used token-budgeted real source code, worked mathematics, function-call
schemas and multilingual text. One causal pass propagated serialized K8 core and K4
expert reconstructions. K6 used the same Hessians. The five DFly2 layers use target
KV and indexer selections from layers 61, 65, 69, 73 and 77. The long-context witness
exercised DSA selection beyond 2,048 tokens only in reference parity.

The per-module errors and coverage counts are in [per-module-errors.csv](per-module-errors.csv).
Serving qualification is tracked in [CuteAFD #2550](https://github.com/tpurtell/cuteafd/issues/2550).
The [Tencent HY License](LICENSE) accompanies the checkpoint.
""", encoding="utf-8")
    return path


class DirectMCWriter(export.DirectMCWriter):
    family = "hy4"
    recipe = recipe
    card = staticmethod(model_card)
