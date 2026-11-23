"""Stream and seal GLM Flash's three MUL1 core views and shared K4 experts."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import re

import mc_package as package
from core_export import packed_specs
from hy4_flash.export import verify_original_discovery
from inkling2.export import DirectMCWriter as SharedWriter, VIEWS, load_source_inventory
from inkling2.publish import LFS_ATTRIBUTES
from kimi_k3.uniform_k2.storage import atomic_json
from . import source

ASSETS = ("LICENSE", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
          "generation_config.json", "processor_config.json")
REFERENCE_REVISION = "62d7ebd7de4938e072b7aaeb881593b79dc56835"


def storage_entry(module, geometry):
    tensors = {name: dict(torch_dtype=package.DTYPES[dtype], shape=shape)
               for name, (dtype, shape) in packed_specs(module, geometry).items()}
    return package.tensor_storage_entry(geometry, tensors, side=1,
                                       mcg_multiplier=None, mul1_multiplier=package.MUL1)


def view_metadata(document, native, core, experts, identity):
    """Build loader and quantizer declarations for exactly one active core per index."""
    configs, quantizes = {}, {}
    expert_storage = {module: storage_entry(module, g) for module, g in experts.items()}
    layers = sorted({int(name.split(".layers.")[1].split(".")[0]) for name in experts})
    ids = sorted({int(name.split(".experts.")[1].split(".")[0]) for name in experts})
    layer_ids = "|".join(map(str, layers))
    expert_ids = "|".join(map(str, ids))
    pattern = (rf"^model\.language_model\.layers\.(?:{layer_ids})\.mlp\.experts\.(?:{expert_ids})"
               r"\.(?:gate_proj|up_proj|down_proj)$")
    native_metadata = {n: dict(dtype=dtype, shape=shape) for n, (dtype, shape) in native.items()}
    for view in VIEWS:
        index, _, quantize, bits = package.mc_manifest.FILES[view]
        config = deepcopy(document)
        declaration = dict(quant_method="exl3", format="exl3", codebook="mul1",
                           codebook_multiplier=package.MUL1, per_module_metadata=f"{quantize}:tensor_storage")
        config["quantization_config"] = declaration
        storage, retained, patterns = dict(expert_storage), dict(native_metadata), [pattern]
        if view == "original":
            for name, g in core.items():
                retained[name + ".weight"] = dict(dtype="BF16", shape=[g["out_features"], g["in_features"]])
        else:
            storage.update({m: storage_entry(m, dict(g, bits=bits)) for m, g in core.items()})
            patterns.extend("^" + re.escape(m) + "$" for m in sorted(core))
        metadata = package.quantize_config_for_view(declaration, tensor_storage=storage,
            retained_native_tensors=retained, provenance=identity, core_view=view,
            upstream_activations="k8-compact")
        metadata, config = package.gptqmodel_mc_view(config, metadata, bits=4,
            core_bits=None if view == "original" else bits, module_include=patterns, codebook="mul1")
        config["quantization_config"]["safetensors_index"] = index
        if view != "original":
            config["quantization_config"].update(mtp_bits=bits, head_bits=bits)
        config["text_config"]["quantization_config"] = dict(config["quantization_config"])
        metadata.update(config["quantization_config"])
        configs[view], quantizes[view] = config, metadata
    return configs, quantizes


def model_card(identity, view_bytes):
    dataset = identity["dataset"]
    agentic = identity["components"]["agentic"]
    standard = identity["components"]["exl3_standard"]
    sizes = {k: n / (1 << 30) for k, n in view_bytes.items()}
    dsa = identity["diagnostics"]["dsa"]
    return f"""---
library_name: cuteafd
license: mit
base_model: {source.BASE_REPO}
datasets:
- {dataset['repo_id']}
tags:
- exl3
- mixture-of-experts
---

# GLM 5.3 Flash EXL3 K4 MC MUL1

## What multi-core means

Each view shares the K4 routed experts, router, embeddings, vision tower, KDA recurrent tensors, DSA indexer, hyper-connections and absorbed MLA KV projection. The selected core includes the attention matrices, dense and shared MLPs, vocabulary head and MTP core.

| View | Indexed weight payload | When to use it |
| --- | ---: | --- |
| `original` (default) | {sizes['original']:.2f} GiB | Preserve source BF16 core and MTP matrices |
| `k8-compact` | {sizes['k8-compact']:.2f} GiB | Use the compact K8 core that supplied downstream calibration activations |
| `k6-compact` | {sizes['k6-compact']:.2f} GiB | Minimize the core footprint |

## How it was made

The source is `{source.BASE_REPO}@{source.BASE_REVISION}`. The calibration panel includes token-budgeted real source code, worked mathematics, function-call schemas and multilingual text. The GLM chat template rendered each agentic prompt with maximum reasoning effort. Standard ExLlamaV3 rows used raw text and random tokens. DSA selects {dsa['selected_groups']:,} complete groups of {dsa['group_tokens']} tokens with the partial tail retained. Calibration prompts exceeding {dsa['max_tokens_before_first_group_exclusion']:,} tokens total {dsa['sparse_calibration_count']}.

| Recipe | Value |
| --- | --- |
| Codebook | EXL3 MUL1, multiplier `0x83DCD12D`, side 1, automatic output scales |
| Widths | Shared K4 experts, K8 and K6 core including MTP |
| Encoder | `tpurtell/GPTQModel@{identity['code']['fork']}` |
| Search | Seed {identity['seed']}, sigma_reg {identity['sigma']} |
| Hessians | FP32 products over at most 128 rows, FP64 outer sum, FP32 storage, checked FP64 Hadamard congruence |
| Panel | `{dataset['repo_id']}@{dataset['revision']}` |
| Calibration order | {agentic['count']:,} agentic prompts ({agentic['tokens']:,} tokens), then {standard['count']:,} standard rows ({standard['tokens']:,} tokens) |
| Total calibration | {agentic['tokens'] + standard['tokens']:,} tokens |
| Standard input IDs | SHA256 `{standard['input_ids_sha256']}` |
| Replay | Serialized K8 core and K4 expert reconstructions in one causal pass. K6 reuses each core Hessian. |
| Recovery | Keep all natural rows. Below 1,024, fill from ranks 9–16 in rank and corpus order. Substitute the candidate for the last route, renormalize and scale by 2.5. Fill any remaining deficit with raw-XTX identity observations. |
| Combine | Accumulate native expert outputs on device in ascending expert order using FP32. |

See `per-module-errors.csv` for projection error and routed recovery counts.

## Running it

Select this checkpoint in cuteafd. Set `CORE_VIEW` to `original`, `k8-compact` or `k6-compact`. The default safetensors index selects the original view. MC-aware readers discover the compact indexes through `cuteafd-mc.json`.

## License

Z.AI's MIT license accompanies the checkpoint. [License](LICENSE)
"""


class DirectMCWriter(SharedWriter):
    family = "glm5_flash"
    prove_shared_source = True

    def __init__(self, output, source_snapshot, identity, *, shard_target=2_000_000_000):
        if identity.get("codebook") != "mul1" or identity.get("codebook_multiplier") != package.MUL1:
            raise ValueError("GLM Flash requires the frozen MUL1 producer identity")
        audit = source.audit(source_snapshot)
        inventory = load_source_inventory(source_snapshot,
            reviewed_payload_bytes=audit["source_payload_bytes"], reviewed_tensor_count=audit["source_tensors"])
        super().__init__(output, source_snapshot, source.BASE_REPO, source.BASE_REVISION,
                         identity, inventory, shard_target=shard_target)

    def write_group(self, label, *, shared=None, original=None, k8=None, k6=None, packed_hashes=None):
        """Check producer XXH3-128 digests of packed tensor bytes during final shard assembly."""
        names = {"shared": (shared or {}).keys() - self.source_inventory["mapping"].keys(),
                 "k8-compact": set(k8 or ()), "k6-compact": set(k6 or ())}
        packed_hashes = packed_hashes or {}
        if (packed_hashes.keys() - names.keys()
                or any(set(packed_hashes.get(kind, {})) != members for kind, members in names.items())
                or any(not isinstance(digest, str) or not re.fullmatch("[0-9a-f]{32}", digest)
                       for rows in packed_hashes.values() for digest in rows.values())):
            raise ValueError("packed producer digest inventory differs")
        return super().write_group(label, shared=shared, original=original, k8=k8, k6=k6,
                                   packed_hashes=packed_hashes)

    def finalize(self, *, error_csv, card_dir):
        core, experts = source.core_modules(8), source.expert_modules()
        selected = {m + ".weight" for m in core.keys() | experts.keys()}
        native = {n: s for n, s in self.source_inventory["specs"].items() if n not in selected}
        shared = native | {n: spec for m, g in experts.items() for n, spec in packed_specs(m, g).items()}
        sizes = source.payloads()
        configs, quantizes = view_metadata(package.load_json(self.source / "config.json"),
                                           native, core, experts, self.identity)
        card_dir = Path(card_dir).resolve()
        if any(card_dir == path or path in card_dir.parents for path in (self.source, self.output)):
            raise ValueError("card workspace must be outside source and sealed output")
        card_dir.mkdir(parents=True, exist_ok=True)
        card_path = card_dir / "README.md"
        card_path.write_text(model_card(self.identity, sizes["view_payload_bytes"]), encoding="utf-8")
        attributes = card_dir / ".gitattributes"
        attributes.write_bytes(LFS_ATTRIBUTES)
        assets = {n: self.source / n for n in ASSETS}
        receipt = super().finalize(core_selection=core, configs=configs, quantizes=quantizes,
            assets={**assets, ".gitattributes": attributes, "README.md": card_path,
                    "per-module-errors.csv": Path(error_csv)}, expected_shared_specs=shared,
            expected_payload_bytes=sizes["package_payload_bytes"],
            exl3_codebook=dict(name="mul1", multiplier=package.MUL1))
        expected = package.load_json(self.output / "model.safetensors.index.json")["weight_map"]
        discovery = verify_original_discovery(self.output, expected, model_type="glm5_next",
                                              reference_revision=REFERENCE_REVISION)
        receipt["original_discovery"] = discovery
        atomic_json(self.output / "package-validation.json", receipt)
        return receipt
