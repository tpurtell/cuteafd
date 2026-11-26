"""Direct original/K8/K6 export from the audited ERNIE source and sealed encodes."""
from copy import deepcopy
from pathlib import Path
import re

import mc_package as mc
from core_export import packed_specs
from inkling2.export import DirectMCWriter as BaseWriter, VIEWS
from inkling2.publish import LFS_ATTRIBUTES
from .source import BASE_REPO, BASE_REVISION


ASSETS = ("LICENSE", "generation_config.json", "tokenizer_config.json", "tokenizer.json",
          "encoding_ernie.py", "decoding_ernie.py", "tokenization_ernie.py")
PUBLISHED = "wrldsuksgo2mars/ERNIE-5.5-Flash-Preview-EXL3-K4-MC"


def projection_inventory(source):
    native, core, experts = {}, {}, {}
    if source.roles.keys() != source.specs.keys():
        raise ValueError("source roles differ from audited tensor inventory")
    for name, role in source.roles.items():
        dtype, shape = source.specs[name]
        if role in ("native", "flashindexer"):
            native[name] = (dtype, shape)
            continue
        if role not in ("core", "routed_expert") or len(shape) != 2 or dtype != "BF16":
            raise ValueError(f"unsupported projection role or geometry: {name}")
        module = name.removesuffix(".weight")
        if mc.original_tensor_name("ernie55_flash", module) != name:
            raise ValueError(f"unknown source projection alias: {name}")
        outputs, inputs = shape
        routed = role == "routed_expert"
        group = "routed" if routed else ("mtp" if module.startswith("model.mtp.") else "core")
        geometry = dict(in_features=inputs, out_features=outputs,
                        stored_out_features=((outputs + 127) // 128) * 128,
                        bits=4 if routed else 8, group=group, codebook="mul1")
        packed_specs(module, geometry)
        (experts if routed else core)[module] = geometry
    return native, core, experts


def view_metadata(source, native, core, experts, identity):
    configs, quantizes = {}, {}
    # The complete role audit fixes the layer ranges, expert IDs and names.
    experts_pattern = (r"^model\.(?:layers\.(?:[1-9]|[1-3][0-9]|4[0-2])|mtp\.[0-2])"
                       r"\.block_sparse_moe\.experts\.(?:[0-9]|[1-9][0-9]|[1-4][0-9]{2}|50[0-9]|51[01])\.w[123]$")
    for view in VIEWS:
        index, _, metadata_name, bits = mc.mc_manifest.FILES[view]
        compact = view != "original"
        modules = dict(experts)
        retained = dict(native)
        patterns = [experts_pattern]
        if compact:
            modules.update({name: dict(g, bits=bits) for name, g in core.items()})
            patterns.extend("^" + re.escape(name) + "$" for name in sorted(core))
        else:
            for name in core:
                key = mc.original_tensor_name("ernie55_flash", name)
                retained[key] = source.specs[key]
        config = deepcopy(source.document)
        declaration = dict(quant_method="exl3", format="exl3", per_module_metadata=f"{metadata_name}:tensor_storage")
        config["quantization_config"] = declaration
        storage = {}
        for module, g in modules.items():
            tensors = {key: dict(shape=shape, torch_dtype=mc.DTYPES[dtype])
                       for key, (dtype, shape) in packed_specs(module, g).items()}
            storage[module] = mc.tensor_storage_entry(g, tensors, side=1, mcg_multiplier=None,
                                                       mul1_multiplier=mc.MUL1)
        meta = dict(tensor_storage=storage, retained_native_tensors={
            name: dict(dtype=dtype, shape=shape) for name, (dtype, shape) in retained.items()},
            core_view=view, upstream_activations="k8-compact", provenance=identity)
        meta, config = mc.gptqmodel_mc_view(config, meta, bits=4, core_bits=bits if compact else None,
                                             module_include=patterns, codebook="mul1")
        for declaration in (config["quantization_config"], meta):
            declaration["safetensors_index"] = index
            if compact:
                declaration["drafter_bits"] = bits
        configs[view], quantizes[view] = config, meta
    return configs, quantizes


def model_card(view_bytes, identity):
    components = identity["components"]
    return f"""---
library_name: cuteafd
license: apache-2.0
base_model: {BASE_REPO}
datasets:
- {identity['dataset']['repo_id']}
tags:
- exl3
- mixture-of-experts
---

# ERNIE 5.5 Flash Preview EXL3 K4 MC

## What multi-core means

All views share one K4 copy of the target and dSpark routed experts. The embedding,
LM head, routers, indexers and dSpark special heads retain their source bytes.

| View | Indexed payload | Use |
| --- | ---: | --- |
| original (default) | {view_bytes['original']/2**30:.2f} GiB | Native target and dSpark core |
| k8-compact | {view_bytes['k8-compact']/2**30:.2f} GiB | Compact K8 target and dSpark core |
| k6-compact | {view_bytes['k6-compact']/2**30:.2f} GiB | Smallest stored core |

## How it was made

The source is `{BASE_REPO}@{BASE_REVISION}`. Baidu's expert QAT targets MXFP4 W4A8.
This checkpoint encodes its BF16 weights using EXL3. The panel uses token-budgeted
real source code, worked mathematics, function-call schemas and multilingual text.
ERNIE chat rendering uses thinking mode with high reasoning effort. ExLlamaV3
standard rows follow in raw-token order. Long-context witnesses are reserved for
parity checks. Calibration uses all 64 DSA indexer heads with flash filtering off.

| Recipe | Setting |
| --- | --- |
| Codebook | MUL1 0x83DCD12D, whole projection side 1, automatic output scales |
| Widths | K4 experts, K8 and K6 core from the same Hessian |
| Seed and regularizer | 787 and 0.025 × mean diagonal |
| Covariance | FP32 products of up to 128 rows, FP64 sum, FP32 storage |
| Replay | Serialized BF16 reconstructions of K8 core and K4 experts feed the causal pass |
| Calibration | {components['agentic']['tokens']:,} agentic tokens followed by {components['exl3_standard']['tokens']:,} standard tokens |
| Recovery | All natural top-10 routes, then ranks 11–20 with learnable expert scales and renormalized gamma, then raw-XTX identity observations to reach 1,024 |
| Combine | Ascending expert order in FP32, then latent normalization and fc2 |

## Running it

Load the default index for the native core. MC-aware loaders select a compact
index through `cuteafd-mc.json`. Serving requires ERNIE 5.5 and EXL3 MUL1 support.

## License

The [Apache License 2.0](LICENSE) accompanies the checkpoint.
"""


def verify_original_discovery(output):
    """Resolve the default index through the pinned offline Transformers file loader."""
    from transformers.modeling_utils import _get_resolved_checkpoint_files

    mapping = mc.load_json(output / "model.safetensors.index.json")["weight_map"]
    files, metadata = _get_resolved_checkpoint_files(
        output, variant=None, gguf_file=None, use_safetensors=True, user_agent=None,
        is_remote_code=False, download_kwargs={"local_files_only": True})
    if (not metadata or metadata["weight_map"] != mapping
            or {Path(path) for path in files} != {output / name for name in mapping.values()}):
        raise ValueError("plain loader did not resolve the complete default ERNIE view")
    return dict(shards=len(files), tensors=len(mapping))


class DirectMCWriter(BaseWriter):
    family = "ernie55_flash"
    prove_shared_source = True

    def __init__(self, output, source, identity, **kwargs):
        if identity.get("schema") != "ernie55-causal-mc-v1":
            raise ValueError("expected a frozen ERNIE producer identity")
        inputs = identity["inputs"]
        recipe = inputs["recipe"]
        if (recipe.get("codebook") != "mul1" or recipe.get("codebook_multiplier") != mc.MUL1
                or inputs["source_revision"] != BASE_REVISION):
            raise ValueError("expected pinned source and frozen MUL1 recipe")
        # Preserve the full producer receipt; mirror the declarations consumed
        # by the common MC validator and by the generated model card.
        identity = dict(producer=identity, codebook=recipe["codebook"], codebook_multiplier=mc.MUL1,
                        dataset=inputs["corpus"]["dataset"], components=inputs["corpus"]["components"])
        self.header = source
        inv = source.as_inventory()
        total = source.payload_bytes
        inv["payload_totals"] = dict(declared=source.index["metadata"]["total_size"], measured=total, reviewed=total)
        super().__init__(output, source.snapshot, BASE_REPO, BASE_REVISION, identity, inv, **kwargs)

    def write_block(self, family, index=None, *, k4, k8, k6):
        """Seal one target layer, draft layer or the remaining outer tensors."""
        if family in ("layers", "mtp") and type(index) is int and index >= 0:
            prefix = f"model.{family}.{index}."
            selected = [n for n in self.header.roles if n.startswith(prefix)]
            label = f"{family}-{index:03d}"
        elif family == "outer" and index is None:
            selected = [n for n in self.header.roles if not re.match(r"^model\.(?:mtp|layers)\.\d+\.", n)]
            label = "outer"
        else:
            raise ValueError("invalid ERNIE block")
        if not selected:
            raise ValueError("empty ERNIE block")
        native, original = [], []
        expected = {4: set(), 8: set(), 6: set()}
        for name in selected:
            role = self.header.roles[name]
            if role == "flashindexer":
                continue
            if role == "native":
                native.append(name)
                continue
            if role == "core":
                original.append(name)
            module = name.removesuffix(".weight")
            for bits in ((4,) if role == "routed_expert" else (8, 6)):
                expected[bits].update(module + "." + s for s in ("trellis", "suh", "svh", "mul1"))
        if any(expected[bits] != entries.keys() for bits, entries in ((4, k4), (8, k8), (6, k6))):
            raise ValueError(f"incomplete ERNIE packed block: {label}")
        return self.write_group(label, original=self.header.source_entries(original),
                                shared=k4 | self.header.source_entries(native), k8=k8, k6=k6)

    def finalize(self, *, error_csv, card_path):
        source = self.header
        native, core, experts = projection_inventory(source)
        # Keep the auxiliary source shard in a dedicated shared output group.
        flash = sorted(n for n, role in source.roles.items() if role == "flashindexer")
        self.write_group("flashindexer", shared=source.source_entries(flash))
        configs, quantizes = view_metadata(source, native, core, experts, self.identity)
        shared = native | {name: spec for module, g in experts.items()
                           for name, spec in packed_specs(module, g).items()}
        sizes = {kind: sum(sum(group["sizes"][kind].values()) for group in self.state["groups"].values())
                 for kind in ("shared", *VIEWS)}
        card = Path(card_path)
        card.write_text(model_card({v: sizes["shared"] + sizes[v] for v in VIEWS}, self.identity))
        attrs = card.with_name("ernie-hub.gitattributes")
        attrs.write_bytes(LFS_ATTRIBUTES)
        assets = {name: source.snapshot / name for name in ASSETS}
        assets.update({"README.md": card, ".gitattributes": attrs, "per-module-errors.csv": error_csv})
        receipt = super().finalize(core_selection=core, configs=configs, quantizes=quantizes, assets=assets,
                                   expected_shared_specs=shared, exl3_codebook=dict(name="mul1", multiplier=mc.MUL1))
        discovery = verify_original_discovery(self.output)
        return dict(receipt, plain_loader=discovery)
