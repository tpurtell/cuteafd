"""Direct three-view Hy4-flash package, with one native copy and alternate Engram."""
from __future__ import annotations

from copy import deepcopy
from collections import Counter
from pathlib import Path
import re

import mc_package as package
from inkling2.publish import LFS_ATTRIBUTES
from inkling2.export import DirectMCWriter as SharedDirectMCWriter, VIEWS
from . import recipe


ASSETS = ("LICENSE", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
          "generation_config.json", "processor_config.json", "preprocessor_config.json",
          "special_tokens_map.json")


def _storage_entry(module, geometry):
    stored = {name: dict(shape=shape, torch_dtype=package.DTYPES[dtype])
              for name, (dtype, shape) in recipe.packed_specs(module, geometry).items()}
    return package.tensor_storage_entry(geometry, stored, side=1,
                                       mcg_multiplier=None, mul1_multiplier=package.MUL1)


def _native_specs(source_inventory, selection):
    excluded = {module + '.weight' for module in selection} | recipe.expert_source_specs().keys()
    return {name: spec for name, spec in source_inventory['specs'].items() if name not in excluded}


def view_metadata(document, inventory, selection, experts, identity, *, k6_drafter_bits):
    """Bind each view to its own index/metadata and exact packed tensor shapes."""
    native = _native_specs(inventory, selection)
    configs, quantizes = {}, {}
    for view in VIEWS:
        index, _, quantize, bits = package.mc_manifest.FILES[view]
        compact = view != 'original'
        config = deepcopy(document)
        declaration = dict(quant_method='exl3', format='exl3', codebook='mul1', codebook_multiplier=package.MUL1,
                           per_module_metadata=f'{quantize}:tensor_storage')
        config['quantization_config'] = declaration
        core = {}
        if compact:
            for module, geometry in selection.items():
                width = k6_drafter_bits if bits == 6 and geometry['group'] == 'mtp' else bits
                core[module] = dict(geometry, bits=width)
        modules = experts | core
        # Exact name patterns for core plus a bounded routed range. The source
        # has no fused expert names in the EXL3 view.
        patterns = [r'^model\.layers\.(?:[1-9]|[1-3][0-9]|4[01])\.mlp\.experts\.'
                    r'(?:[0-9]|[1-9][0-9]|[1-4][0-9]{2}|50[0-9]|51[01])\.(?:gate_proj|up_proj|down_proj)$']
        patterns.extend('^' + re.escape(name) + '$' for name in sorted(core))
        retained = native if compact else native | {name + '.weight': inventory['specs'][name + '.weight']
                                                  for name in selection}
        meta = package.quantize_config_for_view(declaration,
            tensor_storage={module: _storage_entry(module, g) for module, g in modules.items()},
            retained_native_tensors={name: dict(dtype=dtype, shape=shape)
                                     for name, (dtype, shape) in retained.items()},
            provenance=identity, core_view=view, upstream_activations='k8-compact',
            engram_checkpoint_dtype='BF16')
        meta, config = package.gptqmodel_mc_view(config, meta, bits=4,
            core_bits=bits if compact else None, module_include=patterns, codebook='mul1')
        config['quantization_config']['safetensors_index'] = index
        meta['safetensors_index'] = index
        if compact:
            drafter_bits = bits if bits == 8 else k6_drafter_bits
            config['quantization_config']['drafter_bits'] = drafter_bits
            meta['drafter_bits'] = drafter_bits
        configs[view], quantizes[view] = config, meta
    return configs, quantizes


def verify_original_discovery(output, expected_mapping):
    """Run the pinned Transformers default discovery without loading weights."""
    from transformers import AutoConfig
    from transformers.modeling_utils import _get_resolved_checkpoint_files

    output = Path(output)
    config = AutoConfig.from_pretrained(output, local_files_only=True)
    files, metadata = _get_resolved_checkpoint_files(
        output, variant=None, gguf_file=None, use_safetensors=True,
        user_agent=None, is_remote_code=False, download_kwargs={"local_files_only": True})
    if (config.model_type != "hy_v4_flash" or not metadata
            or metadata["weight_map"] != expected_mapping
            or {Path(path) for path in files} != {output / shard for shard in expected_mapping.values()}
            or not all(Path(path).is_file() for path in files)):
        raise ValueError("Transformers default discovery differs from verified Hy4 original headers")
    return dict(status="passed", scope="default config, index and sealed header inventory",
                reference_revision=recipe.REFERENCE_REVISION,
                config_type=type(config).__name__, shards=len(files), tensors=len(expected_mapping))


def model_card(path, engram, *, k6_drafter_bits, selection, identity, view_bytes):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mismatches = sum(row['mismatch_count'] for row in engram['proof'].values())
    rows = sum(row['rows'] for row in engram['proof'].values())
    if mismatches != 0:
        raise ValueError("model card cannot claim an exact Engram alternate with mismatches")
    native_gib = rows * 192 / (1 << 30)
    fp8_gib = rows * 100 / (1 << 30)
    head = 'with a compact head' if 'lm_head' in selection else 'with the original BF16 head'
    dataset = identity['dataset']
    agentic = identity['components']['agentic']
    standard = identity['components']['exl3_standard']
    gib = {view: count / (1 << 30) for view, count in view_bytes.items()}
    text = f"""---
library_name: cuteafd
license: other
license_name: Tencent Hy License, Version 1.0
license_link: LICENSE
base_model: tencent/Hy4-flash
datasets:
- {dataset['repo_id']}
tags:
- exl3
- mixture-of-experts
---

# Hy4-flash EXL3 K4 MC

## What multi-core means

All three indexes share the K4 MUL1 experts, router, embeddings, compressed
indexer, iHC tensors, vision tower and projector. The core view selects the
precision of the target and eligible DFly2 matrices {head}.

| View | Indexed weight payload | When to use it |
| --- | ---: | --- |
| `original` (default) | {gib['original']:.2f} GiB | Source-native core and byte-identical original drafter |
| `k8-compact` | {gib['k8-compact']:.2f} GiB | Compact K8 target and K8 drafter |
| `k6-compact` | {gib['k6-compact']:.2f} GiB | Smallest indexed target, with K{k6_drafter_bits} drafter matrices |

The table counts tensor payload in the selected index. Each index includes
the BF16 Engram tables. The FP8 alternate adds {fp8_gib:.2f} GiB to disk.

## How it was made

Source `{recipe.BASE_REPO}@{recipe.BASE_REVISION}`. The panel includes
token-budgeted real source code, worked mathematics, function-call schemas
and multilingual text. Hy4's own chat template rendered the agentic prompts
with high reasoning effort. The second component uses the standard ExLlamaV3
raw-text and random-token packing. The 2,709-token DSA and SWA-wrap witnesses
checked reference parity separately. Calibration does not exercise DSA selection
beyond the top-2,048 cutoff.

One causal pass propagated the serialized K8 core and K4 expert reconstructions.
K6 used the same core Hessians. DFly2 used the target hidden states, KV and
indexer selections at layers 25, 29, 33, 37 and 41.

| Recipe | Value |
| --- | --- |
| EXL3 | MUL1 (`0x83DCD12D`), side 1, automatic output scales |
| Bits | Shared experts K4, compact core K8 and K6 |
| Search | Seed {identity['seed']}, sigma {identity['sigma']}, checked FP64 Hadamard congruence |
| Hessian | At most 128 rows per FP32 product, FP64 sum, FP32 storage |
| Panel | `{dataset['repo_id']}@{dataset['revision']}` |
| Calibration order | {agentic['count']:,} agentic prompts ({agentic['tokens']:,} tokens), then {standard['count']:,} standard rows ({standard['tokens']:,} tokens) |
| Total calibration | {agentic['tokens'] + standard['tokens']:,} tokens |
| Standard input ID SHA256 | `{standard['input_ids_sha256']}` |
| Routed recovery | Keep all natural rows. Below 1,024, fill from ranks 17–32 in rank and corpus order, normalizing hypothetical gamma over 15 native routes plus the candidate. Fill any remaining deficit with raw-XTX identity observations. |
| Causal combine | Ordered FP32 sum on the GPU over native routes only |
| Replay | Decode serialized trellis and scales to BF16 and promote to FP32 |
| Encoder | `tpurtell/GPTQModel@{identity['code']['fork']}` |

Per-projection errors and routed coverage are in `per-module-errors.csv`.

## Engram tables

The four dedicated BF16 shards contain {native_gib:.2f} GiB of source-identical
table data. The alternate stores 96 E4M3 codes followed by the BF16-rounded
scale in a little-endian F32 field per 100-byte row. Lookup multiplies code and
scale in F32 and casts to BF16. Every one of the {rows:,} rows was compared
bit for bit to the complete stock BF16 lookup round trip, including signed
zero, with {mismatches} mismatches. The ordinary indexes select BF16. The
`engram_fp8` field in `cuteafd-mc.json` identifies the FP8 part files.

## Running it

Use `config.json` and `model.safetensors.index.json` for the default view.
MC-aware loaders select `k8-compact` or `k6-compact` through `cuteafd-mc.json`.
Serving requires Hy4-flash architecture support and MUL1 routed-expert kernels.
CuteAFD tracks bring-up and serving qualification in [issue #2478](https://github.com/tpurtell/cuteafd/issues/2478).

## License

The [Tencent HY License](LICENSE) accompanies this checkpoint.
"""
    path.write_text(text, encoding='utf-8')
    return path


class DirectMCWriter(SharedDirectMCWriter):
    """Reuse direct MC shard writer with a complete Hy4 native byte inventory."""

    family = 'hy4_flash'
    prove_shared_source = True

    def __init__(self, output, source_snapshot, identity, source_inventory, **kwargs):
        if identity.get('codebook') != 'mul1' or identity.get('codebook_multiplier') != package.MUL1:
            raise ValueError('Hy4 direct writer requires the frozen MUL1 producer identity')
        super().__init__(output, source_snapshot, recipe.BASE_REPO, recipe.BASE_REVISION,
                         identity, source_inventory, **kwargs)

    def finalize(self, *, core_selection, engram, document, error_csv, card_path,
                 k6_drafter_bits=6, retained_by_gate=()):
        """Seal and independently validate exact complete per-view inventories."""
        if set(engram['tables']) != recipe.ENGRAM_TABLES.keys() or k6_drafter_bits not in (6, 8):
            raise ValueError('Hy4-flash requires all dedicated Engram tables and a reviewed drafter width')
        reviewed = recipe.reviewed_core_modules(
            include_head='lm_head' in core_selection or 'lm_head' in retained_by_gate,
            include_kv_b=any(name.endswith('.kv_b_proj') for name in (*core_selection, *retained_by_gate)))
        if any(module not in reviewed or any(g[key] != reviewed[module][key] for key in
               ('in_features', 'out_features', 'stored_out_features', 'group'))
               for module, g in core_selection.items()):
            raise ValueError('unreviewed selected core')
        if {n for n in core_selection if n.startswith('model.dfly_layers.')} != {
                n for n in reviewed if n.startswith('model.dfly_layers.') and n not in retained_by_gate}:
            raise ValueError('compact views require every selected DFly2 matrix')
        native = _native_specs(self.source_inventory, core_selection)
        experts = recipe.reviewed_expert_modules()
        shared = native | {name: spec for module, g in experts.items()
                           for name, spec in recipe.packed_specs(module, g).items()}
        configs, quantizes = view_metadata(document, self.source_inventory, core_selection,
                                          experts, self.identity, k6_drafter_bits=k6_drafter_bits)
        assets = {name: self.source / name for name in ASSETS if (self.source / name).is_file()}
        if not {'LICENSE', 'tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja',
                'generation_config.json', 'processor_config.json'} <= assets.keys():
            raise ValueError('source license, tokenizer or processor assets are incomplete')
        aliases = {module for module, geometry in core_selection.items()
                   if geometry['group'] == 'mtp'} if k6_drafter_bits == 8 else ()
        alias_tensors = {name for module in aliases for name in recipe.packed_specs(
            module, core_selection[module], bits=8)}
        sizes = Counter()
        for group in self.state['groups'].values():
            for kind, members in group['sizes'].items():
                sizes[kind] += sum(members.values())
                if kind == 'k8-compact':
                    sizes['k6-compact'] += sum(members[n] for n in alias_tensors.intersection(members))
        sizes['shared'] += sum(self.source_inventory['sizes'][n] for n in engram['tables'])
        assets['README.md'] = model_card(card_path, engram, k6_drafter_bits=k6_drafter_bits,
                                        selection=core_selection, identity=self.identity,
                                        view_bytes={view: sizes[view] + sizes['shared'] for view in VIEWS})
        attributes = Path(card_path).parent / 'hy4-hub.gitattributes'
        attributes.write_bytes(LFS_ATTRIBUTES)
        assets['.gitattributes'] = attributes
        assets['per-module-errors.csv'] = Path(error_csv)
        return super().finalize(core_selection=core_selection, configs=configs, quantizes=quantizes,
                               assets=assets, engram=engram, k6_use_k8=aliases,
                               expected_shared_specs=shared,
                               exl3_codebook=dict(name="mul1", multiplier=package.MUL1))
