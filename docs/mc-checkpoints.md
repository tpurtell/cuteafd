# Multi-core checkpoints

MC schema version 1 replaces attention-only alternate publications. There is no
HQA or compact-attention compatibility path. This document defines a publication
contract; safetensors headers remain authoritative for payload types and shapes.

## Views

One repository contains EXL3 experts, shared non-core tensors and complete,
mutually exclusive dense-core variants. Non-core tensors (including embeddings,
kept-native projections, towers and drafters) are stored once. An index exposes
exactly one variant; unselected tensors are never inventoried or loaded. A
referenced shard may physically hold unselected core tensors: GLM Flash and
Muse share expert shards with K6 core buffers, allowing server-side shard copies
without uploading the same experts again. Only indexed names belong to a view;
unreferenced bytes in such shared shards are allowed.

| CORE_VIEW | Index | Config | Quantization metadata |
| --- | --- | --- | --- |
| original (default) | model.safetensors.index.json | config.json | quantize_config.json |
| k8-compact | model.k8.safetensors.index.json | config.k8.json | quantize_config.k8.json |
| k6-compact | model.k6.safetensors.index.json | config.k6.json | quantize_config.k6.json |
| fp8-compact | original index | original config | original quantization metadata |

The original core is restored byte for byte from the pinned source checkpoint.
Each compact view contains the complete K8 or K6 trellis representation of every
quantized core module, including the head when that family's quantizer includes
it. Per-view config references name that view's own index and metadata. No index
aliases or fallback to another compact width are supported.

`fp8-compact` has no files or stored variant. It derives from `original` using
`FP8_RECIPE`; its identity includes the recipe. Conversion covers only whole-core
roles supported by that family's existing runtime, never routed experts,
embeddings, routers, norms, towers or drafters. Initially the converter retains
the family's existing qualified attention-only scope; whole-core FP8 expansion
is a separate runtime change. Unsupported families refuse it.

An FP8-origin model is not MC and contains no trellis-quantized core tensors,
including unindexed shards. Mistral Large 4 is published as
`wrldsuksgo2mars/Mistral-Large-4-EXL3-K4`, with its original mixed-precision core
and EXL3 routed experts. It has no compact core views.

## Discovery

A repository is MC only when `cuteafd-mc.json` exists. Its size and every selected
JSON file are bounded by the loader's `MAX_JSON_BYTES`. Discovery reads this file
and the selected metadata/index, never all shard headers. Version 1 uses this
shape (hash values here are illustrative placeholders, not publication data):

```json
{
  "schema": "cuteafd-mc-v1",
  "source": {"repo": "organization/original", "revision": "pinned-commit"},
  "original_core_dtype": "BF16",
  "shared_inventory_sha256": "<sha256>",
  "variants": [
    {
      "name": "original",
      "index": "model.safetensors.index.json",
      "config": "config.json",
      "quantize": "quantize_config.json",
      "core_bits": 16,
      "core_tensors": ["model.layers.0.self_attn.q_proj.weight"],
      "metadata_sha256": {
        "model.safetensors.index.json": "<sha256>",
        "config.json": "<sha256>",
        "quantize_config.json": "<sha256>"
      },
      "fingerprint": "<sha256>"
    }
  ]
}
```

Required variants: `original`, plus one or both of `k8-compact`, `k6-compact`.
`fp8-compact` is a runtime-derived selector and never a manifest variant.
Unknown schema versions, fields, variants, duplicate variants/tensor names,
invalid hashes, unsafe filenames or incomplete views fail closed. Core bits
are exactly 16, 8 and 6 respectively. FP8-origin declarations fail before any
shard is opened. `original_core_dtype` is `BF16` (with source-native F32
router-style kept parts allowed), or `F32` for a source whose core is F32;
FP8 is never accepted. Every `core_tensors` name is present in its selected index;
the list is nonempty, unique and lexicographically sorted.

A shared inventory consists of selected `weight_map` entries excluding
`core_tensors`. Its SHA256 is computed over entries sorted by tensor name,
feeding each UTF-8 tensor name, a NUL byte, its UTF-8 shard basename and a newline.
Every view must produce the same shared digest at publication validation;
the loader verifies the selected view against it. Shard basenames cannot
contain path separators, NUL or newline. Tensor names cannot contain NUL/newline.

`metadata_sha256` binds the exact bytes of the three selected metadata files.
A view fingerprint is SHA256 of the UTF-8 bytes
`cuteafd-mc-view-v1\0<name>\0<shared_inventory_sha256>\0`, followed by
`<filename>\0<sha256>\n` for the three metadata files sorted by filename.
This avoids depending on JSON whitespace or object iteration order for the
manifest identity, while binding exact selected metadata bytes. The loader's
full identity additionally hashes the discovery file, selector and FP8 recipe
when applicable. Source revision and inventories are immutable within a
published revision.

## Validation and publication

Validate every view's metadata hashes, fingerprint, complete core module role
set, native source byte equality, trellis widths and authoritative shard headers
before publishing. Validate identical shared inventories across views and that
the original index names no trellis core tensors and the K8 index names no K6
core tensors. Unindexed bytes from other views may remain in shared shards. The loader
checks only the selected view's manifest/index contract during discovery;
normal inventory validates selected shard headers afterward.

Publish only after this format specification has been reviewed. The intended
MC repositories are Kimi-K3.1-Pro-EXL3-K2-MC, Muse-Spark-Open-EXL3-K4-MC and
GLM-5.3-Flash-EXL3-K4-MC under `wrldsuksgo2mars`.
