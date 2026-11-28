# Multi-core checkpoints

MC schema version 1 replaces attention-only alternate publications. There is no
HQA or compact-attention compatibility path. This document defines a publication
contract; safetensors headers remain authoritative for payload types and shapes.

## Views

One repository contains EXL3 experts, shared non-core tensors and complete,
mutually exclusive dense-core variants. Non-core tensors (including embeddings,
kept-native projections, towers and independent drafters) are stored once.
Bundled MTP core is variant-dependent: repaired original views restore its
source BF16 matrices, while stored compact views retain their existing K4
MTP matrices. MTP routed expert encodes remain in the shared inventory. An index exposes
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

`fp8-compact` has no files or stored variant. It reads exactly the `original`
variant's `core_tensors` IDs and its BF16 index, config and quantization metadata;
it never opens a K6/K8 index. Those IDs alone select conversion for owned
operands. GLM and GLM Flash use external drafters: native MTP core and experts
are unused speculator roles, so they are neither loaded nor converted even
when the original inventory declares their BF16 core. Plain checkpoints
have no definition and refuse this selector, including the former plain GLM
Flash conversion path. The shared push loader converts each selected BF16 matrix
into its final arena slot: row-major E4M3, FP32 scales per 128x128 block,
round-to-nearest-even, scale = amax/448 (`FP8_RECIPE`). Admission charges the FP8
payload and scale grid once; conversion scratch is transient and no resident
BF16 copy remains. The view identity includes the metadata and recipe.

Families declare executable roles, not conversion subsets. GLM Flash supports
its MC core, including the vocabulary head, KDA q/k/v/o, MLA and dense/shared
FFNs. Muse executes block FP8 q/k/v/o, selected attention gates, pre/post-expert
projections and its padded vocabulary head. GLM and Step-5 native whole-core
FP8 routes require the v52e coordinator exports; Step includes selected MTP
non-routed core, while GLM native MTP remains unused. Their max-card Standard generated quality and strict speculation losslessness
are qualified; measured speed and quality costs are in docs/options.md and the
family core-view sections. The selector remains opt-in.
Kimi refuses unsupported selected roles. An owned
unsupported tensor always refuses; unused external-drafter MTP is excluded
by ownership, never partially converted. Already-FP8 sources refuse conversion.
KDA/head conversion keys remain engine extras for weights outside the selected
view. Activation-pass keys select W8A8 execution, not target-weight conversion.

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
  "required_features": [],
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
Readers ignore unknown fields in the manifest, source and variants, logging their
names at debug level. Writers and publication validators accept only defined
fields. Metadata that changes how weights are read must be named in the optional
top-level `required_features` array of strings (omitted means empty); an engine
refuses a checkpoint requiring a feature it does not support. No features are
currently defined. Breaking changes to known fields require a schema bump.

The optional top-level `exl3_codebook` object records the uniform EXL3
codebook used by a direct MC package. Its fields are `name` and `multiplier`.
Hy4-flash uses `mul1` with unsigned multiplier `0x83DCD12D` throughout.
The declaration is advisory. Each indexed projection carries its own scalar
I32 `.mul1` tensor alongside `.trellis`, `.suh` and `.svh`. The writer
verifies the scalar's 32-bit value and binds the codebook to each view's
quantization metadata and the producer identity. Hy4 compact DFly2 target
matrices follow the view's declared drafter width. The original view carries
the source-native drafter.

The optional `engram_fp8` inventory records Hy4-flash alternate embedding shards.
Each 100-byte row has 96 E4M3 codes and a BF16-rounded scale stored as F32.
Decode multiplies in F32 and casts to BF16. The exporter compares every row
bitwise with the pinned stock BF16 lookup expression, including signed zero.
All standard indexes use the original BF16 tables.

Calibration corpus provenance is not duplicated in `cuteafd-mc.json` (which stays
strictly discovery/view metadata); it lives in each view's `quantize_config*.json`
under `provenance` (`dataset`: `{repo_id, revision, sha256sums_sha256}`,
`components`: per-component `{count, tokens, input_ids_sha256, token_stream_sha256}`
over `agentic`, `exl3_standard`, and `long_context`, with `long_context` also
recording `mode` and `window_tokens` for weight quants' 2,048-token windows, plus
`corpus_manifest_sha256`)
and in the model card's recipe table. Single source of truth for the panel pin is
`quantization/panel_corpus.py`: new quants calibrate on
`wrldsuksgo2mars/agentic-quant-panel-v3@e3ac6ee612` (all three components);
`wrldsuksgo2mars/agentic-quant-panel-v2@54683731eb` stays pinned for past runs.

Unknown schema versions or variant names, duplicate variants/tensor names,
invalid hashes, unsafe filenames and incomplete views still fail closed. Core bits
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
per-variant fingerprint, while binding exact selected metadata bytes. The loader's
full identity additionally hashes the discovery file, selector and FP8 recipe
when applicable. Adding even advisory metadata changes this full identity, as
expected for a new revision. Source revision and inventories are immutable
within a published revision.

## Validation and publication

Validate every view's metadata hashes, fingerprint, complete core module role
set, native source byte equality, trellis widths and authoritative shard headers
before publishing. Validate identical shared inventories across views and that
the original index names no trellis core tensors and the K8 index names no K6
core tensors. Unindexed bytes from other views may remain in shared shards. The loader
checks only the selected view's manifest/index contract during discovery;
normal inventory validates selected shard headers afterward.

Publish only after this format specification has been reviewed.
[`wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K4-MC`](https://huggingface.co/wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K4-MC)
is published at `558931fe149f59b49d19b2376a692bd7a12fb8b7`, with BF16
`original` (default), `k8-compact` and `k6-compact`. The runtime-derived
`fp8-compact` view converts the selected original core to block FP8;
the family inventory supports its attention, dense/shared FFN and head roles.
See its [min](../examples/configs/glm53-flash-exl3-k4-mc-min.config) and
[max](../examples/configs/glm53-flash-exl3-k4-mc-max.config) configs.
Max K8/K6/FP8 supplementary comparisons live in
`scripts/bench/release-smoke-v51-mc-core-comparison.json`, separate from
the original-view release row. The original max baseline still uses FP8-QK
MLA attention; rerun it when BF16 attention lands (#2146).
The GLM 5.3 and Step-5 publications are also available:

| Repository (`wrldsuksgo2mars/`) | Pinned revision | Views |
| --- | --- | --- |
| `GLM-5.3-EXL3-K4-MC` | `2e42b53e61f0f59f664c915fe2424ce9c9fa5f8e` | original BF16 (default), k8-compact, k6-compact |
| `Step-5-Preview-EXL3-K4-MC` | `aedda90d25bd02bb6520bcaf88e06e6c9bc932c4` | original BF16 (default), k8-compact, k6-compact |

Both replace their standalone original/K6 release rows while keeping historical
reports. See [GLM 5.3](models/glm5.md#core-views) and
[Step-5](models/step5.md#core-views) for launch overrides and pending Standard
fidelity. Whole-core `fp8-compact` is an explicit opt-in for both with the
v52e coordinator exports and qualified runtime routes; the selected original
core alone determines conversion. Unselected native roles and routed experts
are unchanged; max-card measured effects and quality costs are documented in
the family core-view sections. Original remains the default.
The main smoke entries use `original`; max K8/K6 comparisons live separately in
`scripts/bench/release-smoke-v51-mc-core-comparison.json` and do not add
release-table rows. Card measurements wait for kernel pin #1903.

`wrldsuksgo2mars/Muse-Spark-Open-EXL3-K4-MC` is published at
`cc5dc2537083177e3f02ea0b2d9dfc179d40425c`, with original BF16 (default),
`k8-compact` and `k6-compact`. Its new MC min/max configs and release row replace the
regular standalone Core-K6 cards; standalone configs, serving support and
historical reports remain. See
[Muse core views](models/muse_spark.md#core-views). Main Release smoke entries
select `original`, with separate max K8/K6 comparisons in the same comparison
matrix. Whole-core `fp8-compact` remains unsupported until its selected head
and pre/post-expert roles execute (#1796).

`wrldsuksgo2mars/Kimi-K3.1-Pro-EXL3-K2-MC` is also published; compact-view
metadata correction and K3.1 serving qualification are tracked in #2098/#1777.

The non-MC Mistral publication is pinned at
`9c354cee78cd6feddb057549b8edde6e3b319822`; its
[min](../examples/configs/mistral-large4-exl3-k4-min.config) and
[max](../examples/configs/mistral-large4-exl3-k4-max.config) configs select
only `original`.
