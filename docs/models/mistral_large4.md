# Mistral Large 4

`mistralai/Mistral-Large-4-1T-A52B` (official FP8 block-128) is served as an
EXL3 quant, `wrldsuksgo2mars/Mistral-Large-4-EXL3-K4-Core-K6`: K4 routed experts
on six Sparks (with RTX onboarding) and a native K6 dense core (attention,
dense and shared FFN, vocabulary head) on the RTX cards. The quant is in
progress, so this family has no release card or starting config yet. Gate
cards: `ml4-exl3-min` (1× RTX + 6 Sparks) and `ml4-exl3-max` (2× RTX + 6
Sparks) in `scripts/bench/kits/gate/matrix-ml4.json`.

## How it runs

Mistral Large 4 runs on the GLM 5.3 engine as its `ml4` variant, chosen from
the checkpoint's own `config.json` (never a launch key). Operand names come
from the consolidated checkpoint (`layers.N.attention.wq`, `experts.E.w1`,
`output`), and the shard index is the one `config.json:transformers_weights`
names.

- **Attention:** absorbed MLA with no q-LoRA (`wq` straight from the normed
  hidden state). It uses the scalable-softmax query scale
  (`q *= s * log(pos + 1) + b` per head, after RoPE) and interleaved YaRN with
  LeRoPE. Every fourth layer (3, 7, …, 71) is NoPE with 528-byte latent
  records; the others use the 656-byte GLM record. Layers 39, 43, …, 71 read
  layer 35's latent pages and keep no KV of their own.
- **DSA indexer:** only on the 44 `full` layers; `shared` layers reuse the last
  full layer's top-k. The query is `wq_b(LayerNorm(h))` with its own query
  scale. `wq_b` runs in whichever form the checkpoint ships: the K6 core, or
  the official FP8 block-128 weight through `ml4_index_q` (never
  re-quantized). Ties at the top-k cutoff go to the lower key id, as in the
  reference.
- **Router and head:** a softmax top-4 of 128 experts, renormalized. The head
  is soft-capped with `tanh(x / 30) * 30`.
- **Placement:** on two GPUs, `auto` resolves to whole-layer owners (`layers`).
  A layer stays on the GPU of the layer whose latent and top-k it reads.
  `heads` and `context` run GLM 5.3's head split with Mistral Large 4's
  `ml42_*` share programs: each GPU projects its 32 heads and runs its half of
  `o`, followed by an all-reduce. Under `context`, each 64-row latent page
  lives once. GPUs exchange queries and run 64-head partials over their own
  pages at 128^-0.5, then LSE-combine. Staging and commits use each layer's own
  record (656 B, or 528 B on NoPE layers). A layer reading layer 35's latent
  commits no records. Which mode `auto` picks is settled by measurement.
- **Chat:** the server reads the chat contract from the checkpoint itself,
  from its tokenizer's control tokens and its `chat_template.jinja`, not from
  the engine variant. Prompts match `mistral_common`'s v15 instruct rendering
  exactly. That covers `[AVAILABLE_TOOLS]`, plus `[MODEL_SETTINGS]` with the
  JSON schema and a `reasoning_effort` of low, medium or high (default high).
  There is no thinking-off; the server refuses it rather than rendering
  another effort. Output parsing handles an optional `[THINK]…[/THINK]` block
  and back-to-back `[TOOL_CALLS]name[ARGS]{json}` calls. Strict tools, required
  or named `tool_choice`, and `response_format` are enforced by constrained
  decoding, because the template has no tool-choice text.
- **Drafter:** `SPECULATOR=dspark` serves the bundled `mtp_dspark` drafter
  through the shared MLA block-draft path. It has three BF16 non-causal MLA
  layers with block-anchored 2048-record windows, taps target layers 1, 18,
  36, 54 and 71, and uses an RNN Markov head. It stays opt-in
  (`SPECULATOR=off` is the default) until the lossless and delivered-tok/s
  gates pass on hardware.
- **Prefix cache:** snapshots carry each layer's records at their own size,
  and none for latent readers, whose pages belong to their source layer.
- **Not yet:** the vision encoder.

## References

- `python/reference/families/mistral_large4/model.py`: the official
  transformers 5.22.1 classes (hash-pinned) with exact FP8 dequantization.
  The fidelity goldens use it (`golden.py`).
- `ops.py`: op-level references, bit-exact against the official modules.
- `trace.py`: the per-layer engine golden that `cuteafd glm-golden` compares.
