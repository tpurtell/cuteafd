# Inkling 2 Small

Inkling 2 Small is being ported as a first-class family on the shared engine
([#1930](https://github.com/tpurtell/cuteafd/issues/1930), v5.2). Nothing is
served yet; this page records the checkpoint facts the port is built on, read
from each checkpoint's own `config.json` and safetensors headers.

## Architecture

48 layers, hidden 4608, vocabulary 201024 stored (logits trimmed to 200058,
final softcap 30), untied embedding and unembedding. Attention is GQA 48/8 x 128
on every layer: 40 sliding-window-1024 layers and 8 global layers (5, 11, ...,
47), with a relative-position bias (`wr_du`, `rel_logits_proj`), per-token
per-head attention sinks (unscaled extra softmax-denominator logits), an output
gate, L2 clips on q, k and r, short causal K/V convolutions (static taps on local
layers; static plus per-token dynamic taps on global layers) and separate Q/K/V
input mixes on global layers. Residuals are softmax mixes over blocks of six
layers (F32 queries). Each layer also has `attn_sconv` and `mlp_sconv` short
convolutions on the hidden stream. The MoE is 384 routed experts, top-8, no
shared expert; `w13` is interleaved gate/up, the sigmoid uses the uncapped gate,
then tanh softcaps 4 (gate) and 25 (up) apply, and the route weight multiplies
the activated intermediate before the down projection's input quantization.

The bundled MTP drafter (`mtp.safetensors`) is a five-layer dSpark with block
size 8 over DFlare-mixed target features from layers 5, 11, 17, 23, 29, 35, 41
and the final norm, and an `rnn_hidden` Markov head. The bundled vision tower
is a 23-layer ViT with a 2x2 merge projector into the text hidden size.

## Checkpoints

| Checkpoint | Experts | Core |
|---|---|---|
| `thinkingmachines/Inkling-2-Small-MXFP8` @ `3febc62259ee92ea36e2fb6e8303371c4eb3922c` | MXFP8 on all 48 layers: E4M3 weights, UE8M0 scales per row and 32-wide K group, MXFP8 activations | BF16 |
| `thinkingmachines/Inkling-2-Small` @ `aac974122acd33ddb0d3e1f54433a9c5dd2249ce` | layers 0-39 NVFP4 (packed E2M1, E4M3 per-16 scales, F32 per-expert amax and outer denominator; per-token amax activations), layers 40-47 MXFP8 | BF16 |
| `wrldsuksgo2mars/Inkling-2-Small-EXL3-K4-MC` | EXL3 K4 | MC views: `original` BF16, `k8-compact`, `k6-compact` (drafter included) |

The per-layer expert format is derived from each layer's tensor headers (the
presence of `w13_weight_amax` marks an NVFP4 layer), never from
`hf_quant_config.json`, which is only cross-checked. Expert math runs native:
MXFP8 W8A8, NVFP4 W4A4 and EXL3 trellis.

Serving status, layouts, configs and measurements land here stage by stage;
see the tracking issue for the current state.
