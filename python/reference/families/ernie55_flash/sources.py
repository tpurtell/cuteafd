"""Pinned ERNIE 5.5 Flash sources and op-by-op reconciliation."""

CHECKPOINT = "baidu/ERNIE-5.5-Flash-Preview-BF16"
REVISION = "33a9925b17506c5ff57accac2b31a6520c951c4e"
CONFIG_SHA256 = "8c32ad48cecf14cb558ce8db3c00720031671af405bc6d91da9499935331a51f"
INDEX_SHA256 = "c66d18a9144f65af4ab10c33d40ff2df4720aa2bf58b27a71d2c6340baaf12f2"
FLASHINDEXER_SHA256 = "44fd87744c1546365eded8ca9131d6b4e0e250330a824b4a772daae949013fd2"
ENCODING_SHA256 = "0b353d1eb330574cab7704c7a09323fefb0b1b0b594db151f96b67848e3d1912"
TOKENIZER_SHA256 = "20e7c81f36fdfb90db664583fc6426d5c2ac6d6bcaf631c910941a8d8f30f2d9"

VLLM_ERNIE_REPO = "https://github.com/baidu/vLLM-ERNIE"
VLLM_ERNIE_REVISION = "ba322a5634f91d3e6f96ae711df72ddaf3c38b6e"
VLLM_VERSION = "0.32.0"
VLLM_ERNIE_FILES = {
    "src/vllm_ernie/models/ernie5_5/config.py": "f16d057b11bd8b8e920259657b41651d003908f99ff8aff4a94118a4912c343d",
    "src/vllm_ernie/models/ernie5_5/mhc.py": "02d9cf6c7789d77e655e0c2aea57e64c72ae03e717543e965a36bb83d3aab9f5",
    "src/vllm_ernie/models/ernie5_5/nvidia/model.py": "5ed1b6d243472f75cc13154a84f89a61e8db79ef875f49312812e69f40fe4473",
    "src/vllm_ernie/models/ernie5_5/nvidia/attention.py": "c01daadbef2df1263e158d7139f4046d1cbcadc6e7e088249c231fb4362989f5",
    "src/vllm_ernie/models/ernie5_5/nvidia/moe.py": "da723ed088ed7ed1552edf7756f5a85dd6989b1bfd41b34d7e4653609d591d74",
    "src/vllm_ernie/models/ernie5_5/nvidia/flashindexer.py": "e56618630aaf5d7ecafa7dde2fe405caaaa5a9b54264c323786f16ce7980f1fb",
    "src/vllm_ernie/models/ernie5_5/nvidia/dspark.py": "1b962cd4f74f9c48d701bab370b5502f9764aa72a716a4a186b32ecfd124f929",
}

SGLANG_REPO = "https://github.com/sgl-project/sglang"
SGLANG_REVISION = "5ff04f14f46ac86072e8d2596faa8a734b6f1133"
SGLANG_FILES = {
    "python/sglang/srt/configs/ernie55.py": "1f9e2a3cb1cb8f09a8343b96a2381459864ec9592e86a75265abf46069647af9",
    "python/sglang/srt/models/ernie55.py": "b68b2b722bf0f2d8ce77955823491a00337d25b1e388ea118e9ed893fa972957",
    "python/sglang/srt/models/ernie55_dspark.py": "865b72142d6c4f3da25ba5520f4474f864dc03726c3f331fa3c70123aa225bd8",
    "python/sglang/srt/layers/attention/ernie55_backend.py": "01e8beae23aa871f9d606dd51506033f4bf657247d2d7d7bbff538075ae5dc6e",
}

PAPERS = {
    "vha": "arXiv:2610.17959",
    "multi_router_latent_moe": "arXiv:2610.40577",
    "echo_mtp": "arXiv:2610.24766",
}

SEMANTICS = {
    "layer_map": (
        "43 layers (hidden_size=4096): layer 0 is dense SiTU MLP (intermediate_size=14336); "
        "layers 1..42 are multi-router latent MoE (512 routed experts top-10, latent 2048, "
        "moe_intermediate_size=2048, 1 shared expert). Attention is 37 HCA layers (compress_ratio=128) "
        "and 6 MLA-DSA layers (compress_ratio=-2) at indices [7, 14, 21, 28, 35, 42]."
    ),
    "hca_attention": (
        "Both vLLM-ERNIE (nvidia/attention.py + vllm 0.32.0 deepseek_v4/{attention,compressor,sparse_mla,sparse_swa}.py) "
        "and SGLang (models/ernie55.py:294-350 + deepseek_v4_backend.py) agree: "
        "q_c = RMSNorm(linear_q_down_proj(x), q_layernorm); q = linear_q_up_proj(q_c) [64, 512] -> "
        "unweighted per-head RMSNorm (rsqrt(mean(q^2)+eps)) -> interleaved YaRN RoPE (compress_rope_theta=160000, "
        "original_max_position_embeddings=65536, factor=16, beta_fast=32, beta_slow=1, mscale=0) on trailing 64 dims. "
        "kv = RMSNorm(linear_kv_proj(x), kv_layernorm) [512] -> same YaRN RoPE. "
        "Compressor: on complete 128-token blocks ((pos+1)//128 chunks), kv_c = linear_wkv(x), "
        "score = linear_wgate(x) + ape[pos % 128], softmax over 128 tokens per dim, weighted sum, compressor.norm, "
        "YaRN RoPE at chunk_idx * 128. Query at pos attends to compressed chunks [0 .. (pos+1)//128) union "
        "SWA [max(0, pos-127) .. pos] with scale 512^-0.5 and F32 attn_sink [64]. "
        "Output: VHA postmix (I + V @ U^T) over 64 heads -> inverse YaRN RoPE on trailing 64 dims -> "
        "grouped linear_o_group_proj (8 groups of 8x512 -> 1024) -> * sigmoid(gate_proj(q_c)) -> o_proj."
    ),
    "mla_dsa_attention": (
        "Both vLLM-ERNIE (nvidia/attention.py) and SGLang (models/ernie55.py:352-447, ernie55_backend.py:72-190) agree: "
        "q_c = RMSNorm(q_a_proj(x), q_a_layernorm); q = q_b_proj(q_c) [64, 256] (192 nope + 64 interleaved plain RoPE "
        "at rope_theta=8e6); kv_a_proj_with_mqa(x) -> kv_c [512] (kv_a_layernorm) + k_pe [64] (interleaved plain RoPE 8e6); "
        "kv_b_proj(kv_c) -> k_nope [64, 192] + v [64, 256]. "
        "Indexer: wq_b(q_c) [64, 128], k = LayerNorm(wk(x), eps=1e-6) [128], NeoX split-half plain RoPE (8e6) on "
        "leading 64 dims ([:64]), weights = weights_proj(x) * 128^-0.5 * 64^-0.5. Hadamard is orthogonal (invariant "
        "for exact F32 q@k^T before ReLU). Indexer scores strictly before SWA (0 .. pos-128) and selects top-2048; "
        "MLA attends to top-2048(0..pos-128) union SWA [max(0, pos-127)..pos] with scale 256^-0.5 and BF16-stored "
        "softmax_offset [64]. Output: VHA postmix (I + V @ U^T) -> * sigmoid(gate_proj(q_c)) -> o_proj."
    ),
    "flashindexer": (
        "Disagreement/opt-in: vLLM-ERNIE (config.py:38, nvidia/flashindexer.py) defaults enable_flashindexer=False; "
        "when enabled, selects top-8 of 64 indexer heads per token on prefill (gates = x @ linear_gate.T, "
        "scores = gates + flashindexer_bias, w_sel = w * gates_sel). SGLang (5ff04f14) has no FlashIndexer path. "
        "HALOps threshold pre-filter (indexer_fused_topk.py) is an exact kernel optimization that does not change "
        "top-2048 indices. Per main ruling: reference and golden run exact 64-head DSA (enable_flashindexer=False); "
        "flashindexer/flashindexer.safetensors tensors are validated by --check-only and opt-in enable_flashindexer=True "
        "is supported and preflighted."
    ),
    "mhc": (
        "mhc_pre/mhc_post: both vLLM-ERNIE (mhc.py, tests/models/ernie5_5/test_mhc.py) and SGLang (models/ernie55.py, "
        "kernels/ops/layernorm/mhc.py) agree: mapping_proj.weight stored [16384, 24] transposed; RMS factor uses "
        "eps outside sqrt: 1 / (sqrt(mean(x_flat^2)) + rms_norm_eps); pre = sigmoid(...) + hc_eps; "
        "post = 2.0 * sigmoid(...) (no +hc_eps); comb = softmax(dim=-1) + hc_eps, col-norm, then 19 rounds of "
        "row-norm + col-norm with +hc_eps; mhc_post = einsum('...ij,...ih->...jh', comb, residual) + post * x. "
        "Disagreement on mhc_contract (hc_head): vLLM-ERNIE (nvidia/model.py:483, nvidia/dspark.py:207) calls upstream "
        "hc_head_fused_kernel_tilelang with eps INSIDE sqrt: rsqrt(mean(x_flat^2) + rms_norm_eps); SGLang "
        "(models/ernie55.py:243-251) passes eps_outside_sqrt=True. Per main ruling (authority #1), this reference "
        "follows vLLM-ERNIE (eps inside sqrt) by default."
    ),
    "latent_moe_and_situ": (
        "Both vLLM-ERNIE (nvidia/moe.py:130-165, tests/models/ernie5_5/test_moe_routing.py) and SGLang "
        "(models/ernie55.py:563-625) agree: scores = sigmoid(x @ gate.weight.T) + sigmoid(x @ gate.weight_1.T); "
        "ids = topk(scores + e_score_correction_bias, k=10); weights = (scores[ids] / sum.clamp_min(1e-20)) * "
        "routed_scaling_factor_param[ids] (no scalar 3.0). "
        "CRITICAL latent_norm placement: latent = fc1_latent_proj(x) -> routed experts in 2048-d latent space -> "
        "sum weighted expert outputs -> latent_norm(routed) -> fc2_latent_proj(...) + shared_experts(x) "
        "(verified at vLLM-ERNIE src/vllm_ernie/models/ernie5_5/nvidia/moe.py:155-160 and "
        "SGLang python/sglang/srt/models/ernie55.py:609-624). "
        "SiTU activation (vllm 0.32.0 activation.py, sglang layers/activation.py): "
        "(beta * tanh(gate / beta) * sigmoid(gate)) * (linear_beta * tanh(up / linear_beta)) with beta=4.0, linear_beta=25.0. "
        "Precision note: vLLM-ERNIE keeps F32 router weights F32; SGLang casts to BF16 on load; F32 reference keeps F32."
    ),
    "lm_head_and_multimax": (
        "Untied lm_head + piece-wise multimax: logits + t0*relu(r0-logits) + t1*relu(logits-r1) + "
        "t2*relu(r2-logits)^2 + t3*relu(logits-r3)^2 (vLLM-ERNIE nvidia/model.py:103-118, test_multimax.py; "
        "SGLang models/ernie55.py:933-944). vLLM-ERNIE computes target head + multimax in F32 (head_dtype=float32); "
        "this reference computes in IEEE F32."
    ),
    "dspark_drafter": (
        "Bundled under model.mtp.*: taps outputs of target layers [40, 41, 42] (stream-averaged over 4 mHC streams), "
        "concatenated [T, 12288] -> feature_norm(feature_projection(...)) -> per-stage context KV via "
        "stage[i].kv_layernorm(stage[i].linear_kv_proj(main_x)) + draft RoPE. "
        "Block input: [anchor_token, 1024 * 6] (block_size=7, sample_from_anchor=True) -> target embed_tokens -> "
        "4 mHC streams -> 3 SWA-only MoE stages (non-causal attention over context[max(0, prefix_len-128):prefix_len] "
        "union full 7-token draft block) -> mtp.mhc_contract (head_hidden) -> mtp.norm -> target lm_head -> "
        "sequential Markov walk (markov_w1.T embedding -> markov_w2 linear bias) and confidence_head "
        "(sigmoid(proj(cat([head_hidden, markov_embed])))). "
        "Disagreements under 2x2 empirical measurement on real weights: "
        "(1) Draft RoPE: vLLM-ERNIE (nvidia/dspark.py:128-135) uses plain RoPE theta=8e6 vs SGLang "
        "(models/ernie55_dspark.py:88-95) uses YaRN compress_rope_theta=160000; "
        "(2) Multimax order: vLLM-ERNIE (nvidia/dspark.py:266-274, test_dspark_draft_multimax.py) applies multimax "
        "AFTER adding Markov bias vs SGLang (models/ernie55_dspark.py:350-361) applies multimax BEFORE Markov bias."
    ),
}

ROOT_ARITHMETIC = {
    "source": (
        "baidu/vLLM-ERNIE ba322a5634f9 + vLLM 0.32.0 (authority #1), cross-checked against "
        "sgl-project/sglang 5ff04f14f46a (authority #2), ported independently"
    ),
    "arithmetic": (
        "BF16 checkpoint, IEEE F32 forward; 37 HCA layers (128-token compressor + 128-token SWA, YaRN theta=160k, "
        "VHA postmix, inverse RoPE, grouped wo_a, gated wo_b) + 6 MLA-DSA layers (exact 64-head F32 DSA top-2048 "
        "before SWA union 128-token SWA, plain RoPE theta=8e6, FlashIndexer off by default); mHC (hc_mult=4, "
        "20 Sinkhorn iters, eps outside sqrt in mhc_pre, eps inside sqrt in hc_head); 1 dense SiTU + 42 multi-router "
        "latent MoE layers (latent_norm after routed sum before fc2_latent_proj, streamed per expert); untied F32 "
        "lm_head + multimax; fixed-M128 projections and query tiles; bundled dSpark drafter supported separately."
    ),
}
