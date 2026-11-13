"""Immutable arithmetic sources, inspected independently before porting.

These files are provenance, never runtime imports. Both integrations are vendor
branches that were not merged when the reference was written (2026-11-13).
"""
VLLM = {
    "repository": "https://github.com/WoosukKwon/vllm",
    "revision": "d664cd1e96052857aea2aa844eeba571c24758b4",
    "files": {
        "vllm/models/inkling2/configs.py": "a4ff9d082384d9ddc06888a7c54d3c97c0786b50f77409be032832fe649bd5be",
        "vllm/models/inkling2/model.py": "9c8afccd5920ba48d22c7186f746e6d663790ec7011c449074cf53f0dc5c2ff2",
        "vllm/models/inkling2/attention.py": "6726094a290f53729689b8e19e2b35a508a095f9e7a9e23cf1777a1308732123",
        "vllm/models/inkling2/moe.py": "a2324abf17f7a16f6f75eda392dfb76db60f5d046d0e55630fca3aa81436f8ef",
        "vllm/models/inkling2/ops/activation.py": "8ed2ace8df91f990816075329227d1a69eedfd61fe674376bc08ac419420a787",
        "vllm/models/inkling2/ops/attention.py": "4e8bd9165986b051e0286b5caccbc809dfaa16a34c3e2e5ce9b9a7a0aabd448a",
        "vllm/models/inkling2/ops/attn_res.py": "bf810c877088ac55f0b87a791a07fe756e0a6c115ec06aff166297350b847991",
        "vllm/models/inkling2/ops/moe.py": "59c9f5101febd26b1a6502271b5c7935153b166005591150cad8792bf8660deb",
        "vllm/models/inkling2/ops/mxfp8.py": "58431169170558babc59b193eecfa94ddc57d4bfd89d02c5fab20698decc7799",
        "vllm/models/inkling2/ops/sconv.py": "77e590dd25826a9cbad927e3ee11665350a0e88896c089df8c0c13f163e71956",
        "tests/models/inkling2/test_attention.py": "a0e4e22090b88c2c74f1be16a04832995ccdded5969e552ea1b80000738d93b2",
        "tests/models/inkling2/test_core_ops.py": "18a0cdf079f283debe0de4c8c6c474f58b33337482b27559da8f5d8652643cc4",
        "tests/models/inkling2/test_moe.py": "62f3f39f811707cfb1a777bd983f12de095eaed26c6e4d206d16a8f25563035b",
    },
}
SGLANG = {
    "repository": "https://github.com/sgl-project/sglang",
    "revision": "ec5e7ff575beb33e54509fe01934d6a4483b5830",
    "files": {
        "python/sglang/srt/models/inkling_common/inkling2_attn.py": "77301875fef971b751403a033ac68052b288b0f391371da74e80591aabe6fce6",
        "python/sglang/srt/models/inkling_common/inkling2_attn_res.py": "133602b9a430d089122257930ce79eecb8e7a9304c778bbb3d669fda68c1bda8",
        "python/sglang/srt/models/inkling_common/inkling2_sconv.py": "3685766ef160241c0ddc1e377c827b2380f04f1f39a482ad5b14c0490b00bc65",
        "python/sglang/srt/models/inkling_common/attn.py": "777d578e50b08e8f1291825972d11ac9771b01ab77af28de6a6fc1d28d784b9e",
        "python/sglang/kernels/ops/inkling2/attention.py": "5e0e5616d13e947e2f0ca4bae3af28d4a16c7082c4b7c312a6bb9ee43c15ec7e",
        "python/sglang/kernels/ops/inkling2/moe.py": "304d5334f6ee339d01aee943510c750a9ea44843ff9e67a62af1ef7cf04bc243",
        "python/sglang/kernels/ops/inkling2/route.py": "46b8cd021bc9c5d2a95c19574ef2a56dc7205d84c8417eadf2e805f28ca0c8b9",
        "python/sglang/kernels/ops/inkling2/sconv.py": "7de16c2de21b09874a6651d6db8cccb589a21d71105ba306f1aaab3a3f2f9321",
    },
}
SEMANTICS = {
    "relative_attention": "vLLM tests/models/inkling2/test_attention.py:_ref_attention and SGLang inkling_common/attn.py:get_inkling_relative_attention_score_mod: (QK + bias/scale)*scale; bias[distance] only for 0<=distance<extent, zero beyond; SWA distance<1024",
    "sink": "vLLM ops/attention.py:gate_sink and SGLang ops/inkling2/attention.py:gate_sink: zero-valued per-token, per-head sink exp(sink) enters the denominator; multiply sigmoid(output_gate)",
    "attn_res": "vLLM model.py:Inkling2Model.forward and test_core_ops.py:test_attn_res_matches_softmax_mix_then_norm; SGLang inkling2_attn_res.py:Inkling2AttnResState: normalized keys/values, query/sqrt(hidden), six-layer blocks, distinct global Q/K/V queries",
    "sconv": "vLLM test_core_ops.py:_dense_conv and SGLang inkling2_sconv.py plus ops/inkling2/sconv.py: zero causal padding, static oldest-first taps; dynamic lag-first static+per-token taps; add identity and divide sqrt(1+K) or sqrt(1+2K)",
    "softcaps": "vLLM ops/activation.py:_situ and SGLang ops/inkling2/moe.py:_capped_swiglu: sigmoid(original_gate)*softcap4(gate)*softcap25(up); routing gamma before down-input QDQ; final head softcap30",
    "w13": "vLLM ops/activation.py:_situ and SGLang ops/inkling2/moe.py:_swiglu: even rows gate, odd rows up, not contiguous halves",
    "mxfp8": "vLLM ops/mxfp8.py:_mxfp8_block and SGLang ops/inkling2/moe.py:_mxfp8_block: ceil-power-of-two UE8M0 exponent, nearest-even saturating E4M3, dynamic group32 activations before both expert GEMMs; E4M3*2^(UE8M0-127) weight decode",
}
ARITHMETIC = {
    "precision": "IEEE F32 throughout; checkpoint BF16/F32 dense weights cast exactly to F32; native MXFP8 expert weights decoded directly to F32; no BF16 intermediate",
    "activation": "Checkpoint QAT group32 MXFP8 QDQ before w13 and w2, including route weight before w2 QDQ; exponent rounds up, E4M3 rounds nearest-even",
    "geometry": "fixed M128 projections and causal M128 attention tiles; one packed layer plus one decoded expert at a time; layer-major durable residual blocks",
    "excluded": "vision, MTP, KV caching, tensor-core reduction order, approximate CUDA exp/tanh/rsqrt and BF16 serving-boundary narrowing",
}
