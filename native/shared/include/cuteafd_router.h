#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
// Official sqrtsoftplus routing: BF16 hidden [rows,5120], weight [experts,5120],
// FP32 biases [experts]; optional byte image mask [rows] selects bias_vl.
// FP32 scores workspace [rows,experts], u32 IDs and FP32 weights [rows,topk].
// experts=128/topk=3 or experts=384/topk=6; outputs disjoint from all inputs.
// Initialize AOT modules outside graph capture on the serving device first.
int32_t cuteafd_v41_router_initialize();
int32_t cuteafd_v41_router(const uint16_t* hidden,const uint16_t* weight,
    const float* bias,const float* bias_vl,const uint8_t* image_mask,float* scores,
    uint32_t* ids,float* routing,int32_t rows,int32_t experts,void* stream);
// Convert caller-projected FP32 logits in place to sqrtsoftplus scores and
// select/normalize with the same contract as cuteafd_v41_router.
int32_t cuteafd_v41_router_select_logits(float* scores,const float* bias,
    const float* bias_vl,const uint8_t* image_mask,uint32_t* ids,float* routing,
    int32_t rows,int32_t experts,void* stream);
// DeepSeek V4 routing from caller-projected FP32 logits [rows,experts]
// (rewritten in place for score layers): either bias top-k (`bias`) or hash
// layers (`tid2eid` I32 [vocab,topk] indexed by `tokens`), exactly one of
// them; weights are normalized sqrtsoftplus scores times `route_scale`.
int32_t cuteafd_dsv4_router_select(float* logits,const float* bias,const int32_t* tid2eid,
    const uint32_t* tokens,uint32_t* ids,float* routing,int32_t rows,int32_t experts,int32_t topk,
    float route_scale,void* stream);
// Same, with `sigmoid` != 0 selecting sigmoid scores (GLM/Kimi, bias only
// for the choice). Bias routing supports experts <= 1024, topk <= 16; hash
// tables need sqrtsoftplus and retain experts <= 512, topk <= 8. Kimi uses
// experts=896, topk=16, route_scale=1 with no group filtering. Caller projects
// FP32 logits; IDs/weights are row-major [rows,topk], descending corrected
// score with lower expert ID on ties. Weights normalize uncorrected scores.
int32_t cuteafd_router_select(float* logits,const float* bias,const int32_t* tid2eid,
    const uint32_t* tokens,uint32_t* ids,float* routing,int32_t rows,int32_t experts,int32_t topk,
    float route_scale,int32_t sigmoid,void* stream);
// Softmax top-k (Qwen): `logits` FP32 [rows,experts] (read only; rounded to
// BF16 first when `round_bf16`), top-k by logit, weights the top-k softmax
// probabilities renormalized (rounded to BF16 when `round_bf16`) times
// `route_scale`. experts <= 512, topk <= 16.
int32_t cuteafd_router_select_softmax(const float* logits,uint32_t* ids,float* routing,int32_t rows,
    int32_t experts,int32_t topk,float route_scale,int32_t round_bf16,void* stream);
#ifdef __cplusplus
}
#endif
