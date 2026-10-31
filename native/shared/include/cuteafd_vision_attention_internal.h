#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
int32_t cuteafd_vision_attention_create(int32_t head_dim, void** owner);
int32_t cuteafd_vision_attention_destroy(void* owner);
int32_t cuteafd_vision_attention_launch(void* owner, void* q, void* k, void* v,
    void* out, void* lse, void* cu_seqlens, float scale, void* stream);
/* Pixtral (head 128) pointwise programs, loaded with a head-128 owner. */
int32_t cuteafd_vision_axial_rotary_d128(void* owner, void* q, void* k, void* cos, void* sin,
    void* out_q, void* out_k, int32_t rows, void* stream);
int32_t cuteafd_vision_rms_h2048(void* owner, void* x, void* weight, void* out, int32_t rows, void* stream);
#ifdef __cplusplus
}
#endif
