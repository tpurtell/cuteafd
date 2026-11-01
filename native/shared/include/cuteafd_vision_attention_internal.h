#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
// An entry owns its attention and pointwise modules on the creating device.
int32_t cuteafd_vision_attention_create_entry(int32_t head_dim, int32_t windowed,
    const char* pointwise_group, void** owner);
// Scalars are encoded as their low 32/64 bits, matching the exported ABI.
int32_t cuteafd_vision_pointwise_launch(void* owner, const char* name,
    void* const* pointers, const uint64_t* scalars, void* stream);
int32_t cuteafd_vision_attention_destroy(void* owner);
int32_t cuteafd_vision_attention_launch(void* owner, void* q, void* k, void* v,
    void* out, void* lse, void* cu_seqlens, float scale, void* stream);
#ifdef __cplusplus
}
#endif
