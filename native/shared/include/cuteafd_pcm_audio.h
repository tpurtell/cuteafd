#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

/* Shared causal PCM pyramid, BF16 weights/activations throughout. All offsets
 * address one admitted weight arena, aligned to 256 bytes. No MiMo RVQ ABI. */
typedef struct {
  uint64_t conv, conv_bias, norm, fc1, fc1_bias, fc2, fc2_bias;
} cuteafd_pcm_block;
typedef struct {
  uint64_t projection, bias, norm;
  cuteafd_pcm_block blocks[2];
} cuteafd_pcm_stage;
typedef struct {
  uint32_t abi_version, max_samples, output_width, reserved;
  uint64_t weight_bytes;
  cuteafd_pcm_stage stages[4];
  uint64_t norm, projection, bias, post_norm;
} cuteafd_pcm_spec;
typedef struct {
  uint64_t weights, scratch, device_allocations, encodes;
} cuteafd_pcm_ledger;
/* Observed BF16 data is device-resident and valid only during the callback.
 * Stages 0..11: patch, block0, block1 per pyramid level; 12: final embedding.
 * token_offset is the global output-token offset of this streaming chunk. */
typedef int32_t (*cuteafd_pcm_observer)(void*, int32_t stage, uint32_t token_offset,
                                      const void*, uint32_t rows, uint32_t width);
int32_t cuteafd_pcm_required(const cuteafd_pcm_spec*, cuteafd_pcm_ledger*);
int32_t cuteafd_pcm_create(const cuteafd_pcm_spec*, int32_t device, uint64_t admitted, void**);
int32_t cuteafd_pcm_upload(void*, uint64_t offset, const void*, uint64_t bytes);
/* PCM is padded on host to ceil(samples/1920)*1920, cast once to BF16.
 * samples is the unpadded positive sample count. Encoding resets all histories;
 * chunks within one call preserve exact convolution inputs, including <6 rows. */
int32_t cuteafd_pcm_encode(void*, const uint16_t* padded_pcm, uint32_t samples,
                          uint16_t* output, uint64_t output_bytes,
                          cuteafd_pcm_observer, void* context);
int32_t cuteafd_pcm_get_ledger(void*, cuteafd_pcm_ledger*);
int32_t cuteafd_pcm_destroy(void*);
#ifdef __cplusplus
}
#endif
