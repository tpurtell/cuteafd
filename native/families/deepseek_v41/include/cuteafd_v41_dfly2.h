#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

// Per-request context view for V4.1 DFly2 attention (window ring + paged global
// source + captured layer top-512 selected IDs). All pointers live on the
// stream device and remain immutable through completion/replay.
typedef struct cuteafd_v41_dfly2_request {
  const uint8_t* window_values;
  const uint8_t* window_scales;
  const uint8_t* source_values;
  const uint8_t* source_scales;
  const uint32_t* source_pages;
  const int32_t* selected;
  uint64_t window_begin;
  uint64_t committed_end;
  uint64_t source_capacity;
  uint32_t page_stride;
  uint32_t source_fp4;
} cuteafd_v41_dfly2_request_t;

// Initialize dynamic shared-memory launch attributes on the active device.
int32_t cuteafd_v41_dfly2_attention_initialize(void);

// DFly2 two-segment sparse + window (+ optional bidirectional BF16 block)
// attention.
// - query/output: BF16 [requests * block_size, 64, 512]
// - draft_kv: optional BF16 [requests, block_size, 512] (null for attentive
//   correction context-only read where block_size == 1)
// - sink: FP32 [64]
// - requests_dev: device array of `requests` cuteafd_v41_dfly2_request_t
// - alive_out: optional U8 [requests * block_size] (1 if >=1 valid key, else 0)
// - partial: optional FP32 [requests * block_size, parts, 64, 514] split scratch
//   (null for sequential single-pass; when non-null, parts is 1..11)
int32_t cuteafd_v41_dfly2_attention(
    const uint16_t* query, const uint16_t* draft_kv, const float* sink,
    const cuteafd_v41_dfly2_request_t* requests_dev, uint16_t* output,
    uint8_t* alive_out, float* partial, uint64_t scratch_bytes,
    int32_t requests, int32_t block_size, int32_t parts, int32_t source_fp4,
    void* stream);

// Embed [requests, block_size] block: slot 0 = anchor_ids[r], slots 1..B =
// mask_token_id. Writes BF16 block_out [requests * block_size, 5120] and
// anchor_out [requests, 5120].
int32_t cuteafd_v41_dfly2_embed_block(
    const uint16_t* embed_table, const uint32_t* anchor_ids,
    uint16_t* block_out, uint16_t* anchor_out,
    int32_t requests, int32_t block_size, uint32_t mask_token_id,
    void* stream);

// Gather BF16 [requests, 5120] rows from shared embed_table [129280, 5120].
int32_t cuteafd_v41_dfly2_embed_tokens(
    const uint16_t* embed_table, const uint32_t* token_ids, uint16_t* out,
    int32_t requests, void* stream);

// Concatenate BF16 [rows, width_a] and [rows, width_b] -> [rows, width_a+width_b].
int32_t cuteafd_v41_dfly2_concat_bf16(
    const uint16_t* a, const uint16_t* b, uint16_t* out,
    int32_t rows, int32_t width_a, int32_t width_b, void* stream);

// Apply slot-0 MTP fusion (`block_x[r,0] += gate * fused[r]` where
// `positions[r] > 0`), then broadcast 4 identical mHC streams into
// `residual [requests * block_size, 4, 5120]` and set `pre_mix = 0.25f`.
int32_t cuteafd_v41_dfly2_mtp_fuse_and_seed_hc(
    const uint16_t* block_x, const uint16_t* fused, const uint16_t* gate,
    const uint64_t* positions, uint16_t* residual, float* pre_mix,
    int32_t requests, int32_t block_size, void* stream);

// BF16 SwiGLU: silu(gate) * up with optional clamp (limit > 0 clamps gate <=
// limit and up in [-limit, limit]).
int32_t cuteafd_v41_dfly2_swiglu(
    const uint16_t* gate, const uint16_t* up, uint16_t* out,
    int32_t rows, int32_t inter, float limit, void* stream);

// Elementwise BF16 additions (count elements).
int32_t cuteafd_v41_dfly2_add2_bf16(
    const uint16_t* a, const uint16_t* b, uint16_t* out,
    uint64_t count, void* stream);

int32_t cuteafd_v41_dfly2_add3_bf16(
    const uint16_t* a, const uint16_t* b, const uint16_t* c, uint16_t* out,
    uint64_t count, void* stream);

// Extract `block_hidden[r, slot, 0..hidden_size]` -> `out[r, 0..hidden_size]`.
int32_t cuteafd_v41_dfly2_slice_slot(
    const uint16_t* block_hidden, uint16_t* out,
    int32_t requests, int32_t block_size, int32_t slot, int32_t hidden_size,
    void* stream);

// Write `kv_row[r, 0..512]` -> `block_keys[r, slot, 0..512]`.
int32_t cuteafd_v41_dfly2_scatter_kv_slot(
    const uint16_t* kv_row, uint16_t* block_keys,
    int32_t requests, int32_t max_slots, int32_t slot, void* stream);

// Attentive correction gated block attention + context combination for one
// sequential slot `slot` (1..max_slots-1).
// - query: BF16 [requests, 64, 512]
// - block_keys: BF16 [requests, max_slots, 512] (entries 0..=slot initialized)
// - ctx_read: BF16 [requests, 64, 512]
// - ctx_alive: U8 [requests]
// - gate_logits: BF16 [requests, 64]
// - rel_bias: FP32 [64, 16]
// - block_sink: FP32 [64]
// - gate_bias: FP32 [64, 16]
// - out: BF16 [requests, 64, 512]
int32_t cuteafd_v41_dfly2_block_gated_add(
    const uint16_t* query, const uint16_t* block_keys,
    const uint16_t* ctx_read, const uint8_t* ctx_alive,
    const uint16_t* gate_logits, const float* rel_bias,
    const float* block_sink, const float* gate_bias,
    uint16_t* out, int32_t requests, int32_t max_slots, int32_t slot,
    void* stream);

// Single-pass tiled vocab shard reduction: for each row in `logits [rows,
// vocab_cols]` FP32, computes argmax `out_ids[r]`, `out_max[r]`, and
// `out_sum_exp[r] = sum_j exp(logit[j] - out_max[r])`.
// `scratch` must hold at least `rows * 64 * 16` bytes (1024 B/row).
int32_t cuteafd_v41_dfly2_shard_stats_f32(
    const float* logits, void* scratch, uint32_t* out_ids,
    float* out_max, float* out_sum_exp,
    int32_t rows, int32_t vocab_cols, void* stream);

// Single-GPU finalize: converts `(ids, sum_exp)` -> `curr_tokens[r]` and
// scatters into `all_tokens[r * drafts + slot_idx]` and
// `all_probs[r * drafts + slot_idx] = 1.0f / sum_exp[r]`.
int32_t cuteafd_v41_dfly2_finalize_slot_f32(
    const uint32_t* ids, const float* sum_exp,
    uint32_t* curr_tokens, uint32_t* all_tokens, float* all_probs,
    int32_t rows, int32_t drafts, int32_t slot_idx, void* stream);

// Two-GPU vocab shard merge: merges shard 0 `[0, split)` and shard 1
// `[split, 129280)`, writes `curr_tokens[r]` and scatters into
// `all_tokens[r * drafts + slot_idx]` and `all_probs[r * drafts + slot_idx]`.
int32_t cuteafd_v41_dfly2_merge_stats_f32(
    const uint32_t* ids0, const float* max0, const float* sum0,
    const uint32_t* ids1, const float* max1, const float* sum1,
    uint32_t* curr_tokens, uint32_t* all_tokens, float* all_probs,
    int32_t rows, int32_t split, int32_t drafts, int32_t slot_idx,
    void* stream);

// Owned cuBLAS handle bound to a pre-allocated DFly2 workspace plane.
int32_t cuteafd_v41_dfly2_blas_create(void* workspace, uint64_t bytes, void** output);
int32_t cuteafd_v41_dfly2_blas_destroy(void* handle);
int32_t cuteafd_v41_dfly2_gemm_bf16(
    void* handle, const uint16_t* x, const uint16_t* w, uint16_t* out,
    int32_t rows, int32_t k, int32_t n, void* stream);

#ifdef __cplusplus
}
#endif
