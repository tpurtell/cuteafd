// CPU-only fake native DSO. Its observable destructor proves actual dlclose;
// no CUDA, device, or transport implementation is linked into this fixture.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../../../../../native/shared/cuda/expert_pack_layout.h"
#include "cuteafd_mxfp4_layouts.h"

struct host_buffer { void *ptr; size_t bytes; uint64_t flags; };
static char *event_path;
static int quiet_events;
void fixture_quiet_events(int quiet) { quiet_events = quiet; }
static int pack_status, drain_status, drain_calls, drain_after;
/* Per thread, as CUDA's current device: two GPU loader threads bind their own. */
static _Thread_local int current_device;
static int bind_after = -1, bind_calls;
void fixture_configure_bind_after(int after) { bind_after = after; bind_calls = 0; }
static size_t memory_total[8], memory_used[8], graph_bytes;
static int immediate_copies;
static int queued_experts;
static int queue_expert(void **ptrs, int rows, void *stream);
static int queue_segment_expert(void **ptrs, int rows, void *stream);
void fixture_configure_queued_experts(void) { queued_experts = 1; }

void fixture_configure_immediate_copies(void) { immediate_copies = 1; }

void fixture_configure_memory(int device, size_t total, size_t used, size_t graph) {
    memory_total[device] = total;
    memory_used[device] = used;
    graph_bytes = graph;
}

struct device_info {
    int device_id, cuda_available, compute_capability_major, compute_capability_minor;
    int integrated, can_map_host_memory, unified_addressing;
    uint64_t total_memory_bytes;
    char name[128], driver_version[64], runtime_version[64];
};
int cuteafd_cuda_device_info(int device, struct device_info *info) {
    memset(info, 0, sizeof(*info));
    info->device_id = device;
    info->cuda_available = 1;
    info->compute_capability_major = 12;
    info->compute_capability_minor = 0;
    info->total_memory_bytes = memory_total[device];
    return 0;
}

int cuteafd_cuda_get_device(int *device) { *device = current_device; return 0; }
int cuteafd_cuda_memory_info(size_t *free_bytes, size_t *total) {
    *total = memory_total[current_device];
    *free_bytes = *total - memory_used[current_device];
    return 0;
}

static void event(char value) {
    if (quiet_events) return;
    FILE *file = fopen(event_path, "a");
    if (!file || fputc(value, file) == EOF || fclose(file)) abort();
}

void fixture_set_event_path(const char *path) {
    event_path = strdup(path);
    if (!event_path) abort();
}

__attribute__((destructor)) static void fixture_unloaded(void) {
    if (event_path) {
        event('U');
        free(event_path);
    }
}

int cuteafd_rdma_rc_endpoint_try_poll(void *handle, uint32_t sends,
                                    uint32_t recvs, void *out) {
    (void)handle; (void)sends; (void)recvs; (void)out;
    return 0;
}

int cuteafd_set_expert_hidden(uint32_t hidden) { (void)hidden; return 0; }
int cuteafd_cuda_enable_peer(int peer) { (void)peer; return 0; }
int cuteafd_tp2_native_initialize(void) { return 0; }
int cuteafd_peer_exchange_initialize(void) { return 0; }


/* Head fixture records the scratch selected by the real Rust handle binding. */
struct fixture_head { void *workspace; int width, rows, vocab; };
int cuteafd_vocabulary_head_create_vocab(void *workspace, uint64_t bytes,
        int width, int rows, int vocab, void **out) {
    if (!workspace || bytes < (4u << 20) || width <= 0 || rows <= 0 || vocab <= 0) return 1;
    struct fixture_head *head = malloc(sizeof(*head));
    if (!head) return 1;
    *head = (struct fixture_head){workspace, width, rows, vocab};
    *out = head;
    return 0;
}
int cuteafd_vocabulary_head_launch_width(void *opaque, const uint16_t *input,
        const uint16_t *weight, float *output, int rows, void *stream) {
    (void)weight; (void)stream;
    struct fixture_head *head = opaque;
    if (!head || rows <= 0 || rows > head->rows) return 1;
    /* Identity marker proves both the input and per-handle scratch window. */
    ((uint16_t *)head->workspace)[0] = input[0];
    output[0] = (float)input[0];
    return 0;
}
int cuteafd_v41_markov_destroy(void *head) { free(head); return 0; }


/* Static native expert ABI: CPU slot-ownership tests allocate real fake-device
 * buffers but never launch expert math. Each loaded module caches these words. */
static uint32_t fp8_words[16];
static uint64_t fp8_scratch[6];
static int fp8_segments;
void fixture_configure_fp8_segments(void) { fp8_segments = 1; }
void fixture_configure_fp8(const uint32_t *words, const uint64_t *scratch) {
    memcpy(fp8_words, words, sizeof(fp8_words));
    memcpy(fp8_scratch, scratch, sizeof(fp8_scratch));
}
int cuteafd_fp8moe_info(uint32_t *words, uint32_t count) {
    if (count != 16) return 1;
    memcpy(words, fp8_words, sizeof(fp8_words)); return 0;
}
int cuteafd_fp8moe_scratch_bytes(uint32_t capacity, uint64_t *bytes) {
    for (uint32_t i = 0; i < fp8_words[9]; ++i) {
        if (fp8_words[10 + i] == capacity) { *bytes = fp8_scratch[i]; return 0; }
    }
    return 1;
}
int cuteafd_fp8moe_create(void **out) { *out = malloc(1); return *out ? 0 : 1; }
int cuteafd_fp8moe_launch(void *ctx, uint32_t capacity, void **ptrs, int rows, void *stream) {
    (void)ctx; (void)capacity;
    if (queued_experts) return queue_expert(ptrs, rows, stream);
    return 1; /* Ordinary lifetime tests never launch expert math. */
}
int cuteafd_fp8moe_segment_info(uint32_t form, uint32_t *maximum, uint32_t *minimum, uint32_t *act) {
    if (form > 2) return 1;
    *maximum = fp8_segments ? 16384 : 0;
    *minimum = fp8_segments ? 769 : 0;
    *act = fp8_segments ? 2 * fp8_words[2] : 0; return 0;
}
int cuteafd_fp8moe_segment_scratch_bytes(uint32_t rows, uint64_t *bytes) {
    if (!fp8_segments || !rows || rows > 16384) return 1;
    uint64_t pairs = (uint64_t)rows * fp8_words[4], bound = 16384ull * fp8_words[4];
    uint64_t experts = fp8_words[3];
    uint64_t tiles = (pairs + 127) / 128 + (pairs < experts ? pairs : experts);
    uint64_t max_tiles = (bound + 127) / 128 + (bound < experts ? bound : experts);
    uint64_t regions[] = {4 * (4 + 3 * experts + max_tiles), 4 * pairs, 4 * pairs,
        tiles * 128 * 2 * fp8_words[2], pairs * fp8_words[1] * 2};
    *bytes = 0;
    for (size_t i = 0; i < 5; ++i) *bytes += (regions[i] + 1023) / 1024 * 1024;
    return 0;
}
int cuteafd_fp8moe_launch_segments(void *ctx, void **ptrs, int rows, void *stream) {
    (void)ctx;
    return queued_experts && fp8_segments ? queue_segment_expert(ptrs, rows, stream) : 1;
}
void cuteafd_fp8moe_destroy(void *ctx) { free(ctx); }

int cuteafd_alloc_host_buffer(size_t bytes, struct host_buffer *out) {
    out->ptr = malloc(bytes);
    if (!out->ptr) return 1;
    out->bytes = bytes;
    out->flags = 1;
    event('A');
    return 0;
}

int cuteafd_free_host_buffer(struct host_buffer *buffer) {
    event('F');
    free(buffer->ptr);
    memset(buffer, 0, sizeof(*buffer));
    return 0;
}

struct device_buffer { void *ptr; size_t bytes; int device_id; uint64_t flags; };

int cuteafd_cuda_host_buffer_device_alias(struct host_buffer host, struct device_buffer *out) {
    if (!host.ptr || !host.bytes || !out) return 1;
    *out = (struct device_buffer){ .ptr = host.ptr, .bytes = host.bytes,
        .device_id = current_device, .flags = 2 };
    return 0;
}

void fixture_configure_pack(int pack, int drain) {
    pack_status = pack;
    drain_status = drain;
    drain_calls = 0;
    drain_after = 0;
}

void fixture_configure_drain_after(int after) { drain_after = after; }

int cuteafd_alloc_device_buffer(size_t bytes, struct device_buffer *out) {
    if (posix_memalign(&out->ptr, 256, bytes)) return 1;
    /* Zeroed so operand digests never read unspecified padding. */
    memset(out->ptr, 0, bytes);
    out->bytes = bytes;
    out->device_id = current_device;
    out->flags = 1;
    memory_used[current_device] += bytes;
    event('D');
    return 0;
}

int cuteafd_free_device_buffer(struct device_buffer *buffer) {
    event('d');
    if (buffer->flags == 1) memory_used[buffer->device_id] -= buffer->bytes;
    free(buffer->ptr);
    memset(buffer, 0, sizeof(*buffer));
    return 0;
}

int cuteafd_alloc_managed_device_buffer(size_t bytes, struct device_buffer *out) {
    // Host-resident managed pages deliberately do not reduce physical free.
    out->ptr = malloc(bytes);
    if (!out->ptr) return 1;
    out->bytes = bytes;
    out->device_id = current_device;
    out->flags = 2;
    event('D');
    return 0;
}

struct graph_exec { int device; size_t bytes; };
int cuteafd_cuda_graph_end_capture(void *stream, void **out) {
    (void)stream;
    struct graph_exec *graph = malloc(sizeof(*graph));
    if (!graph) return 1;
    graph->device = current_device;
    graph->bytes = graph_bytes;
    memory_used[current_device] += graph_bytes;
    *out = graph;
    event('G');
    return 0;
}
int cuteafd_cuda_graph_exec_destroy(void *handle) {
    struct graph_exec *graph = handle;
    memory_used[graph->device] -= graph->bytes;
    free(graph);
    event('g');
    return 0;
}

int cuteafd_fp8_w8a16_pack(const void *source, void *packed, void *scale,
                         int n, int k, int rule, void *stream) {
    (void)source; (void)packed; (void)scale; (void)n; (void)k;
    (void)rule; (void)stream;
    event('P');
    return pack_status;
}

struct upload { void *dst; const void *src; size_t bytes; };
struct upload_event { struct upload *copies; size_t count, fence; };

/* Opt-in stream interpreter: copies and fake expert reads execute only when
 * explicitly advanced, so tests can submit N+1 while N is still outstanding. */
struct queued_op {
    int kind;
    void *stream, *dst;
    const void *src;
    size_t bytes, dst_pitch, src_pitch, rows, hidden, topk;
    void *ptrs[11];
};
static struct queued_op queued_ops[65536];
static size_t queued_count, queued_done;
static int enqueue(struct queued_op op) {
    if (queued_count == 65536) return 1;
    queued_ops[queued_count++] = op;
    return 0;
}
static int queue_expert(void **ptrs, int rows, void *stream) {
    struct queued_op op = { .kind = 1, .stream = stream, .rows = (size_t)rows,
        .hidden = fp8_words[1], .topk = fp8_words[4] };
    memcpy(op.ptrs, ptrs, sizeof(op.ptrs));
    return enqueue(op);
}
static int queue_segment_expert(void **ptrs, int rows, void *stream) {
    struct queued_op op = { .kind = 2, .stream = stream, .rows = (size_t)rows,
        .hidden = fp8_words[1], .topk = fp8_words[4] };
    memcpy(op.ptrs, ptrs, sizeof(op.ptrs));
    return enqueue(op);
}
static void advance_queue(size_t until) {
    while (queued_done < until) {
        struct queued_op *op = &queued_ops[queued_done++];
        if (op->kind == 0) {
            for (size_t row = 0; row < op->rows; ++row)
                memcpy((char *)op->dst + row * op->dst_pitch,
                    (const char *)op->src + row * op->src_pitch, op->bytes);
        } else {
            const int32_t *ids = op->ptrs[1];
            const float *gates = op->ptrs[2];
            uint16_t *out = op->ptrs[9];
            const uint8_t *input = op->ptrs[0];
            size_t stride = op->hidden + op->hidden / 32;
            for (size_t row = 0; row < op->rows; ++row) {
                const uint8_t *row_input = op->kind == 2
                    ? (const uint8_t *)(uintptr_t)((const uint64_t *)input)[row] : input + row * stride;
                uint16_t *row_out = op->kind == 2
                    ? (uint16_t *)(uintptr_t)((const uint64_t *)out)[row] : out + row * op->hidden;
                uint16_t value = row_input[0];
                for (size_t route = 0; route < op->topk; ++route) {
                    size_t at = row * op->topk + route;
                    value += (uint16_t)((ids[at] + 1) * (int)gates[at]);
                }
                for (size_t col = 0; col < op->hidden; ++col)
                    row_out[col] = value;
            }
        }
    }
}
void fixture_advance_queued_experts(size_t operations) {
    size_t until = queued_done + operations;
    advance_queue(until < queued_count ? until : queued_count);
}
static struct upload pending_uploads[65536];
static size_t pending_count;
static void finish_uploads(struct upload_event *upload) {
    for (size_t i = 0; i < upload->count; ++i) {
        struct upload *copy = &upload->copies[i];
        memcpy(copy->dst, copy->src, copy->bytes);
    }
    free(upload->copies);
    upload->copies = NULL;
    upload->count = 0;
}
int cuteafd_register_host_mapping(void *ptr, size_t bytes) {
    if (!ptr || !bytes) return 1;
    event('G'); return 0;
}
int cuteafd_unregister_host_mapping(void *ptr) {
    if (!ptr) return 1;
    event('g'); return 0;
}
int cuteafd_cuda_stream_create(void **out) {
    *out = (void *)1;
    event('V'); return 0;
}
static void *upload_events[128];
static size_t upload_event_count;

int cuteafd_cuda_event_create_ordering(void **out) {
    if (upload_event_count == 128) return 1;
    *out = calloc(1, sizeof(struct upload_event));
    if (!*out) return 1;
    upload_events[upload_event_count++] = *out;
    event('C');
    return 0;
}
int cuteafd_cuda_event_create(void **out) {
    return cuteafd_cuda_event_create_ordering(out);
}

int cuteafd_cuda_stream_wait_event(void *stream, void *handle) {
    (void)stream; (void)handle;
    event('W');
    return 0;
}

int cuteafd_cuda_zero_bytes_async(void *dst, size_t bytes, void *stream) {
    (void)stream;
    memset(dst, 0, bytes);
    return 0;
}

int cuteafd_copy_h2d_async(struct device_buffer dst, const void *source,
                         size_t bytes, void *stream);

int cuteafd_copy_d2d_async(struct device_buffer dst, struct device_buffer src,
                           size_t bytes, void *stream) {
    if (bytes > dst.bytes || bytes > src.bytes) return 1;
    if (!immediate_copies) return cuteafd_copy_h2d_async(dst, src.ptr, bytes, stream);
    if (dst.device_id != current_device) return 1;
    memcpy(dst.ptr, src.ptr, bytes);
    event('C');
    return pack_status;
}

int cuteafd_copy_d2h(void *dst, struct device_buffer src, size_t bytes) {
    if (bytes > src.bytes) return 1;
    memcpy(dst, src.ptr, bytes);
    return 0;
}

int cuteafd_copy_d2h_async(void *dst, struct device_buffer src, size_t bytes, void *stream) {
    (void)stream;
    if (!immediate_copies || bytes > src.bytes || src.device_id != current_device) return 1;
    memcpy(dst, src.ptr, bytes);
    event('J');
    return 0;
}

int cuteafd_copy_h2d_async(struct device_buffer dst, const void *source,
                         size_t bytes, void *stream) {
    (void)stream;
    if (bytes > dst.bytes) return 1;
    if (queued_experts) return enqueue((struct queued_op){ .stream = stream,
        .dst = dst.ptr, .src = source, .bytes = bytes, .rows = 1 });
    if (immediate_copies) {
        if (dst.device_id != current_device) return 1;
        memcpy(dst.ptr, source, bytes);
    } else {
        if (pending_count == 65536) return 1;
        pending_uploads[pending_count++] = (struct upload){dst.ptr, source, bytes};
    }
    event('Q');
    return 0;
}
int cuteafd_copy_h2d_batch_async(const struct device_buffer *dsts, const void *const *sources,
                                 const size_t *bytes, size_t count, void *stream) {
    for (size_t index = 0; index < count; ++index) {
        if (cuteafd_copy_h2d_async(dsts[index], sources[index], bytes[index], stream)) return 1;
    }
    return 0;
}
int cuteafd_cuda_event_record(void *handle, void *stream) {
    (void)stream;
    struct upload_event *upload = handle;
    if (queued_experts) { upload->fence = queued_count; return 0; }
    if (upload->count) return 1;
    upload->copies = malloc(pending_count * sizeof(struct upload));
    if (pending_count && !upload->copies) return 1;
    if (pending_count) memcpy(upload->copies, pending_uploads, pending_count * sizeof(struct upload));
    upload->count = pending_count;
    pending_count = 0;
    event('R');
    return 0;
}
int cuteafd_cuda_event_query(void *handle, int *ready) {
    struct upload_event *upload = handle;
    *ready = queued_experts ? queued_done >= upload->fence : upload->count == 0;
    return 0;
}
int cuteafd_cuda_event_elapsed_ms(void *start, void *end, float *ms) {
    struct upload_event *first = start, *last = end;
    *ms = (float)(last->fence - first->fence) * 0.001f;
    return 0;
}
int cuteafd_cuda_event_synchronize(void *handle) {
    if (queued_experts) { advance_queue(((struct upload_event *)handle)->fence); return 0; }
    event('S');
    int status = ++drain_calls > drain_after ? drain_status : 0;
    if (status) return status;
    for (size_t i = 0; i < upload_event_count; ++i) {
        if (upload_events[i] == handle) {
            struct upload_event *upload = handle;
            finish_uploads(upload);
        }
    }
    return 0;
}

int cuteafd_cuda_event_destroy(void *handle) {
    for (size_t i = 0; i < upload_event_count; ++i) {
        if (upload_events[i] == handle) { free(handle); upload_events[i] = NULL; }
    }
    event('E');
    return 0;
}

int cuteafd_cuda_stream_query(void *stream, int *ready) {
    (void)stream;
    *ready = queued_experts ? queued_done == queued_count : immediate_copies;
    return 0;
}

int cuteafd_cuda_stream_synchronize(void *stream) {
    (void)stream;
    if (queued_experts) { advance_queue(queued_count); return 0; }
    event('S');
    int status = ++drain_calls > drain_after ? drain_status : 0;
    if (status) return status;
    for (size_t i = 0; i < upload_event_count; ++i) {
        if (upload_events[i]) finish_uploads(upload_events[i]);
    }
    for (size_t i = 0; i < pending_count; ++i) {
        struct upload *copy = &pending_uploads[i];
        memcpy(copy->dst, copy->src, copy->bytes);
    }
    pending_count = 0;
    return 0;
}

int cuteafd_cuda_stream_destroy(void *stream) {
    (void)stream;
    event('T');
    return 0;
}

int cuteafd_cuda_set_device(int device) {
    if (bind_after >= 0 && bind_calls++ >= bind_after) return 1;
    current_device = device;
    return 0;
}

int cuteafd_copy_h2d(struct device_buffer dst, const void *source, size_t bytes) {
    if (bytes > dst.bytes) return 1;
    memcpy(dst.ptr, source, bytes);
    event('H');
    return 0;
}

int cuteafd_copy_d2d(struct device_buffer dst, struct device_buffer src, size_t bytes) {
    if (bytes > dst.bytes || bytes > src.bytes) return 1;
    memcpy(dst.ptr, src.ptr, bytes); return 0;
}
int cuteafd_cuda_zero_bytes(void *dst, size_t bytes) { memset(dst, 0, bytes); return 0; }
int cuteafd_zero_pitched_async(void *dst, uint64_t bytes, uint64_t pitch,
    uint64_t width, uint64_t rows, void *stream) {
    (void)stream;
    if (!dst || !width || !rows || pitch < width || width > bytes ||
        rows - 1 > (bytes - width) / pitch) return 1;
    for (uint64_t row = 0; row < rows; ++row)
        memset((uint8_t *)dst + row * pitch, 0, width);
    event('Z');
    return pack_status;
}
int cuteafd_pack_expert_plane_async(const uint8_t *source, uint64_t source_bytes,
    uint8_t *destination, uint64_t destination_bytes, uint32_t intermediate,
    uint32_t plane, void *stream) {
    int status = cuteafd_cuda_stream_synchronize(stream);
    if (status) return status;
    if (pack_status) return pack_status;
    uint32_t padded = cuteafd_mxfp4_stored_width(5120, intermediate);
    if (!padded) return 1;
    int compact = padded == intermediate && intermediate % 128;
    uint32_t n = plane < 2 ? 2 * padded : 5120;
    uint32_t k = plane < 2 ? 5120 : intermediate;
    int scales = plane % 2;
    uint32_t element = scales ? 1 : 4, divisor = scales ? 32 : 8;
    for (uint64_t index = 0; index < destination_bytes / element; ++index) {
        uint32_t row, col;
        cuteafd_expert_pack_layout_coordinates(index, n, k,
            compact && plane < 2 ? padded : 0, scales, compact, &row, &col);
        uint64_t at = ((uint64_t)row * (k / divisor) + col) * element;
        if (row < n && col < k / divisor && at + element <= source_bytes)
            memcpy(destination + index * element, source + at, element);
        else memset(destination + index * element, 0, element);
    }
    event('P'); return 0;
}
int cuteafd_cuda_nvfp4_swizzle_scale_async(struct device_buffer src, struct device_buffer dst,
    size_t rows, size_t cols, void *stream) {
    int status = cuteafd_cuda_stream_synchronize(stream);
    if (status) return status;
    if (pack_status) return pack_status;
    size_t rp = (rows + 127) / 128 * 128, cp = (cols + 3) / 4 * 4;
    if (src.bytes < rows * cols || dst.bytes < rp * cp) return 1;
    for (size_t row = 0; row < rp; ++row) for (size_t col = 0; col < cp; ++col)
        ((uint8_t *)dst.ptr)[cuteafd_nvfp4_scale_offset(row, col, cp)] =
            row < rows && col < cols ? ((uint8_t *)src.ptr)[row * cols + col] : 0;
    event('P'); return 0;
}

int cuteafd_copy_d2d_2d_async(struct device_buffer dst, size_t dst_pitch,
                            struct device_buffer src, size_t src_pitch,
                            size_t width, size_t rows, void *stream) {
    if (queued_experts) return enqueue((struct queued_op){ .stream = stream,
        .dst = dst.ptr, .src = src.ptr, .bytes = width,
        .dst_pitch = dst_pitch, .src_pitch = src_pitch, .rows = rows });
    for (size_t r = 0; r < rows; ++r)
        memcpy((char *)dst.ptr + r * dst_pitch,
               (const char *)src.ptr + r * src_pitch, width);
    event('C');
    return pack_status;
}

int cuteafd_fp8_grid_kmajor(const void *grid, void *kmajor, int64_t rows, int64_t kb, void *stream) {
    (void)stream;
    for (int64_t b = 0; b < kb; ++b)
        for (int64_t r = 0; r < rows; ++r)
            ((float *)kmajor)[b * rows + r] = ((const float *)grid)[(r / 128) * kb + b];
    event('K');
    return pack_status;
}

int cuteafd_fp8_block_dequant(const void *weight, const void *scale, void *out,
                            int rows, int cols, void *stream) {
    (void)weight; (void)scale; (void)out; (void)rows; (void)cols; (void)stream;
    event('B');
    return pack_status;
}

int cuteafd_fp8_row_quant_rule(const void *weight, void *out, void *scale,
                             int rows, int cols, int rule, void *stream) {
    (void)weight; (void)out; (void)scale; (void)rows; (void)cols;
    (void)rule; (void)stream;
    event('Q');
    return pack_status;
}

int cuteafd_last_error(char *out, size_t bytes) {
    snprintf(out, bytes, "injected CPU fixture failure");
    return 0;
}

/* EXL3 contexts model untracked driver residency and device-bound destruction. */
struct exl3_context { int device; size_t bytes; };
static int exl3_create(void **out, size_t bytes, char marker) {
    struct exl3_context *context = malloc(sizeof(*context));
    if (!context) return 1;
    context->device = current_device;
    context->bytes = bytes;
    memory_used[current_device] += bytes;
    *out = context;
    event(marker);
    return 0;
}
static void exl3_destroy(void *handle, char marker) {
    struct exl3_context *context = handle;
    if (context->device != current_device) abort();
    memory_used[current_device] -= context->bytes;
    event(marker);
    free(context);
}
int cuteafd_exl3_info(uint32_t *out, uint32_t count) {
    const uint32_t words[16] = {2,5120,1152,6,1,6,2,1,1,1,1,3,4,0,0,4};
    if (count != 16) return 1;
    memcpy(out, words, sizeof(words));
    return 0;
}
int cuteafd_exl3_create(void **out) { return exl3_create(out, 4096, 'K'); }
void cuteafd_exl3_destroy(void *handle) { exl3_destroy(handle, 'k'); }
int cuteafd_exl3_core(void *handle, void *const *ptrs, const int *scalars, void *stream) {
    (void)handle; (void)ptrs; (void)scalars; (void)stream; return 0;
}
int cuteafd_exl3_sum(void *handle, void *const *ptrs, const int *scalars, void *stream) {
    return cuteafd_exl3_core(handle, ptrs, scalars, stream);
}
int cuteafd_exl3_routes_create(void **out) { return exl3_create(out, 1024, 'R'); }
int cuteafd_exl3_routes_destroy(void *handle) { exl3_destroy(handle, 'r'); return 0; }
int cuteafd_exl3_routes_launch(void *handle, void *const *ptrs,
                             const uint64_t *bytes, int rows, void *stream) {
    (void)handle; (void)ptrs; (void)bytes; (void)rows; (void)stream; return 0;
}
int cuteafd_exl3_wire_initialize(void **out) { return exl3_create(out, 512, 'W'); }
void cuteafd_exl3_wire_destroy(void *handle) { exl3_destroy(handle, 'w'); }
int cuteafd_exl3_wire_decode(void *handle, const uint8_t *input, uint64_t in_bytes,
                           uint16_t *output, uint64_t out_bytes, uint32_t rows, void *stream) {
    (void)handle; (void)input; (void)in_bytes; (void)output;
    (void)out_bytes; (void)rows; (void)stream; return 0;
}
