// CPU-only fake native DSO. Its observable destructor proves actual dlclose;
// no CUDA, device, or transport implementation is linked into this fixture.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <stdatomic.h>
#include <time.h>
#include <sys/mman.h>
#include "../../../../../native/shared/cuda/expert_pack_layout.h"
#include "cuteafd_mxfp4_layouts.h"

struct host_buffer { void *ptr; size_t bytes; uint64_t flags; };
static char *event_path;
static int quiet_events;
void fixture_quiet_events(int quiet) { quiet_events = quiet; }
static int pack_status, drain_status, drain_calls, drain_after;
/* Per thread, as CUDA's current device: two GPU loader threads bind their own. */
static _Thread_local int current_device;
static int strict_device_free, free_owner = -1, free_current = -1;
static size_t device_free_mismatches;
void fixture_strict_device_free(int enabled) {
    strict_device_free = enabled;
    device_free_mismatches = 0;
    free_owner = free_current = -1;
}
size_t fixture_device_free_mismatch(int *owner, int *current) {
    *owner = free_owner;
    *current = free_current;
    return device_free_mismatches;
}
static int bind_after = -1, bind_calls;
void fixture_configure_bind_after(int after) { bind_after = after; bind_calls = 0; }
static size_t memory_total[8], memory_used[8], graph_bytes;
static int immediate_copies;
/* Untouched, per-owner virtual extents: allocation tests never launch math. */
static int extent_only;
static int peer_abort_initialization;
void fixture_peer_abort_initialization(void) { peer_abort_initialization = 1; }
static size_t mapped_bytes;
size_t fixture_mapped_bytes(void) { return mapped_bytes; }
void fixture_extent_only(void) { extent_only = 1; immediate_copies = 1; }
struct fixture_program_info {
    char name[64]; uint32_t pointers, scalars; char scalar_kinds[16];
};
static struct fixture_program_info draft_programs[2048];
static struct { uint32_t ready; char event; } program_replies[2048];
static uint32_t draft_program_count;
static void event(char value);
int fixture_register_program(uint32_t index, const char *name, uint32_t pointers,
                             const char *scalar_kinds, uint32_t scalars) {
    if (index >= 2048 || strlen(name) >= 64 || scalars > 16) return 1;
    struct fixture_program_info *info = &draft_programs[index];
    memset(info, 0, sizeof(*info));
    memset(&program_replies[index], 0, sizeof(program_replies[index]));
    memcpy(info->name, name, strlen(name));
    info->pointers = pointers; info->scalars = scalars;
    memcpy(info->scalar_kinds, scalar_kinds, scalars);
    draft_program_count = index + 1;
    return 0;
}
int fixture_draft_projection(uint32_t index, uint32_t k, uint32_t n) {
    if (!extent_only || index >= 32) return 1;
    struct fixture_program_info *info = &draft_programs[index];
    memset(info, 0, sizeof(*info));
    snprintf(info->name, sizeof(info->name), "draft_bf16_k%u_n%u", k, n);
    info->pointers = 3; info->scalars = 1; info->scalar_kinds[0] = 'i';
    draft_program_count = index + 1;
    return 0;
}
uint32_t cuteafd_dsv4_program_count(void) { return draft_program_count; }
int cuteafd_dsv4_program_info(uint32_t index, void *out) {
    if (index >= draft_program_count || !out) return 1;
    memcpy(out, &draft_programs[index], sizeof(draft_programs[index])); return 0;
}
int cuteafd_dsv4_program_load(uint32_t index) { return index >= draft_program_count; }
int fixture_register_reply(uint32_t index, uint32_t ready, char dispatch_event) {
    if (index >= draft_program_count || !ready || !dispatch_event) return 1;
    program_replies[index].ready = ready;
    program_replies[index].event = dispatch_event;
    return 0;
}
int cuteafd_dsv4_program_launch(uint32_t index, void **ptrs, const uint64_t *scalars, void *stream) {
    (void)stream;
    if (index >= draft_program_count || !program_replies[index].ready || !ptrs || !ptrs[0]
            || !scalars || !scalars[0] || *(uint32_t *)ptrs[0] != program_replies[index].ready) return 1;
    event(program_replies[index].event);
    return 0;
}
int cuteafd_rtx_tp2_partial_async(const float *routed, const uint16_t *shared,
    void *out, uint64_t count, uint32_t dtype, void *stream) {
    (void)routed; (void)shared; (void)out; (void)count; (void)dtype; (void)stream; return 1;
}
int cuteafd_rtx_tp2_sum_async(const void *left, const void *right, uint16_t *out,
    uint64_t count, uint32_t dtype, void *stream) {
    (void)left; (void)right; (void)out; (void)count; (void)dtype; (void)stream; return 1;
}
/* Allocation-only expert construction resolves these ABIs but never launches. */
int64_t cuteafd_l2_cache_bytes(void) { return 96 * 1024 * 1024; }
int cuteafd_peer_copy_initialize(void) { return extent_only ? 0 : 1; }
int cuteafd_peer_copy_async(void *dst, const void *src, uint64_t bytes, void *stream) {
    (void)dst; (void)src; (void)bytes; (void)stream; return 1;
}
int cuteafd_peer_copy_rows_async(void *dst, const void *src, uint64_t bytes,
    uint64_t rows, uint64_t dst_pitch, uint64_t src_pitch, void *stream) {
    (void)dst; (void)src; (void)bytes; (void)rows;
    (void)dst_pitch; (void)src_pitch; (void)stream; return 1;
}
int cuteafd_peer_abort_initialize(void) { return extent_only || peer_abort_initialization ? 0 : 1; }
int cuteafd_peer_wait_abortable(const uint32_t *flag, uint32_t *state,
    const uint32_t *aborted, void *stream) {
    (void)flag; (void)state; (void)aborted; (void)stream; return 1;
}
int cuteafd_peer_abort_publish(uint32_t *aborted, void *stream) {
    (void)aborted; (void)stream; return 1;
}
static int trace_stream_queries;
void fixture_trace_stream_queries(void) { trace_stream_queries = 1; }
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
static int native_initializations;
int fixture_native_initializations(void) { return native_initializations; }
int cuteafd_tp2_native_initialize(void) { ++native_initializations; return 0; }
int cuteafd_peer_exchange_initialize(void) { return 0; }


/* Head fixture records the scratch selected by the real Rust handle binding. */
struct fixture_head { void *workspace; int width, rows, vocab; };
int cuteafd_vocabulary_head_create_vocab(void *workspace, uint64_t bytes,
        int width, int rows, int vocab, int mode, void **out) {
    (void)mode;
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
int cuteafd_v41_vocabulary_shard_create(void *workspace, uint64_t bytes, int vocab, int mode, void **out) {
    return cuteafd_vocabulary_head_create_vocab(workspace, bytes, 5120, 128, vocab, mode, out);
}
int cuteafd_v41_vocabulary_head_launch(void *head, const uint16_t *input,
        const uint16_t *weight, float *output, int rows, void *stream) {
    return cuteafd_vocabulary_head_launch_width(head, input, weight, output, rows, stream);
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

/* Dispatch-only fused-reply stand-ins: verify core/epilogue ordering and the
 * shared scratch views, not CUDA numerics (those have separate device gates). */
int cuteafd_fp8moe_reply_layout(uint32_t capacity, uint32_t form, uint32_t rows,
        uint32_t segmented, uint64_t *out, uint32_t count) {
    (void)form; (void)segmented;
    if (!rows || rows > capacity || count != 4) return 1;
    out[0] = 1; out[1] = 0;
    out[2] = ((uint64_t)rows * fp8_words[4] * 4 + 1023) / 1024 * 1024;
    out[3] = (uint64_t)rows * fp8_words[4] * fp8_words[1] * 2;
    return 0;
}
static int reply_core(void **ptrs, int rows, int segmented) {
    uint64_t layout[4];
    if (rows < 1 || cuteafd_fp8moe_reply_layout((uint32_t)rows, 0,
            (uint32_t)rows, (uint32_t)segmented, layout, 4)) return 1;
    *(uint32_t *)((char *)ptrs[10] + layout[2]) = 0x1166;
    event(segmented ? 's' : 'c');
    return 0;
}
int cuteafd_fp8moe_launch_core(void *ctx, uint32_t capacity, void **ptrs, int rows, void *stream) {
    (void)ctx; (void)capacity; (void)stream;
    return reply_core(ptrs, rows, 0);
}
int cuteafd_fp8moe_launch_segments_core(void *ctx, void **ptrs, int rows, void *stream) {
    (void)ctx; (void)stream;
    return reply_core(ptrs, rows, 1);
}

/* Mirrors the native ABI: every allocation names a kind (1..12) and an owner,
 * and the attempted tag is retained (owner copied) for failure reports. */
static __thread uint32_t attempted_kind;
static __thread char *attempted_owner;
static int valid_tag(uint32_t kind, const char *owner) {
    if (!(kind >= 1 && kind <= 12 && owner && owner[0])) return 0;
    attempted_kind = kind;
    free(attempted_owner);
    attempted_owner = strdup(owner);
    return attempted_owner != NULL;
}
/* Native returns 2 (CUTEAFD_STATUS_BUFFER_TOO_SMALL) with the required size. */
int cuteafd_last_allocation_tag(uint32_t *kind, char *owner, size_t owner_bytes, size_t *required) {
    if (!kind || !required || (!owner && owner_bytes)) return 1;
    const char *value = attempted_owner ? attempted_owner : "";
    *kind = attempted_kind;
    *required = strlen(value) + 1;
    if (*required > owner_bytes) return 2;
    memcpy(owner, value, *required);
    return 0;
}

int cuteafd_alloc_host_buffer(size_t bytes, struct host_buffer *out, uint32_t kind, const char *owner) {
    if (!valid_tag(kind, owner)) return 1;
    out->ptr = malloc(bytes);
    if (!out->ptr) return 1;
    out->bytes = bytes;
    out->flags = 1 | 4;
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

int cuteafd_cuda_dmabuf_allocation_granularity(size_t *out) {
    *out = 2 * 1024 * 1024;
    return 0;
}

int cuteafd_alloc_device_buffer(size_t bytes, struct device_buffer *out, uint32_t kind, const char *owner) {
    if (!valid_tag(kind, owner)) return 1;
    if (extent_only) {
        out->ptr = mmap(NULL, bytes, PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
        if (out->ptr == MAP_FAILED) return 1;
        mapped_bytes += bytes;
    } else {
        if (posix_memalign(&out->ptr, 256, bytes)) return 1;
        /* Operand digests must not read unspecified padding. */
        memset(out->ptr, 0, bytes);
    }
    out->bytes = bytes;
    out->device_id = current_device;
    out->flags = 1;
    memory_used[current_device] += bytes;
    event('D');
    return 0;
}

int cuteafd_free_device_buffer(struct device_buffer *buffer) {
    if (strict_device_free && buffer->ptr && buffer->device_id != current_device) {
        ++device_free_mismatches;
        free_owner = buffer->device_id;
        free_current = current_device;
        return 1;
    }
    event('d');
    if (buffer->flags == 1) memory_used[buffer->device_id] -= buffer->bytes;
    if (extent_only && buffer->flags == 1) {
        if (munmap(buffer->ptr, buffer->bytes)) return 1;
        mapped_bytes -= buffer->bytes;
    } else { free(buffer->ptr); }
    memset(buffer, 0, sizeof(*buffer));
    return 0;
}

int cuteafd_alloc_managed_device_buffer(size_t bytes, struct device_buffer *out, uint32_t kind, const char *owner) {
    if (!valid_tag(kind, owner)) return 1;
    // Host-resident managed pages deliberately do not reduce physical free.
    out->ptr = malloc(bytes);
    if (!out->ptr) return 1;
    out->bytes = bytes;
    out->device_id = current_device;
    out->flags = 2;
    event('D');
    return 0;
}

static int capture_status;
void fixture_configure_capture(int status) { capture_status = status; }
struct graph_exec { int device; size_t bytes; };
int cuteafd_cuda_graph_end_capture(void *stream, void **out) {
    (void)stream;
    if (capture_status) return capture_status;
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

int fixture_fp8_w8a16_pack(const void *source, void *packed, void *scale,
                           int n, int k, int rule);
int cuteafd_fp8_w8a16_pack(const void *source, void *packed, void *scale,
                         int n, int k, int rule, void *stream) {
    (void)stream;
    event('P');
    if (pack_status || !immediate_copies || extent_only) return pack_status;
    return fixture_fp8_w8a16_pack(source, packed, scale, n, k, rule);
}

int fixture_packed_head_pack(const void *source, void *const *planes,
                             int64_t capacity, int n, int k, void *status);
int cuteafd_packed_head_pack(const void *source, void *const *planes,
                             int64_t capacity, int n, int k, void *status, void *stream) {
    (void)stream;
    event('P');
    if (pack_status || !immediate_copies || extent_only) return pack_status;
    return fixture_packed_head_pack(source, planes, capacity, n, k, status);
}

struct upload { void *dst; const void *src; size_t bytes; void *stream; };
struct upload_event { struct upload *copies; size_t count, fence; void *stream; };
static pthread_mutex_t upload_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t upload_ready = PTHREAD_COND_INITIALIZER;
static _Atomic size_t next_stream;
static void *blocked_stream;
static int block_first_stream;
void fixture_block_first_stream(int block) {
    pthread_mutex_lock(&upload_mutex);
    block_first_stream = block;
    pthread_cond_broadcast(&upload_ready);
    pthread_mutex_unlock(&upload_mutex);
}

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
int cuteafd_register_host_mapping(void *ptr, size_t bytes, uint32_t kind, const char *owner) {
    if (!ptr || !bytes || !valid_tag(kind, owner)) return 1;
    event('G'); return 0;
}
int cuteafd_unregister_host_mapping(void *ptr) {
    if (!ptr) return 1;
    event('g'); return 0;
}
int cuteafd_cuda_stream_create(void **out) {
    *out = (void *)(uintptr_t)(atomic_fetch_add(&next_stream, 1) + 1);
    pthread_mutex_lock(&upload_mutex);
    if (!blocked_stream) blocked_stream = *out;
    pthread_mutex_unlock(&upload_mutex);
    event('V'); return 0;
}
static void *upload_events[128];
static size_t upload_event_count;

int cuteafd_cuda_event_create_ordering(void **out) {
    pthread_mutex_lock(&upload_mutex);
    size_t slot = 0;
    while (slot < upload_event_count && upload_events[slot]) ++slot;
    if (slot == 128) { pthread_mutex_unlock(&upload_mutex); return 1; }
    *out = calloc(1, sizeof(struct upload_event));
    if (!*out) { pthread_mutex_unlock(&upload_mutex); return 1; }
    upload_events[slot] = *out;
    if (slot == upload_event_count) ++upload_event_count;
    pthread_mutex_unlock(&upload_mutex);
    event('C');
    return 0;
}
int cuteafd_cuda_event_create_blocking(void **out) {
    return cuteafd_cuda_event_create_ordering(out);
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
    if (!extent_only) memset(dst, 0, bytes);
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
    if (extent_only) return 0;
    if (queued_experts) return enqueue((struct queued_op){ .stream = stream,
        .dst = dst.ptr, .src = source, .bytes = bytes, .rows = 1 });
    if (immediate_copies) {
        if (dst.device_id != current_device) return 1;
        memcpy(dst.ptr, source, bytes);
    } else {
        pthread_mutex_lock(&upload_mutex);
        if (pending_count == 65536) { pthread_mutex_unlock(&upload_mutex); return 1; }
        pending_uploads[pending_count++] = (struct upload){dst.ptr, source, bytes, stream};
        pthread_mutex_unlock(&upload_mutex);
    }
    event('Q');
    return 0;
}
int cuteafd_copy_h2d_2d_async(struct device_buffer dst, size_t dst_pitch,
                            const void *source, size_t src_pitch,
                            size_t width, size_t rows, void *stream) {
    if (width > dst_pitch || width > src_pitch) return 1;
    if (!width || !rows) return 0;
    if ((rows - 1) > (SIZE_MAX - width) / dst_pitch
            || (rows - 1) > (SIZE_MAX - width) / src_pitch
            || (rows - 1) * dst_pitch + width > dst.bytes) return 1;
    /* Reuse the ordinary upload's immediate, queued and deferred semantics. */
    for (size_t row = 0; row < rows; ++row) {
        struct device_buffer target = dst;
        target.ptr = (char *)dst.ptr + row * dst_pitch;
        target.bytes = width;
        if (cuteafd_copy_h2d_async(target,
                (const char *)source + row * src_pitch, width, stream)) return 1;
    }
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
    pthread_mutex_lock(&upload_mutex);
    if (upload->count) { pthread_mutex_unlock(&upload_mutex); return 1; }
    upload->stream = stream;
    upload->copies = malloc(pending_count * sizeof(struct upload));
    if (pending_count && !upload->copies) { pthread_mutex_unlock(&upload_mutex); return 1; }
    size_t retained = 0;
    upload->count = 0;
    for (size_t i = 0; i < pending_count; ++i) {
        if (pending_uploads[i].stream == stream) upload->copies[upload->count++] = pending_uploads[i];
        else pending_uploads[retained++] = pending_uploads[i];
    }
    pending_count = retained;
    pthread_mutex_unlock(&upload_mutex);
    event('R');
    return 0;
}
int cuteafd_cuda_event_query(void *handle, int *ready) {
    struct upload_event *upload = handle;
    pthread_mutex_lock(&upload_mutex);
    *ready = queued_experts ? queued_done >= upload->fence : upload->count == 0;
    pthread_mutex_unlock(&upload_mutex);
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
    pthread_mutex_lock(&upload_mutex);
    struct upload_event *upload = handle;
    // Bound an injected stall even when a regression prevents the producer
    // from returning its credit. The test still fails its release-order check.
    struct timespec deadline;
    clock_gettime(CLOCK_REALTIME, &deadline);
    deadline.tv_sec += 5;
    while (block_first_stream && upload->stream == blocked_stream) {
        if (pthread_cond_timedwait(&upload_ready, &upload_mutex, &deadline)) {
            block_first_stream = 0;
            break;
        }
    }
    int status = ++drain_calls > drain_after ? drain_status : 0;
    if (!status) {
        finish_uploads(upload);
        event('L'); /* Successful event completion, after every retained copy. */
    }
    pthread_mutex_unlock(&upload_mutex);
    if (status) return status;
    return 0;
}

int cuteafd_cuda_event_destroy(void *handle) {
    pthread_mutex_lock(&upload_mutex);
    for (size_t i = 0; i < upload_event_count; ++i) {
        if (upload_events[i] == handle) { free(handle); upload_events[i] = NULL; }
    }
    pthread_mutex_unlock(&upload_mutex);
    event('E');
    return 0;
}

int cuteafd_cuda_stream_query(void *stream, int *ready) {
    (void)stream;
    if (trace_stream_queries) event('q');
    *ready = queued_experts ? queued_done == queued_count : immediate_copies;
    return 0;
}

int cuteafd_cuda_stream_synchronize(void *stream) {
    (void)stream;
    if (queued_experts) { advance_queue(queued_count); return 0; }
    event('S');
    pthread_mutex_lock(&upload_mutex);
    int status = ++drain_calls > drain_after ? drain_status : 0;
    if (status) { pthread_mutex_unlock(&upload_mutex); return status; }
    for (size_t i = 0; i < upload_event_count; ++i) {
        if (upload_events[i] && ((struct upload_event *)upload_events[i])->stream == stream)
            finish_uploads(upload_events[i]);
    }
    size_t retained = 0;
    for (size_t i = 0; i < pending_count; ++i) {
        struct upload *copy = &pending_uploads[i];
        if (copy->stream == stream) memcpy(copy->dst, copy->src, copy->bytes);
        else pending_uploads[retained++] = *copy;
    }
    pending_count = retained;
    pthread_mutex_unlock(&upload_mutex);
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
    if (extent_only) return 0;
    memcpy(dst.ptr, source, bytes);
    event('H');
    return 0;
}

int cuteafd_copy_d2d(struct device_buffer dst, struct device_buffer src, size_t bytes) {
    if (bytes > dst.bytes || bytes > src.bytes) return 1;
    memcpy(dst.ptr, src.ptr, bytes); return 0;
}
int cuteafd_cuda_zero_bytes(void *dst, size_t bytes) { if (!extent_only) memset(dst, 0, bytes); return 0; }
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
int cuteafd_pack_expert_window_async(const uint8_t *source, uint64_t source_bytes,
    uint8_t *destination, uint64_t destination_bytes, uint32_t intermediate,
    uint32_t projection, uint32_t first, uint32_t rows, void *stream) {
    int status = cuteafd_cuda_stream_synchronize(stream);
    if (status) return status;
    if (pack_status) return pack_status;
    uint32_t stored = cuteafd_mxfp4_stored_width(5120, intermediate);
    struct cuteafd_expert_pack_window window;
    if (!source || !destination || !cuteafd_expert_pack_window_geometry(5120,
        intermediate, stored, projection, first, rows, &window) ||
        source_bytes < (uint64_t)rows * window.k / 2 ||
        destination_bytes < (uint64_t)window.n * window.k / 2) return 1;
    for (uint64_t index = window.begin; index < window.begin + window.count; ++index) {
        uint32_t row, col;
        cuteafd_expert_pack_layout_coordinates(index, window.n, window.k,
            window.half, 0, window.compact, &row, &col);
        if (row < window.row_first || row >= window.row_end) continue;
        uint64_t at = ((uint64_t)(row - window.row_first) * (window.k / 8) + col) * 4;
        if (row - window.row_first < rows && col < window.k / 8)
            memcpy(destination + index * 4, source + at, 4);
        else memset(destination + index * 4, 0, 4);
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
static uint32_t exl3_words[16] = {2,5120,1152,6,1,6,2,1,1,1,1,3,4,0,0,4};
void fixture_configure_exl3(const uint32_t *words) { memcpy(exl3_words, words, sizeof(exl3_words)); }
int cuteafd_exl3_info(uint32_t *out, uint32_t count) {
    const uint32_t *words = exl3_words;
    if (count != 16) return 1;
    memcpy(out, words, sizeof(exl3_words));
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

/* Device SEND registration fails deliberately; the real ring must select and
 * register its existing pinned payload slots, not retry device allocation. */
int cuteafd_rdma_rc_endpoint_register_device_region(void *handle, void *ptr,
        size_t bytes, uint32_t *region) {
    (void)handle; (void)region;
    if (!ptr || bytes != 2 * 1024 * 1024) return 1;
    event('G');
    return 7;
}
int cuteafd_rdma_rc_endpoint_register_region(void *handle, void *ptr,
        size_t bytes, uint32_t *region) {
    (void)handle;
    if (!ptr || !bytes || !region) return 1;
    event('H');
    *region = 7;
    return 0;
}
struct fixture_endpoint_info {
    int rdma_enabled;
    void *handle;
    uint32_t port_num, qp_num, psn, lid, active_mtu;
    size_t send_frame_bytes, recv_frame_bytes, send_registered_span_bytes, recv_registered_span_bytes;
    uint32_t max_send_wr, max_recv_wr, max_sge;
    char gid_hex[33], device_name[128], status[128];
};
struct fixture_endpoint_view {
    void *host_ptr, *device_ptr;
    size_t bytes;
    int device_id;
    uint64_t host_flags;
};
struct fixture_endpoint {
    void *send, *recv;
    size_t send_bytes, recv_bytes;
    uint64_t flags;
    struct fixture_endpoint *next;
};
static struct fixture_endpoint *fixture_endpoints;
static int endpoint_view_failure, endpoint_destroy_failure;
void fixture_configure_endpoint_failures(int view, int destroy) {
    endpoint_view_failure = view;
    endpoint_destroy_failure = destroy;
}
int cuteafd_rdma_rc_endpoint_create_on_gid_with_buffer_flags(const char *name,
        uint32_t port, uint32_t gid, uint32_t psn, size_t send_frame, size_t recv_frame,
        size_t send_span, size_t recv_span, uint32_t send_wr, uint32_t recv_wr,
        uint32_t sge, uint64_t flags, struct fixture_endpoint_info *out) {
    (void)name; (void)gid;
    struct fixture_endpoint *endpoint = calloc(1, sizeof(*endpoint));
    if (!endpoint) return 1;
    endpoint->send = malloc(send_span);
    endpoint->recv = malloc(recv_span);
    if (!endpoint->send || !endpoint->recv) {
        free(endpoint->send); free(endpoint->recv); free(endpoint); return 1;
    }
    endpoint->send_bytes = send_span;
    endpoint->recv_bytes = recv_span;
    endpoint->flags = flags;
    endpoint->next = fixture_endpoints;
    fixture_endpoints = endpoint;
    memset(out, 0, sizeof(*out));
    out->rdma_enabled = 1; out->handle = endpoint; out->port_num = port;
    out->psn = psn; out->send_frame_bytes = send_frame; out->recv_frame_bytes = recv_frame;
    out->send_registered_span_bytes = send_span; out->recv_registered_span_bytes = recv_span;
    out->max_send_wr = send_wr; out->max_recv_wr = recv_wr; out->max_sge = sge;
    return 0;
}
int cuteafd_rdma_rc_endpoint_buffer_view(void *handle, int receive,
        struct fixture_endpoint_view *out) {
    if (endpoint_view_failure || (receive != 0 && receive != 1)) return 1;
    struct fixture_endpoint *endpoint = fixture_endpoints;
    while (endpoint && endpoint != handle) endpoint = endpoint->next;
    if (!endpoint) return 1;
    memset(out, 0, sizeof(*out));
    out->host_ptr = receive ? endpoint->recv : endpoint->send;
    out->bytes = receive ? endpoint->recv_bytes : endpoint->send_bytes;
    out->host_flags = endpoint->flags; out->device_id = current_device;
    if (endpoint->flags & 2) out->device_ptr = out->host_ptr;
    return 0;
}
int cuteafd_rdma_rc_endpoint_destroy(void *handle) {
    event('X');
    if (endpoint_destroy_failure) return 1;
    struct fixture_endpoint **link = &fixture_endpoints;
    while (*link && *link != handle) link = &(*link)->next;
    if (*link) {
        struct fixture_endpoint *endpoint = *link;
        *link = endpoint->next;
        free(endpoint->send); free(endpoint->recv); free(endpoint);
    }
    return 0;
}

/* CPU-only draft liveness probes; none dereference their launch pointers. */
int cuteafd_v41_dspark_cache_write_fp8(const uint16_t *src, const void *writes,
        uint8_t *ring, int rows, int slots, void *stream) {
    (void)src; (void)writes; (void)ring; (void)rows; (void)slots; (void)stream;
    return 0;
}
int cuteafd_cuda_graph_begin_capture(void *stream) { (void)stream; return 0; }
int cuteafd_fp8_w8a16_linear(const void *x, const void *w, const void *s, void *out,
        int f32, int rows, int k, int n, void *scratch, size_t bytes, void *stream) {
    (void)x; (void)w; (void)s; (void)out; (void)f32; (void)rows;
    (void)k; (void)n; (void)scratch; (void)bytes; (void)stream; return 98;
}
int cuteafd_fp8_linear(const void *x, const void *w, const void *s, void *out,
        int f32, int rows, int k, int n, int mode, void *scratch, size_t bytes, void *stream) {
    (void)mode;
    return cuteafd_fp8_w8a16_linear(x, w, s, out, f32, rows, k, n, scratch, bytes, stream);
}
int cuteafd_packed_head_linear(const void *x, void *const *planes, void *out,
        int f32, int rows, int k, int n, int coarse, void *scratch, size_t bytes, void *stream) {
    (void)planes; (void)coarse;
    return cuteafd_fp8_w8a16_linear(x, NULL, NULL, out, f32, rows, k, n, scratch, bytes, stream);
}

/* A fake media tower honouring the native create contract: its two backings
 * go through the tagged allocator; a failed create keeps the owner only when a
 * backing could not be freed. fixture_media_free_fails makes every free fail. */
struct media_backings { void *weights; uint64_t weight_bytes; void *workspace; uint64_t workspace_bytes; };
struct media_tags { uint32_t weight_kind; const char *weight_owner; uint32_t workspace_kind; const char *workspace_owner; };
struct fixture_media { void *weights, *workspace; uint64_t weight_bytes, workspace_bytes; };
static int media_free_fails, media_fail_after_weights;
void fixture_media_configure(int free_fails, int fail_after_weights) {
    media_free_fails = free_fails; media_fail_after_weights = fail_after_weights;
}
static uint32_t media_release(struct fixture_media *o) {
    uint32_t released = 0;
    if (o->weights && !media_free_fails) { free(o->weights); o->weights = NULL; released |= 1; }
    if (o->workspace && !media_free_fails) { free(o->workspace); o->workspace = NULL; released |= 2; }
    return released;
}
int cuteafd_fixture_media_create(uint64_t weight_bytes, uint64_t workspace_bytes,
        const struct media_tags *tags, void **out) {
    *out = NULL;
    struct fixture_media *o = calloc(1, sizeof *o);
    if (!o) return 2;
    if (!valid_tag(tags->weight_kind, tags->weight_owner) || !(o->weights = malloc(weight_bytes))) { free(o); return 2; }
    o->weight_bytes = weight_bytes;
    /* As native: the tag is checked and retained before the allocation that fails. */
    if (!valid_tag(tags->workspace_kind, tags->workspace_owner) || media_fail_after_weights
            || !(o->workspace = malloc(workspace_bytes))) {
        media_release(o);
        if (o->weights || o->workspace) *out = o; else free(o);
        return 2;
    }
    o->workspace_bytes = workspace_bytes;
    *out = o;
    return 0;
}
int cuteafd_fixture_media_get_backings(void *p, struct media_backings *out) {
    struct fixture_media *o = p;
    *out = (struct media_backings){ o->weights, o->weights ? o->weight_bytes : 0,
        o->workspace, o->workspace ? o->workspace_bytes : 0 };
    return 0;
}
int cuteafd_fixture_media_destroy_backings(void *p, uint32_t *released) {
    struct fixture_media *o = p;
    *released = media_release(o);
    if (!o->weights && !o->workspace) free(o);
    return media_free_fails ? 1 : 0;
}
