// CPU-only fake native DSO. Its observable destructor proves actual dlclose;
// no CUDA, device, or transport implementation is linked into this fixture.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
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

int64_t cuteafd_l2_cache_bytes(void) { return 1 << 20; }
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
int cuteafd_peer_diagnostic_initialize(void) { return 0; }

/* Ownership-only MLA programs: real Rust manifest/ABI validation still runs.
 * Numerical launches are deliberately unsupported by this CPU fixture. */
struct fixture_program_info {
    char name[64]; uint32_t pointers, scalars; uint8_t scalar_kinds[16];
};
static const char *mla_programs[] = {
    "fixture_mla_draft_producer_m64",
    "fixture_mla_draft_attention_full_m64",
    "fixture_mla_draft_attention_w4096_m64",
    "fixture_mla_draft_producer_fp8_m64",
    "fixture_mla_draft_attention_full_fp8_m64",
    "fixture_mla_draft_attention_w4096_fp8_m64",
    "fixture_mla_draft_output_m64"
};
/* Opt-in allocation-only target metadata; the MLA table stays unchanged unless
 * a GLM ownership fixture explicitly enables this extension before querying it. */
static int glm_target_programs;
void fixture_configure_glm_target_programs(void) { glm_target_programs = 1; }
/* Real peer construction probes this ABI; no peer work runs in the ownership fixture. */
int cuteafd_peer_abort_initialize(void) { return glm_target_programs ? 0 : 1; }
int cuteafd_peer_wait_abortable(const uint32_t *flag, uint32_t *state,
        const uint32_t *aborted, void *stream) {
    (void)flag; (void)state; (void)aborted; (void)stream;
    return 1;
}
int cuteafd_peer_abort_publish(uint32_t *aborted, void *stream) {
    (void)aborted; (void)stream;
    return 1;
}
static const char *glm_target_bases[] = {
    "glm_index_producer", "glm2_producer", "glm2_sparse_mla_decode", "glm2_o",
    "glm2_ffn_i1024", "glm2_ffn_i6144", "glm_index_topk_decode",
    "glm_index_producer", "glm2_producer", "glm2_sparse_mla_prefill", "glm2_o",
    "glm2_ffn_i1024", "glm2_ffn_i6144", "glm_index_topk_prefill",
    "glm_producer", "glm_sparse_mla_decode", "glm_o", "glm_ffn_i2048", "glm_ffn_i12288",
    "glm_producer", "glm_sparse_mla_prefill", "glm_o", "glm_ffn_i2048", "glm_ffn_i12288"
};
/* A separate, bounded ownership table keeps the existing MLA/GLM fixtures intact. */
static int exlr8_programs, exlr8_flash;
void fixture_configure_exlr8_programs(int flash) { exlr8_programs = 1; exlr8_flash = flash; }
static const char *exlr8_suffixes[] = { "shared_a16_serving", "a16_serving", "a16_grouped64",
    "a16_grouped", "a8_grouped", "shared_group", "group", "grouped_group" };
static int native_vocabulary_programs;
void fixture_configure_native_vocabulary_programs(void) { native_vocabulary_programs = 1; }
/* Allocation-only drafter constructors still admit the canonical BF16 ABI. */
static uint32_t draft_projection_count, draft_projection_shapes[16][2];
static int draft_projection_mla;
int fixture_configure_draft_projections(const uint32_t *shapes, uint32_t count, int include_mla) {
    if (!shapes || count == 0 || count > 16) return 1;
    memcpy(draft_projection_shapes, shapes, count * sizeof(draft_projection_shapes[0]));
    draft_projection_count = count;
    draft_projection_mla = include_mla;
    return 0;
}
uint32_t cuteafd_dsv4_program_count(void) {
    return draft_projection_count ? draft_projection_count + (draft_projection_mla ? 7 : 0)
        : (native_vocabulary_programs ? 2 : (exlr8_programs ? 8 : (glm_target_programs ? 31 : 7)));
}
int cuteafd_dsv4_program_info(uint32_t index, struct fixture_program_info *info) {
    if (index >= cuteafd_dsv4_program_count() || !info) return 1;
    memset(info, 0, sizeof(*info));
    if (index < draft_projection_count) {
        snprintf(info->name, sizeof(info->name), "draft_bf16_k%u_n%u",
            draft_projection_shapes[index][0], draft_projection_shapes[index][1]);
        info->pointers = 3;
        info->scalars = 1;
        info->scalar_kinds[0] = 'i';
        return 0;
    }
    if (draft_projection_count) index -= draft_projection_count;
    if (native_vocabulary_programs) {
        strcpy(info->name, index == 0 ? "exlr8_dense_fp8_row_embedding_k128_n256"
                                     : "exlr8_dense_fp8_row_k128_n256_s1");
        info->pointers = index == 0 ? 4 : 5;
        info->scalars = 2;
        memcpy(info->scalar_kinds, "ii", 2);
        return 0;
    }
    if (exlr8_programs) {
        snprintf(info->name, sizeof(info->name), "exlr8_%s%s", exlr8_flash ? "glmf_" : "", exlr8_suffixes[index]);
        info->pointers = index >= 5 ? 8 : (index >= 2 ? 15 : 13);
        const char *scalars = index >= 2 && index < 5 ? "iiii" : "ii";
        info->scalars = (uint32_t)strlen(scalars);
        memcpy(info->scalar_kinds, scalars, info->scalars);
        return 0;
    }
    if (index >= 7) {
        uint32_t target = index - 7;
        int prefill = (target >= 7 && target < 14) || target >= 19;
        snprintf(info->name, sizeof(info->name), "%s_m%d", glm_target_bases[target], prefill ? 4096 : 64);
        return 0;
    }
    strcpy(info->name, mla_programs[index]);
    const int kind = index == 6 ? 2 : (index % 3 == 0 ? 0 : 1);
    info->pointers = kind == 0 ? 13 : (kind == 1 ? 8 : 5);
    const char *scalars = kind == 0 ? "fi" : (kind == 1 ? "iilif" : "i");
    info->scalars = (uint32_t)strlen(scalars);
    memcpy(info->scalar_kinds, scalars, info->scalars);
    return 0;
}
int cuteafd_dsv4_program_load(uint32_t index) { return index < cuteafd_dsv4_program_count() ? 0 : 1; }
int cuteafd_dsv4_program_launch(uint32_t index, void **pointers,
        const uint64_t *scalars, void *stream) {
    (void)index; (void)pointers; (void)scalars; (void)stream;
    return 1;
}


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

/* Raw SM-copy exports carry no device IDs. Resolve their views against live
 * owners, never the caller's view metadata, under the same lock as the trace. */
struct live_allocation {
    void *base;
    size_t bytes;
    int device;
    struct live_allocation *next;
};
static struct live_allocation *live_allocations;
static pthread_mutex_t copy_lock = PTHREAD_MUTEX_INITIALIZER;
struct fixture_copy_record {
    uint32_t kind;
    int32_t device, destination_device, source_device;
    uint64_t destination, source, destination_backing, source_backing;
    uint64_t destination_offset, source_offset, bytes, width, rows;
    uint64_t destination_pitch, source_pitch, stream;
};
/* Kinds: initialize, SM linear, SM rows, DMA linear, DMA rows. */
static struct fixture_copy_record copy_records[65536];
static size_t copy_record_count;
static int record_copies;
void fixture_record_peer_copies(void) {
    pthread_mutex_lock(&copy_lock);
    record_copies = 1;
    copy_record_count = 0;
    pthread_mutex_unlock(&copy_lock);
}
size_t fixture_peer_copy_record_count(void) {
    pthread_mutex_lock(&copy_lock);
    size_t count = copy_record_count;
    pthread_mutex_unlock(&copy_lock);
    return count;
}
int fixture_peer_copy_record(size_t index, struct fixture_copy_record *out) {
    pthread_mutex_lock(&copy_lock);
    int status = !out || index >= copy_record_count;
    if (!status) *out = copy_records[index];
    pthread_mutex_unlock(&copy_lock);
    return status;
}
static int track_allocation(struct device_buffer buffer) {
    struct live_allocation *entry = malloc(sizeof(*entry));
    if (!entry) return 1;
    *entry = (struct live_allocation){buffer.ptr, buffer.bytes, buffer.device_id, NULL};
    pthread_mutex_lock(&copy_lock);
    entry->next = live_allocations;
    live_allocations = entry;
    pthread_mutex_unlock(&copy_lock);
    return 0;
}
static void untrack_allocation(void *base) {
    pthread_mutex_lock(&copy_lock);
    for (struct live_allocation **at = &live_allocations; *at; at = &(*at)->next) {
        if ((*at)->base == base) {
            struct live_allocation *entry = *at;
            *at = entry->next;
            free(entry);
            break;
        }
    }
    pthread_mutex_unlock(&copy_lock);
}
static struct live_allocation *find_allocation(const void *ptr, size_t bytes) {
    uintptr_t address = (uintptr_t)ptr;
    for (struct live_allocation *entry = live_allocations; entry; entry = entry->next) {
        uintptr_t base = (uintptr_t)entry->base;
        if (address >= base && address - base <= entry->bytes &&
                bytes <= entry->bytes - (address - base)) return entry;
    }
    return NULL;
}
static int record_copy(uint32_t kind, void *destination, const void *source,
        size_t width, size_t rows, size_t destination_pitch, size_t source_pitch,
        void *stream, int strict) {
    pthread_mutex_lock(&copy_lock);
    if (!strict && !record_copies) { pthread_mutex_unlock(&copy_lock); return 0; }
    if (!width || !rows || destination_pitch < width || source_pitch < width ||
            rows > SIZE_MAX / width || (rows > 1 &&
            (destination_pitch > (SIZE_MAX - width) / (rows - 1) ||
             source_pitch > (SIZE_MAX - width) / (rows - 1)))) {
        pthread_mutex_unlock(&copy_lock);
        return 1;
    }
    size_t dst_bytes = (rows - 1) * destination_pitch + width;
    size_t src_bytes = (rows - 1) * source_pitch + width;
    struct live_allocation *dst = find_allocation(destination, dst_bytes);
    struct live_allocation *src = find_allocation(source, src_bytes);
    int status = !dst || !src || dst->device != current_device ||
        (record_copies && copy_record_count == 65536);
    if (!status && record_copies) {
        copy_records[copy_record_count++] = (struct fixture_copy_record){
            .kind = kind, .device = current_device,
            .destination_device = dst->device, .source_device = src->device,
            .destination = (uintptr_t)destination, .source = (uintptr_t)source,
            .destination_backing = (uintptr_t)dst->base, .source_backing = (uintptr_t)src->base,
            .destination_offset = (uintptr_t)destination - (uintptr_t)dst->base,
            .source_offset = (uintptr_t)source - (uintptr_t)src->base,
            .bytes = width * rows, .width = width, .rows = rows,
            .destination_pitch = destination_pitch, .source_pitch = source_pitch,
            .stream = (uintptr_t)stream
        };
    }
    pthread_mutex_unlock(&copy_lock);
    return status;
}

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
    if (track_allocation(*out)) { free(out->ptr); memset(out, 0, sizeof(*out)); return 1; }
    memory_used[current_device] += bytes;
    event('D');
    return 0;
}

int cuteafd_free_device_buffer(struct device_buffer *buffer) {
    event('d');
    if (buffer->flags == 1) memory_used[buffer->device_id] -= buffer->bytes;
    untrack_allocation(buffer->ptr);
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
    if (track_allocation(*out)) { free(out->ptr); memset(out, 0, sizeof(*out)); return 1; }
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

int cuteafd_peer_copy_initialize(void) {
    pthread_mutex_lock(&copy_lock);
    int status = record_copies && copy_record_count == 65536;
    if (!status && record_copies) {
        copy_records[copy_record_count++] = (struct fixture_copy_record){
            .device = current_device, .destination_device = -1, .source_device = -1
        };
    }
    pthread_mutex_unlock(&copy_lock);
    return status;
}
static int peer_extent(uint64_t width, uint64_t rows, uint64_t pitch, size_t *bytes) {
    if (!width || width > (1ull << 40) || !rows || rows > 4096 || pitch < width ||
            (rows > 1 && pitch > ((1ull << 40) - width) / (rows - 1))) return 1;
    *bytes = (size_t)((rows - 1) * pitch + width);
    return 0;
}
static int peer_copy(uint32_t kind, void *destination, const void *source,
        uint64_t width, uint64_t rows, uint64_t destination_pitch,
        uint64_t source_pitch, void *stream) {
    size_t dst_bytes, src_bytes;
    uintptr_t dst = (uintptr_t)destination, src = (uintptr_t)source;
    if (!stream || !dst || !src ||
            peer_extent(width, rows, destination_pitch, &dst_bytes) ||
            peer_extent(width, rows, source_pitch, &src_bytes) ||
            dst > UINTPTR_MAX - dst_bytes || src > UINTPTR_MAX - src_bytes ||
            !(dst + dst_bytes <= src || src + src_bytes <= dst)) return 1;
    if (queued_experts && queued_count == 65536) return 1;
    if (!queued_experts && !immediate_copies && rows > 65536 - pending_count) return 1;
    if (record_copy(kind, destination, source, width, rows,
            destination_pitch, source_pitch, stream, 1)) return 1;
    if (queued_experts) return enqueue((struct queued_op){ .stream = stream,
        .dst = destination, .src = source, .bytes = width, .rows = rows,
        .dst_pitch = destination_pitch, .src_pitch = source_pitch });
    for (size_t row = 0; row < rows; ++row) {
        void *to = (char *)destination + row * destination_pitch;
        const void *from = (const char *)source + row * source_pitch;
        if (immediate_copies) memcpy(to, from, width);
        else pending_uploads[pending_count++] = (struct upload){to, from, width};
    }
    return 0;
}
int cuteafd_peer_copy_async(void *destination, const void *source, uint64_t bytes, void *stream) {
    return peer_copy(1, destination, source, bytes, 1, bytes, bytes, stream);
}
int cuteafd_peer_copy_rows_async(void *destination, const void *source,
        uint64_t width, uint64_t rows, uint64_t destination_pitch,
        uint64_t source_pitch, void *stream) {
    return peer_copy(2, destination, source, width, rows, destination_pitch, source_pitch, stream);
}

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
int cuteafd_cuda_event_create_blocking(void **out) {
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
    if (record_copy(3, dst.ptr, src.ptr, bytes, 1, bytes, bytes, stream, 0)) return 1;
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
    if (record_copy(4, dst.ptr, src.ptr, width, rows, dst_pitch, src_pitch, stream, 0)) return 1;
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
static uint32_t exl3_capacity = 1;
void fixture_configure_exl3_capacity(uint32_t capacity) { exl3_capacity = capacity; }
int cuteafd_exl3_info(uint32_t *out, uint32_t count) {
    const uint32_t words[16] = {2,5120,1152,6,exl3_capacity,6,2,1,1,1,1,3,4,0,0,4};
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
