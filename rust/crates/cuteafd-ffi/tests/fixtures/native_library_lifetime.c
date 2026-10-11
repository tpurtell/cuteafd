// CPU-only fake native DSO. Its observable destructor proves actual dlclose;
// no CUDA, device, or transport implementation is linked into this fixture.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct host_buffer { void *ptr; size_t bytes; uint64_t flags; };
static char *event_path;
static int pack_status, drain_status, drain_calls, drain_after, current_device;
static size_t memory_total[8], memory_used[8], graph_bytes;

void fixture_configure_memory(int device, size_t total, size_t used, size_t graph) {
    memory_total[device] = total;
    memory_used[device] = used;
    graph_bytes = graph;
}

int cuteafd_cuda_get_device(int *device) { *device = current_device; return 0; }
int cuteafd_cuda_memory_info(size_t *free_bytes, size_t *total) {
    *total = memory_total[current_device];
    *free_bytes = *total - memory_used[current_device];
    return 0;
}

static void event(char value) {
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

void fixture_configure_pack(int pack, int drain) {
    pack_status = pack;
    drain_status = drain;
    drain_calls = 0;
    drain_after = 0;
}

void fixture_configure_drain_after(int after) { drain_after = after; }

int cuteafd_alloc_device_buffer(size_t bytes, struct device_buffer *out) {
    out->ptr = malloc(bytes);
    if (!out->ptr) return 1;
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

int cuteafd_cuda_event_synchronize(void *handle) {
    (void)handle;
    event('S');
    return ++drain_calls > drain_after ? drain_status : 0;
}

int cuteafd_cuda_event_destroy(void *handle) {
    (void)handle;
    event('E');
    return 0;
}

int cuteafd_cuda_stream_query(void *stream, int *ready) {
    (void)stream;
    *ready = 0;
    return 0;
}

int cuteafd_cuda_stream_synchronize(void *stream) {
    (void)stream;
    event('S');
    return ++drain_calls > drain_after ? drain_status : 0;
}

int cuteafd_cuda_set_device(int device) { current_device = device; return 0; }

int cuteafd_copy_h2d(struct device_buffer dst, const void *source, size_t bytes) {
    if (bytes > dst.bytes) return 1;
    memcpy(dst.ptr, source, bytes);
    event('H');
    return 0;
}

int cuteafd_copy_d2d_2d_async(struct device_buffer dst, size_t dst_pitch,
                            struct device_buffer src, size_t src_pitch,
                            size_t width, size_t rows, void *stream) {
    (void)stream;
    for (size_t r = 0; r < rows; ++r)
        memcpy((char *)dst.ptr + r * dst_pitch,
               (const char *)src.ptr + r * src_pitch, width);
    event('C');
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
