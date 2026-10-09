#include "cuteafd_native.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <string>
#include <vector>

#if CUTEAFD_NATIVE_ENABLE_CUDA
#include <cuda.h>
#include <cuda_runtime_api.h>
#endif

#if CUTEAFD_NATIVE_ENABLE_RDMA
#include <infiniband/verbs.h>
#include <poll.h>
#include <unistd.h>
#endif

#if CUTEAFD_NATIVE_ENABLE_NCCL
#include <nccl.h>
#endif

namespace {

thread_local std::string g_last_error;
constexpr int kRdmaRcEndpointActiveEventPollTimeoutMs = 30000;

cuteafd_status_t ok();
#if CUTEAFD_NATIVE_ENABLE_CUDA
cuteafd_status_t fail_cuda(cuteafd_status_t status, const char* action, cudaError_t err);
#endif
#if CUTEAFD_NATIVE_ENABLE_NCCL
cuteafd_status_t fail_nccl(const char* action, ncclResult_t err);
#endif

cuteafd_status_t fail(cuteafd_status_t status, const std::string& message) {
  g_last_error = message;
  return status;
}

cuteafd_status_t ok() {
  g_last_error.clear();
  return CUTEAFD_STATUS_OK;
}

cuteafd_status_t write_c_string(const std::string& value, char* out, size_t out_len) {
  if (out == nullptr) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "output buffer is null");
  }
  if (out_len == 0) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "output buffer length is zero");
  }
  if (value.size() + 1 > out_len) {
    const size_t copy_len = out_len - 1;
    if (copy_len > 0) {
      std::memcpy(out, value.data(), copy_len);
    }
    out[out_len - 1] = '\0';
    return fail(CUTEAFD_STATUS_BUFFER_TOO_SMALL, "output buffer is too small");
  }
  std::memcpy(out, value.c_str(), value.size() + 1);
  return ok();
}

void set_fixed_string(char* dst, size_t dst_len, const char* value) {
  if (dst_len == 0) {
    return;
  }
  const size_t copy_len = std::min(dst_len - 1, std::strlen(value));
  std::memcpy(dst, value, copy_len);
  dst[copy_len] = '\0';
}

std::string build_version() {
  std::string version = "cuteafd_native 0.1.0";
  version += CUTEAFD_NATIVE_ENABLE_CUDA ? " cuda=on" : " cuda=off";
  version += CUTEAFD_NATIVE_ENABLE_RDMA ? " rdma=on" : " rdma=off";
  version += CUTEAFD_NATIVE_ENABLE_NCCL ? " nccl=on" : " nccl=off";
  return version;
}

#if CUTEAFD_NATIVE_ENABLE_NCCL
struct NcclCommHandle {
  ncclComm_t comm = nullptr;
  int world_size = 0;
  int rank = -1;
};

cuteafd_status_t fail_nccl(const char* action, ncclResult_t err) {
  return fail(CUTEAFD_STATUS_INTERNAL_ERROR,
              std::string(action) + ": " + ncclGetErrorString(err));
}
#endif

uint16_t read_le16(const unsigned char* bytes) {
  return static_cast<uint16_t>(bytes[0]) |
         static_cast<uint16_t>(static_cast<uint16_t>(bytes[1]) << 8);
}

uint32_t read_le32(const unsigned char* bytes) {
  return static_cast<uint32_t>(bytes[0]) | (static_cast<uint32_t>(bytes[1]) << 8) |
         (static_cast<uint32_t>(bytes[2]) << 16) |
         (static_cast<uint32_t>(bytes[3]) << 24);
}

uint64_t read_le64(const unsigned char* bytes) {
  uint64_t value = 0;
  for (size_t idx = 0; idx < 8; ++idx) {
    value |= static_cast<uint64_t>(bytes[idx]) << (idx * 8);
  }
  return value;
}

cuteafd_status_t validate_protocol_v2_frame(const void* frame, size_t frame_bytes,
                                          uint16_t expected_kind,
                                          const char* label) {
  constexpr unsigned char kMagic[8] = {'D', 'S', '4', '1', 'R', 'T', 'E', '3'};
  constexpr uint16_t kVersion = 3;
  constexpr uint32_t kHotHeaderBytes = 96;
  constexpr uint32_t kDebugHeaderBytes = 128;
  constexpr uint32_t kDebugChecksumFlag = 1u;
  if (frame == nullptr) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT,
                std::string("ProtocolV2 ") + label + " frame pointer is null");
  }
  if (frame_bytes < kHotHeaderBytes) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT,
                std::string("ProtocolV2 ") + label + " frame is shorter than header");
  }
  const unsigned char* bytes = static_cast<const unsigned char*>(frame);
  if (std::memcmp(bytes, kMagic, sizeof(kMagic)) != 0) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT,
                std::string("ProtocolV2 ") + label + " frame has invalid magic");
  }
  const uint16_t version = read_le16(bytes + 8);
  if (version != kVersion) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT,
                std::string("ProtocolV2 ") + label + " frame has unsupported version");
  }
  const uint16_t kind = read_le16(bytes + 10);
  if (kind != expected_kind) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT,
                std::string("ProtocolV2 ") + label + " frame has unexpected kind");
  }
  const uint32_t header_bytes = read_le32(bytes + 12);
  const size_t flags_offset = expected_kind == 1 ? 84 : 68;
  const uint32_t flags = read_le32(bytes + flags_offset);
  const uint32_t expected_header_bytes =
      (flags & kDebugChecksumFlag) != 0 ? kDebugHeaderBytes : kHotHeaderBytes;
  if (header_bytes != expected_header_bytes) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT,
                std::string("ProtocolV2 ") + label + " frame has unexpected header length");
  }
  const size_t wire_bytes_offset = expected_kind == 1 ? 76 : 60;
  const uint64_t wire_bytes = read_le64(bytes + wire_bytes_offset);
  if (wire_bytes != frame_bytes) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT,
                std::string("ProtocolV2 ") + label + " frame wire bytes mismatch");
  }
  return ok();
}

cuteafd_status_t compute_host_buffer_plan(const void* ptr, size_t bytes, size_t alignment,
                                        cuteafd_rdma_host_buffer_plan_t* out) {
  if (ptr == nullptr) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "RDMA host buffer pointer is null");
  }
  if (bytes == 0) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "RDMA host buffer byte size is zero");
  }
  if (alignment == 0) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "RDMA host buffer alignment is zero");
  }
  if (out == nullptr) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "RDMA host buffer plan output pointer is null");
  }

  const uintptr_t original_addr = reinterpret_cast<uintptr_t>(ptr);
  const uintptr_t registered_addr = original_addr - (original_addr % alignment);
  const size_t prefix_bytes = static_cast<size_t>(original_addr - registered_addr);
  if (bytes > std::numeric_limits<size_t>::max() - prefix_bytes) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "RDMA host buffer span overflows size_t");
  }
  size_t registered_span_bytes = prefix_bytes + bytes;
  const size_t remainder = registered_span_bytes % alignment;
  if (remainder != 0) {
    const size_t padding = alignment - remainder;
    if (registered_span_bytes > std::numeric_limits<size_t>::max() - padding) {
      return fail(CUTEAFD_STATUS_INVALID_ARGUMENT,
                  "RDMA host buffer aligned span overflows size_t");
    }
    registered_span_bytes += padding;
  }

  std::memset(out, 0, sizeof(*out));
  out->original_addr = original_addr;
  out->original_bytes = bytes;
  out->alignment = alignment;
  out->registered_addr = registered_addr;
  out->prefix_bytes = prefix_bytes;
  out->registered_span_bytes = registered_span_bytes;
  out->span_aligned = registered_span_bytes % alignment == 0 ? 1 : 0;
  out->rdma_enabled = CUTEAFD_NATIVE_ENABLE_RDMA ? 1 : 0;
  return ok();
}

#if CUTEAFD_NATIVE_ENABLE_RDMA
void set_first_rdma_device_info(ibv_device* device, cuteafd_rdma_device_info_t* out) {
  const char* device_name = ibv_get_device_name(device);
  set_fixed_string(out->first_device_name, sizeof(out->first_device_name),
                   device_name != nullptr ? device_name : "unknown-rdma-device");
  set_fixed_string(out->first_device_transport, sizeof(out->first_device_transport),
                   "libibverbs");
  out->first_device_guid = static_cast<uint64_t>(ibv_get_device_guid(device));

  ibv_context* context = ibv_open_device(device);
  if (context != nullptr) {
    out->first_device_openable = 1;
    set_fixed_string(out->status, sizeof(out->status), "first RDMA device opened");
    ibv_close_device(context);
  } else {
    out->first_device_openable = 0;
    set_fixed_string(out->status, sizeof(out->status), "ibv_open_device failed");
  }
}

int hex_nibble(char value) {
  if (value >= '0' && value <= '9') {
    return value - '0';
  }
  if (value >= 'a' && value <= 'f') {
    return value - 'a' + 10;
  }
  if (value >= 'A' && value <= 'F') {
    return value - 'A' + 10;
  }
  return -1;
}

cuteafd_status_t gid_from_hex(const char* value, ibv_gid* out) {
  if (value == nullptr || out == nullptr) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "RDMA GID hex pointer is null");
  }
  if (std::strlen(value) != 32) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "RDMA GID hex must contain 32 hex characters");
  }
  std::memset(out, 0, sizeof(*out));
  for (size_t idx = 0; idx < 16; ++idx) {
    const int hi = hex_nibble(value[idx * 2]);
    const int lo = hex_nibble(value[idx * 2 + 1]);
    if (hi < 0 || lo < 0) {
      return fail(CUTEAFD_STATUS_INVALID_ARGUMENT,
                  "RDMA GID hex contains a non-hex character");
    }
    out->raw[idx] = static_cast<uint8_t>((hi << 4) | lo);
  }
  return ok();
}

void gid_to_hex(const ibv_gid& gid, char* out, size_t out_len) {
  static constexpr char kHex[] = "0123456789abcdef";
  if (out == nullptr || out_len == 0) {
    return;
  }
  if (out_len < 33) {
    out[0] = '\0';
    return;
  }
  for (size_t idx = 0; idx < 16; ++idx) {
    out[idx * 2] = kHex[(gid.raw[idx] >> 4) & 0xf];
    out[idx * 2 + 1] = kHex[gid.raw[idx] & 0xf];
  }
  out[32] = '\0';
}

bool gid_is_zero(const ibv_gid& gid) {
  for (uint8_t value : gid.raw) {
    if (value != 0) {
      return false;
    }
  }
  return true;
}

bool gid_is_ipv4_mapped(const ibv_gid& gid) {
  for (int idx = 0; idx < 10; ++idx) {
    if (gid.raw[idx] != 0) {
      return false;
    }
  }
  return gid.raw[10] == 0xff && gid.raw[11] == 0xff;
}

cuteafd_status_t select_rc_gid(ibv_context* context, const ibv_port_attr& port_attr,
                             uint32_t port_num, ibv_gid* out_gid,
                             uint32_t* out_gid_index) {
  if (context == nullptr || out_gid == nullptr || out_gid_index == nullptr) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "RDMA RC GID selection argument is null");
  }
  if (port_attr.link_layer != IBV_LINK_LAYER_ETHERNET) {
    if (ibv_query_gid(context, static_cast<uint8_t>(port_num), 0, out_gid) != 0) {
      return fail(CUTEAFD_STATUS_RDMA_UNAVAILABLE, "ibv_query_gid failed for RC endpoint");
    }
    *out_gid_index = 0;
    return ok();
  }

  bool have_fallback = false;
  ibv_gid fallback_gid = {};
  uint32_t fallback_index = 0;
  const uint32_t gid_count = std::max<uint32_t>(1, port_attr.gid_tbl_len);
  for (uint32_t index = 0; index < gid_count; ++index) {
    ibv_gid_entry entry = {};
    if (ibv_query_gid_ex(context, port_num, index, &entry, 0) != 0) {
      continue;
    }
    if (gid_is_zero(entry.gid)) {
      continue;
    }
    const bool ipv4_mapped = gid_is_ipv4_mapped(entry.gid);
    if (entry.gid_type == IBV_GID_TYPE_ROCE_V2 && ipv4_mapped) {
      *out_gid = entry.gid;
      *out_gid_index = index;
      return ok();
    }
    if (!have_fallback ||
        (entry.gid_type == IBV_GID_TYPE_ROCE_V2 &&
         (ipv4_mapped || !gid_is_ipv4_mapped(fallback_gid))) ||
        (ipv4_mapped && !gid_is_ipv4_mapped(fallback_gid))) {
      fallback_gid = entry.gid;
      fallback_index = index;
      have_fallback = true;
    }
  }
  if (!have_fallback) {
    return fail(CUTEAFD_STATUS_RDMA_UNAVAILABLE, "no usable RoCE GID found for RC endpoint");
  }
  *out_gid = fallback_gid;
  *out_gid_index = fallback_index;
  return ok();
}

cuteafd_status_t modify_rc_qp_to_init(ibv_qp* qp, uint32_t port_num) {
  ibv_qp_attr attr = {};
  attr.qp_state = IBV_QPS_INIT;
  attr.pkey_index = 0;
  attr.port_num = static_cast<uint8_t>(port_num);
  attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ |
                         IBV_ACCESS_REMOTE_WRITE;
  const int flags =
      IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;
  if (ibv_modify_qp(qp, &attr, flags) != 0) {
    return fail(CUTEAFD_STATUS_RDMA_UNAVAILABLE, "ibv_modify_qp INIT failed");
  }
  return ok();
}

// `flow_label` 0 lets the kernel derive the RoCE v2 flow label (and with it
// the UDP source port) from both QP numbers, so it changes with every new
// pair of QPs; a nonzero label fixes it.
cuteafd_status_t modify_rc_qp_to_rtr(ibv_context* context, ibv_qp* qp,
                                   const ibv_port_attr& port_attr, uint32_t port_num,
                                   uint32_t remote_qp_num, uint32_t remote_psn,
                                   uint32_t remote_lid, const ibv_gid* remote_gid,
                                   uint32_t local_gid_index, uint32_t flow_label = 0) {
  ibv_qp_attr attr = {};
  attr.qp_state = IBV_QPS_RTR;
  attr.path_mtu = port_attr.active_mtu;
  attr.dest_qp_num = remote_qp_num;
  attr.rq_psn = remote_psn;
  attr.max_dest_rd_atomic = 1;
  attr.min_rnr_timer = 12;
  attr.ah_attr.dlid = remote_lid != 0 ? static_cast<uint16_t>(remote_lid) : port_attr.lid;
  attr.ah_attr.sl = 0;
  attr.ah_attr.src_path_bits = 0;
  attr.ah_attr.port_num = static_cast<uint8_t>(port_num);
  if (port_attr.link_layer == IBV_LINK_LAYER_ETHERNET) {
    ibv_gid gid = {};
    if (remote_gid != nullptr) {
      gid = *remote_gid;
    } else {
      uint32_t selected_gid_index = 0;
      const cuteafd_status_t status =
          select_rc_gid(context, port_attr, port_num, &gid, &selected_gid_index);
      if (status != CUTEAFD_STATUS_OK) {
        return status;
      }
    }
    attr.ah_attr.is_global = 1;
    attr.ah_attr.grh.dgid = gid;
    attr.ah_attr.grh.sgid_index = local_gid_index;
    attr.ah_attr.grh.hop_limit = 1;
    attr.ah_attr.grh.flow_label = flow_label;
  }
  const int flags = IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
                    IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER;
  if (ibv_modify_qp(qp, &attr, flags) != 0) {
    return fail(CUTEAFD_STATUS_RDMA_UNAVAILABLE, "ibv_modify_qp RTR failed");
  }
  return ok();
}

cuteafd_status_t modify_rc_qp_to_rts(ibv_qp* qp, uint32_t local_psn) {
  ibv_qp_attr attr = {};
  attr.qp_state = IBV_QPS_RTS;
  attr.timeout = 14;
  attr.retry_cnt = 7;
  attr.rnr_retry = 7;
  attr.sq_psn = local_psn;
  attr.max_rd_atomic = 1;
  const int flags = IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                    IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC;
  if (ibv_modify_qp(qp, &attr, flags) != 0) {
    return fail(CUTEAFD_STATUS_RDMA_UNAVAILABLE, "ibv_modify_qp RTS failed");
  }
  return ok();
}

struct CuteafdRdmaRcEndpointHandle {
  ibv_context* context = nullptr;
  ibv_pd* pd = nullptr;
  ibv_comp_channel* send_channel = nullptr;
  ibv_comp_channel* recv_channel = nullptr;
  ibv_cq* send_cq = nullptr;
  ibv_cq* recv_cq = nullptr;
  ibv_qp* qp = nullptr;
  ibv_mr* send_mr = nullptr;
  ibv_mr* recv_mr = nullptr;
  unsigned char* send_buffer = nullptr;
  unsigned char* recv_buffer = nullptr;
  ibv_port_attr port_attr = {};
  uint32_t port_num = 0;
  uint32_t psn = 0;
  uint32_t lid = 0;
  uint32_t gid_index = 0;
  size_t send_frame_bytes = 0;
  size_t recv_frame_bytes = 0;
  size_t send_registered_span_bytes = 0;
  size_t recv_registered_span_bytes = 0;
  uint64_t host_buffer_flags = CUTEAFD_HOST_BUFFER_FLAG_NONE;
  // GPU landing: receives scatter the first `landing_header_bytes` into the
  // host slot and the rest into this dma-buf registered device range.
  ibv_mr* landing_mr = nullptr;
  unsigned char* landing_ptr = nullptr;
  size_t landing_bytes = 0;
  size_t landing_header_bytes = 0;
  // Host ranges sends may gather from (`cuteafd_rdma_rc_endpoint_register_region`).
  std::vector<ibv_mr*> regions;
  // Device range a peer RDMA-writes into (`cuteafd_rdma_rc_endpoint_expose_device`).
  ibv_mr* exposed_mr = nullptr;
  // Send-buffer prefix a peer RDMA-reads (`cuteafd_rdma_rc_endpoint_expose_send_read`).
  ibv_mr* read_mr = nullptr;
  // Registered host words the completion flags of written responses are sent
  // from (a ring, so a value is never rewritten while its write may be queued).
  ibv_mr* flag_source_mr = nullptr;
  uint64_t* flag_source = nullptr;
  uint32_t flag_cursor = 0;
  uint32_t pending_send_completions = 0;
  uint32_t pending_recv_completions = 0;
  std::chrono::steady_clock::time_point busy_poll_until = {};
};

constexpr auto kRdmaRcEndpointBusyPollBudget = std::chrono::milliseconds(1);
// Four execution lanes can leave one QP unused for roughly a second during c1
// decode. Keep that active lane rotation polling; callers use a short explicit
// timeout when they are truly idle.
constexpr auto kRdmaRcEndpointRecentActivityBusyPollWindow = std::chrono::seconds(5);

void drain_rdma_rc_cq_events(ibv_comp_channel* channel) {
  if (channel == nullptr) {
    return;
  }
  pollfd fd = {};
  fd.fd = channel->fd;
  fd.events = POLLIN;
  while (poll(&fd, 1, 0) > 0) {
    if ((fd.revents & POLLIN) == 0) {
      return;
    }
    ibv_cq* event_cq = nullptr;
    void* event_context = nullptr;
    if (ibv_get_cq_event(channel, &event_cq, &event_context) != 0) {
      return;
    }
    ibv_ack_cq_events(event_cq, 1);
    fd.revents = 0;
  }
}

void destroy_rdma_rc_endpoint(CuteafdRdmaRcEndpointHandle* endpoint) {
  if (endpoint == nullptr) {
    return;
  }
  if (endpoint->qp != nullptr) {
    const int rc = ibv_destroy_qp(endpoint->qp);
    if (rc != 0) {
      std::fprintf(stderr, "warning: ibv_destroy_qp qp=%p rc=%d errno=%d\n",
                   static_cast<void*>(endpoint->qp), rc, errno);
    }
  }
  const auto deregister = [](ibv_mr* mr) {
    const int rc = ibv_dereg_mr(mr);
    if (rc != 0) {
      std::fprintf(stderr, "warning: ibv_dereg_mr mr=%p rc=%d errno=%d\n",
                   static_cast<void*>(mr), rc, errno);
    }
  };
  if (endpoint->recv_mr != nullptr) {
    deregister(endpoint->recv_mr);
  }
  if (endpoint->landing_mr != nullptr) {
    deregister(endpoint->landing_mr);
  }
  if (endpoint->exposed_mr != nullptr) {
    deregister(endpoint->exposed_mr);
  }
  if (endpoint->read_mr != nullptr) {
    deregister(endpoint->read_mr);
  }
  if (endpoint->flag_source_mr != nullptr) {
    deregister(endpoint->flag_source_mr);
  }
  std::free(endpoint->flag_source);
  for (ibv_mr* region : endpoint->regions) {
    deregister(region);
  }
  endpoint->regions.clear();
  if (endpoint->send_mr != nullptr) {
    deregister(endpoint->send_mr);
  }
  drain_rdma_rc_cq_events(endpoint->recv_channel);
  drain_rdma_rc_cq_events(endpoint->send_channel);
  if (endpoint->recv_cq != nullptr) {
    ibv_destroy_cq(endpoint->recv_cq);
  }
  if (endpoint->send_cq != nullptr) {
    ibv_destroy_cq(endpoint->send_cq);
  }
  if (endpoint->recv_channel != nullptr) {
    ibv_destroy_comp_channel(endpoint->recv_channel);
  }
  if (endpoint->send_channel != nullptr) {
    ibv_destroy_comp_channel(endpoint->send_channel);
  }
  if (endpoint->pd != nullptr) {
    ibv_dealloc_pd(endpoint->pd);
  }
  if (endpoint->context != nullptr) {
    ibv_close_device(endpoint->context);
  }
#if CUTEAFD_NATIVE_ENABLE_CUDA
  if ((endpoint->host_buffer_flags & CUTEAFD_HOST_BUFFER_FLAG_PINNED) != 0) {
    if (endpoint->send_buffer != nullptr) {
      cudaFreeHost(endpoint->send_buffer);
    }
    if (endpoint->recv_buffer != nullptr) {
      cudaFreeHost(endpoint->recv_buffer);
    }
  } else {
    std::free(endpoint->send_buffer);
    std::free(endpoint->recv_buffer);
  }
#else
  std::free(endpoint->send_buffer);
  std::free(endpoint->recv_buffer);
#endif
  delete endpoint;
}
#endif

#if CUTEAFD_NATIVE_ENABLE_CUDA
cuteafd_status_t fail_cuda(cuteafd_status_t status, const char* action, cudaError_t err) {
  return fail(status, std::string(action) + ": " + cudaGetErrorString(err));
}

void set_version_string(char* dst, size_t dst_len, int version) {
  char buffer[64] = {};
  std::snprintf(buffer, sizeof(buffer), "%d.%d", version / 1000, (version % 1000) / 10);
  set_fixed_string(dst, dst_len, buffer);
}

cuteafd_status_t fill_cuda_graph_capture_info(cudaGraph_t graph, cudaGraphExec_t graph_exec,
                                            cuteafd_cuda_graph_capture_info_t* out) {
  size_t node_count = 0;
  cudaError_t err = cudaGraphGetNodes(graph, nullptr, &node_count);
  if (err != cudaSuccess) {
    return fail_cuda(CUTEAFD_STATUS_INTERNAL_ERROR, "cudaGraphGetNodes count failed", err);
  }

  std::vector<cudaGraphNode_t> nodes(node_count);
  if (node_count > 0) {
    size_t copied_nodes = node_count;
    err = cudaGraphGetNodes(graph, nodes.data(), &copied_nodes);
    if (err != cudaSuccess) {
      return fail_cuda(CUTEAFD_STATUS_INTERNAL_ERROR, "cudaGraphGetNodes failed", err);
    }
    nodes.resize(copied_nodes);
    node_count = copied_nodes;
  }

  size_t kernel_node_count = 0;
  size_t memcpy_node_count = 0;
  size_t memset_node_count = 0;
  for (cudaGraphNode_t node : nodes) {
    cudaGraphNodeType type;
    err = cudaGraphNodeGetType(node, &type);
    if (err != cudaSuccess) {
      return fail_cuda(CUTEAFD_STATUS_INTERNAL_ERROR, "cudaGraphNodeGetType failed", err);
    }
    switch (type) {
      case cudaGraphNodeTypeKernel:
        ++kernel_node_count;
        break;
      case cudaGraphNodeTypeMemcpy:
        ++memcpy_node_count;
        break;
      case cudaGraphNodeTypeMemset:
        ++memset_node_count;
        break;
      default:
        break;
    }
  }

  out->graph = reinterpret_cast<void*>(graph);
  out->graph_exec = reinterpret_cast<void*>(graph_exec);
  out->node_count = node_count;
  out->kernel_node_count = kernel_node_count;
  out->memcpy_node_count = memcpy_node_count;
  out->memset_node_count = memset_node_count;
  return ok();
}
#endif

}  // namespace

extern "C" void cuteafd_set_last_error_message(const char* message) {
  g_last_error = message != nullptr ? message : "";
}

extern "C" cuteafd_status_t cuteafd_native_version(char* out, size_t out_len) {
  return write_c_string(build_version(), out, out_len);
}

extern "C" cuteafd_status_t cuteafd_cuda_device_info(int device_id, cuteafd_cuda_device_info_t* out) {
  if (out == nullptr) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "device info output pointer is null");
  }
  if (device_id < 0) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "device id must be non-negative");
  }

  std::memset(out, 0, sizeof(*out));
  out->device_id = device_id;

#if CUTEAFD_NATIVE_ENABLE_CUDA
  cudaDeviceProp props = {};
  cudaError_t err = cudaGetDeviceProperties(&props, device_id);
  if (err != cudaSuccess) {
    return fail_cuda(CUTEAFD_STATUS_CUDA_UNAVAILABLE, "cudaGetDeviceProperties failed", err);
  }
  int driver_version = 0;
  int runtime_version = 0;
  cudaDriverGetVersion(&driver_version);
  cudaRuntimeGetVersion(&runtime_version);
  out->cuda_available = 1;
  out->compute_capability_major = props.major;
  out->compute_capability_minor = props.minor;
  out->integrated = props.integrated;
  out->can_map_host_memory = props.canMapHostMemory;
  out->unified_addressing = props.unifiedAddressing;
  out->total_memory_bytes = props.totalGlobalMem;
  set_fixed_string(out->name, sizeof(out->name), props.name);
  set_version_string(out->driver_version, sizeof(out->driver_version), driver_version);
  set_version_string(out->runtime_version, sizeof(out->runtime_version), runtime_version);
  return ok();
#else
  out->cuda_available = 0;
  set_fixed_string(out->name, sizeof(out->name), "cuda-disabled-host-fallback");
  set_fixed_string(out->driver_version, sizeof(out->driver_version), "unavailable");
  set_fixed_string(out->runtime_version, sizeof(out->runtime_version), "unavailable");
  return ok();
#endif
}

extern "C" cuteafd_status_t cuteafd_cuda_memory_info(size_t* free_bytes, size_t* total_bytes) {
  if (free_bytes == nullptr || total_bytes == nullptr || free_bytes == total_bytes) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "memory info requires distinct output pointers");
  }
  *free_bytes = 0;
  *total_bytes = 0;
#if CUTEAFD_NATIVE_ENABLE_CUDA
  const cudaError_t err = cudaMemGetInfo(free_bytes, total_bytes);
  if (err != cudaSuccess) {
    return fail_cuda(CUTEAFD_STATUS_INTERNAL_ERROR, "cudaMemGetInfo failed", err);
  }
  return ok();
#else
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE, "memory info requires CUDA");
#endif
}

extern "C" cuteafd_status_t cuteafd_cuda_get_device(int* device_id) {
  if (device_id == nullptr) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "device output is null");
  }
  *device_id = -1;
#if CUTEAFD_NATIVE_ENABLE_CUDA
  const auto err = cudaGetDevice(device_id);
  if (err != cudaSuccess) return fail_cuda(CUTEAFD_STATUS_INTERNAL_ERROR, "cudaGetDevice failed", err);
  return ok();
#else
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE, "device selection requires CUDA");
#endif
}

extern "C" cuteafd_status_t cuteafd_cuda_set_device(int device_id) {
  if (device_id < 0) return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "negative CUDA device");
#if CUTEAFD_NATIVE_ENABLE_CUDA
  const auto err = cudaSetDevice(device_id);
  if (err != cudaSuccess) return fail_cuda(CUTEAFD_STATUS_INTERNAL_ERROR, "cudaSetDevice failed", err);
  return ok();
#else
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE, "device selection requires CUDA");
#endif
}

extern "C" cuteafd_status_t cuteafd_cuda_enable_peer(int peer_device_id) {
  if (peer_device_id < 0) return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "negative peer device");
#if CUTEAFD_NATIVE_ENABLE_CUDA
  int device = -1;
  auto err = cudaGetDevice(&device);
  if (err != cudaSuccess) return fail_cuda(CUTEAFD_STATUS_INTERNAL_ERROR, "cudaGetDevice failed", err);
  if (device == peer_device_id) return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "peer equals current device");
  int capable = 0;
  err = cudaDeviceCanAccessPeer(&capable, device, peer_device_id);
  if (err != cudaSuccess) return fail_cuda(CUTEAFD_STATUS_INTERNAL_ERROR, "peer capability query failed", err);
  if (!capable) return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE, "direct CUDA peer access unavailable");
  err = cudaDeviceEnablePeerAccess(peer_device_id, 0);
  if (err == cudaErrorPeerAccessAlreadyEnabled) {
    cudaGetLastError();
    return ok();
  }
  if (err != cudaSuccess) return fail_cuda(CUTEAFD_STATUS_INTERNAL_ERROR, "peer enable failed", err);
  return ok();
#else
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE, "peer access requires CUDA");
#endif
}

extern "C" cuteafd_status_t cuteafd_copy_peer_async(
    cuteafd_device_buffer_t dst, cuteafd_device_buffer_t src, size_t bytes,
    void* cuda_stream) {
  if (cuda_stream == nullptr || dst.ptr == nullptr || src.ptr == nullptr ||
      dst.device_id < 0 || src.device_id < 0 || dst.device_id == src.device_id ||
      dst.flags != CUTEAFD_DEVICE_BUFFER_FLAG_NONE || src.flags != CUTEAFD_DEVICE_BUFFER_FLAG_NONE ||
      bytes > dst.bytes || bytes > src.bytes) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "invalid peer buffers, extent, or stream");
  }
#if CUTEAFD_NATIVE_ENABLE_CUDA
  int device = -1;
  auto err = cudaGetDevice(&device);
  if (err != cudaSuccess) return fail_cuda(CUTEAFD_STATUS_INTERNAL_ERROR, "cudaGetDevice failed", err);
  if (device != dst.device_id) return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "peer destination is not current device");
  if (bytes == 0) return ok();
  err = cudaMemcpyPeerAsync(dst.ptr, dst.device_id, src.ptr, src.device_id, bytes,
                            reinterpret_cast<cudaStream_t>(cuda_stream));
  if (err != cudaSuccess) return fail_cuda(CUTEAFD_STATUS_COPY_FAILED, "cudaMemcpyPeerAsync failed", err);
  return ok();
#else
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE, "peer copy requires CUDA");
#endif
}

extern "C" cuteafd_status_t cuteafd_copy_device_rows_async(
    cuteafd_device_buffer_t dst, cuteafd_device_buffer_t src, size_t width,
    size_t rows, size_t dst_pitch, size_t src_pitch, void* cuda_stream) {
  const auto span = [width, rows](size_t pitch, size_t bytes) {
    return width > 0 && rows > 0 && pitch >= width && bytes >= width &&
        rows - 1 <= (bytes - width) / pitch;
  };
  if (!cuda_stream || !dst.ptr || !src.ptr || dst.device_id < 0 || src.device_id < 0 ||
      dst.flags != CUTEAFD_DEVICE_BUFFER_FLAG_NONE || src.flags != CUTEAFD_DEVICE_BUFFER_FLAG_NONE ||
      !span(dst_pitch, dst.bytes) || !span(src_pitch, src.bytes)) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "invalid row copy buffers, pitches, extent, or stream");
  }
  const size_t dst_span = (rows - 1) * dst_pitch + width;
  const size_t src_span = (rows - 1) * src_pitch + width;
  const auto d = reinterpret_cast<uintptr_t>(dst.ptr);
  const auto s = reinterpret_cast<uintptr_t>(src.ptr);
  if (dst_span > std::numeric_limits<uintptr_t>::max() - d ||
      src_span > std::numeric_limits<uintptr_t>::max() - s ||
      (d < s + src_span && s < d + dst_span)) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "overlapping or overflowing row copy addresses");
  }
#if CUTEAFD_NATIVE_ENABLE_CUDA
  int device = -1;
  auto err = cudaGetDevice(&device);
  if (err != cudaSuccess) return fail_cuda(CUTEAFD_STATUS_INTERNAL_ERROR, "cudaGetDevice failed", err);
  if (device != dst.device_id) return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "row destination is not current device");
  auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  if (dst.device_id == src.device_id) {
    err = cudaMemcpy2DAsync(dst.ptr, dst_pitch, src.ptr, src_pitch, width, rows,
                            cudaMemcpyDeviceToDevice, stream);
  } else {
    cudaMemcpy3DPeerParms params{};
    params.srcPtr = cudaPitchedPtr{src.ptr, src_pitch, width, rows};
    params.srcDevice = src.device_id;
    params.dstPtr = cudaPitchedPtr{dst.ptr, dst_pitch, width, rows};
    params.dstDevice = dst.device_id;
    params.extent = cudaExtent{width, rows, 1};
    err = cudaMemcpy3DPeerAsync(&params, stream);
  }
  if (err != cudaSuccess) return fail_cuda(CUTEAFD_STATUS_COPY_FAILED, "device row copy failed", err);
  return ok();
#else
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE, "device row copy requires CUDA");
#endif
}

extern "C" cuteafd_status_t cuteafd_alloc_device_buffer(size_t bytes, cuteafd_device_buffer_t* out) {
  if (out == nullptr) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "device buffer output pointer is null");
  }
  if (bytes == 0) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "device buffer allocation size is zero");
  }

#if CUTEAFD_NATIVE_ENABLE_CUDA
  void* ptr = nullptr;
  cudaError_t err = cudaMalloc(&ptr, bytes);
  if (err != cudaSuccess) {
    return fail_cuda(CUTEAFD_STATUS_ALLOCATION_FAILED, "cudaMalloc failed", err);
  }
#else
  void* ptr = std::malloc(bytes);
  if (ptr == nullptr) {
    return fail(CUTEAFD_STATUS_ALLOCATION_FAILED, "host fallback allocation failed");
  }
#endif

  out->ptr = ptr;
  out->bytes = bytes;
#if CUTEAFD_NATIVE_ENABLE_CUDA
  int device_id = -1;
  cudaGetDevice(&device_id);
  out->device_id = device_id;
  out->flags = CUTEAFD_DEVICE_BUFFER_FLAG_NONE;
#else
  out->device_id = -1;
  out->flags = CUTEAFD_DEVICE_BUFFER_FLAG_HOST_FALLBACK;
#endif
  return ok();
}

extern "C" cuteafd_status_t cuteafd_alloc_managed_device_buffer(size_t bytes,
                                                           cuteafd_device_buffer_t* out) {
  if (out == nullptr) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "managed device buffer output pointer is null");
  }
  if (bytes == 0) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "managed device buffer allocation size is zero");
  }

#if CUTEAFD_NATIVE_ENABLE_CUDA
  void* ptr = nullptr;
  cudaError_t err = cudaMallocManaged(&ptr, bytes, cudaMemAttachGlobal);
  if (err != cudaSuccess) {
    return fail_cuda(CUTEAFD_STATUS_ALLOCATION_FAILED, "cudaMallocManaged failed", err);
  }
  out->ptr = ptr;
  out->bytes = bytes;
  int device_id = -1;
  cudaGetDevice(&device_id);
  out->device_id = device_id;
  out->flags = CUTEAFD_DEVICE_BUFFER_FLAG_MANAGED;
  return ok();
#else
  void* ptr = std::malloc(bytes);
  if (ptr == nullptr) {
    return fail(CUTEAFD_STATUS_ALLOCATION_FAILED,
                "managed host fallback allocation failed");
  }
  out->ptr = ptr;
  out->bytes = bytes;
  out->device_id = -1;
  out->flags =
      CUTEAFD_DEVICE_BUFFER_FLAG_HOST_FALLBACK | CUTEAFD_DEVICE_BUFFER_FLAG_MANAGED;
  return ok();
#endif
}

#if !CUTEAFD_NATIVE_ENABLE_CUDA
extern "C" cuteafd_status_t cuteafd_cuda_engram_dequant_bf16_async(
    const uint8_t*, const uint8_t*, uint16_t*, int, void*) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE, "CUDA engram dequant is unavailable in this build");
}
extern "C" cuteafd_status_t cuteafd_cuda_engram_gate_bf16_async(
    const uint16_t*, const uint16_t*, const uint16_t*, const uint16_t*,
    const uint8_t*, uint16_t*, int, void*) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE, "CUDA engram gate is unavailable in this build");
}
extern "C" cuteafd_status_t cuteafd_cuda_b12x_quantize_bf16_nvfp4_row_payload_async(
    cuteafd_device_buffer_t, cuteafd_device_buffer_t, size_t, size_t, void*) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE, "CUDA kernels are not built");
}
#endif

extern "C" cuteafd_status_t cuteafd_alloc_host_buffer(size_t bytes, cuteafd_host_buffer_t* out) {
  if (out == nullptr) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "host buffer output pointer is null");
  }
  if (bytes == 0) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "host buffer allocation size is zero");
  }

#if CUTEAFD_NATIVE_ENABLE_CUDA
  void* ptr = nullptr;
  cudaError_t err =
      cudaHostAlloc(&ptr, bytes, cudaHostAllocPortable | cudaHostAllocMapped);
  if (err != cudaSuccess) {
    return fail_cuda(CUTEAFD_STATUS_ALLOCATION_FAILED, "cudaHostAlloc failed", err);
  }
  out->ptr = ptr;
  out->bytes = bytes;
  out->flags = CUTEAFD_HOST_BUFFER_FLAG_PINNED | CUTEAFD_HOST_BUFFER_FLAG_MAPPED;
#else
  void* ptr = std::malloc(bytes);
  if (ptr == nullptr) {
    return fail(CUTEAFD_STATUS_ALLOCATION_FAILED, "host buffer fallback allocation failed");
  }
  out->ptr = ptr;
  out->bytes = bytes;
  out->flags = CUTEAFD_HOST_BUFFER_FLAG_HOST_FALLBACK;
#endif
  return ok();
}

extern "C" cuteafd_status_t cuteafd_cuda_host_buffer_device_alias(cuteafd_host_buffer_t host,
                                                                cuteafd_device_buffer_t* out) {
  if (out == nullptr) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "mapped host device alias output is null");
  }
  if (host.ptr == nullptr || host.bytes == 0) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "mapped host buffer is empty");
  }
  if ((host.flags & CUTEAFD_HOST_BUFFER_FLAG_MAPPED) == 0) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "host buffer is not CUDA-mapped");
  }

#if CUTEAFD_NATIVE_ENABLE_CUDA
  void* device_ptr = nullptr;
  cudaError_t err = cudaHostGetDevicePointer(&device_ptr, host.ptr, 0);
  if (err != cudaSuccess) {
    return fail_cuda(CUTEAFD_STATUS_INTERNAL_ERROR, "cudaHostGetDevicePointer failed", err);
  }
  int device_id = -1;
  err = cudaGetDevice(&device_id);
  if (err != cudaSuccess) {
    return fail_cuda(CUTEAFD_STATUS_INTERNAL_ERROR, "cudaGetDevice failed", err);
  }
  out->ptr = device_ptr;
  out->bytes = host.bytes;
  out->device_id = device_id;
  out->flags = CUTEAFD_DEVICE_BUFFER_FLAG_MAPPED_HOST;
  return ok();
#else
  (void)host;
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "mapped host device aliases require a CUDA build");
#endif
}

extern "C" cuteafd_status_t cuteafd_free_host_buffer(cuteafd_host_buffer_t* buf) {
  if (buf == nullptr) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "host buffer pointer is null");
  }
  if (buf->ptr != nullptr) {
#if CUTEAFD_NATIVE_ENABLE_CUDA
    cudaError_t err = cudaFreeHost(buf->ptr);
    if (err != cudaSuccess) {
      return fail_cuda(CUTEAFD_STATUS_INTERNAL_ERROR, "cudaFreeHost failed", err);
    }
#else
    std::free(buf->ptr);
#endif
  }
  buf->ptr = nullptr;
  buf->bytes = 0;
  buf->flags = CUTEAFD_HOST_BUFFER_FLAG_NONE;
  return ok();
}

extern "C" cuteafd_status_t cuteafd_free_device_buffer(cuteafd_device_buffer_t* buf) {
  if (buf == nullptr) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "device buffer pointer is null");
  }
  if (buf->ptr != nullptr) {
#if CUTEAFD_NATIVE_ENABLE_CUDA
    cudaError_t err = cudaFree(buf->ptr);
    if (err != cudaSuccess) {
      return fail_cuda(CUTEAFD_STATUS_INTERNAL_ERROR, "cudaFree failed", err);
    }
#else
    std::free(buf->ptr);
#endif
  }
  buf->ptr = nullptr;
  buf->bytes = 0;
  buf->device_id = -1;
  buf->flags = CUTEAFD_DEVICE_BUFFER_FLAG_NONE;
  return ok();
}

extern "C" cuteafd_status_t cuteafd_cuda_stream_create(void** out_cuda_stream) {
  if (out_cuda_stream == nullptr) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "CUDA stream output pointer is null");
  }
  *out_cuda_stream = nullptr;
#if CUTEAFD_NATIVE_ENABLE_CUDA
  cudaStream_t stream = nullptr;
  cudaError_t err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  if (err != cudaSuccess) {
    return fail_cuda(CUTEAFD_STATUS_CUDA_UNAVAILABLE, "cudaStreamCreateWithFlags failed", err);
  }
  *out_cuda_stream = reinterpret_cast<void*>(stream);
  return ok();
#else
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA stream creation is unavailable in this build");
#endif
}

extern "C" cuteafd_status_t cuteafd_cuda_stream_destroy(void* cuda_stream) {
  if (cuda_stream == nullptr) {
    return ok();
  }
#if CUTEAFD_NATIVE_ENABLE_CUDA
  cudaError_t err = cudaStreamDestroy(reinterpret_cast<cudaStream_t>(cuda_stream));
  if (err != cudaSuccess) {
    return fail_cuda(CUTEAFD_STATUS_INTERNAL_ERROR, "cudaStreamDestroy failed", err);
  }
  return ok();
#else
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA stream destruction is unavailable in this build");
#endif
}

extern "C" cuteafd_status_t cuteafd_cuda_stream_synchronize(void* cuda_stream) {
#if CUTEAFD_NATIVE_ENABLE_CUDA
  cudaError_t err = cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(cuda_stream));
  if (err != cudaSuccess) {
    return fail_cuda(CUTEAFD_STATUS_INTERNAL_ERROR, "cudaStreamSynchronize failed", err);
  }
#else
  if (cuda_stream != nullptr) {
    return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
                "CUDA stream synchronization is unavailable in this build");
  }
#endif
  return ok();
}

extern "C" cuteafd_status_t cuteafd_cuda_stream_query(void* cuda_stream, int32_t* ready) {
  if (ready == nullptr) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "CUDA stream query output is null");
  }
  *ready = 0;
#if CUTEAFD_NATIVE_ENABLE_CUDA
  const cudaError_t err = cudaStreamQuery(reinterpret_cast<cudaStream_t>(cuda_stream));
  if (err == cudaErrorNotReady) return ok();
  if (err != cudaSuccess) {
    return fail_cuda(CUTEAFD_STATUS_INTERNAL_ERROR, "cudaStreamQuery failed", err);
  }
  *ready = 1;
  return ok();
#else
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA stream query is unavailable in this build");
#endif
}

extern "C" cuteafd_status_t cuteafd_cuda_stream_wait_event(void* cuda_stream, void* cuda_event) {
  if (cuda_stream == nullptr || cuda_event == nullptr) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT,
                "cuteafd_cuda_stream_wait_event requires non-null stream and event");
  }
#if CUTEAFD_NATIVE_ENABLE_CUDA
  const cudaError_t err = cudaStreamWaitEvent(reinterpret_cast<cudaStream_t>(cuda_stream),
                                              reinterpret_cast<cudaEvent_t>(cuda_event), 0);
  if (err != cudaSuccess) {
    return fail_cuda(CUTEAFD_STATUS_CUDA_UNAVAILABLE, "cudaStreamWaitEvent failed", err);
  }
  return ok();
#else
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA stream event waits are unavailable in this build");
#endif
}

extern "C" cuteafd_status_t cuteafd_cuda_event_create(void** out_cuda_event) {
  if (out_cuda_event == nullptr) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "CUDA event output pointer is null");
  }
  *out_cuda_event = nullptr;
#if CUTEAFD_NATIVE_ENABLE_CUDA
  cudaEvent_t event = nullptr;
  cudaError_t err = cudaEventCreateWithFlags(&event, cudaEventDefault);
  if (err != cudaSuccess) {
    return fail_cuda(CUTEAFD_STATUS_CUDA_UNAVAILABLE, "cudaEventCreateWithFlags failed", err);
  }
  *out_cuda_event = reinterpret_cast<void*>(event);
  return ok();
#else
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE, "CUDA event creation is unavailable in this build");
#endif
}

// Ordering-only event: no timestamp, so record and stream waits are cheaper.
extern "C" cuteafd_status_t cuteafd_cuda_event_create_ordering(void** out_cuda_event) {
  if (out_cuda_event == nullptr) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "CUDA event output pointer is null");
  }
  *out_cuda_event = nullptr;
#if CUTEAFD_NATIVE_ENABLE_CUDA
  cudaEvent_t event = nullptr;
  cudaError_t err = cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
  if (err != cudaSuccess) {
    return fail_cuda(CUTEAFD_STATUS_CUDA_UNAVAILABLE, "cudaEventCreateWithFlags failed", err);
  }
  *out_cuda_event = reinterpret_cast<void*>(event);
  return ok();
#else
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE, "CUDA event creation is unavailable in this build");
#endif
}

extern "C" cuteafd_status_t cuteafd_cuda_event_destroy(void* cuda_event) {
  if (cuda_event == nullptr) {
    return ok();
  }
#if CUTEAFD_NATIVE_ENABLE_CUDA
  cudaError_t err = cudaEventDestroy(reinterpret_cast<cudaEvent_t>(cuda_event));
  if (err != cudaSuccess) {
    return fail_cuda(CUTEAFD_STATUS_INTERNAL_ERROR, "cudaEventDestroy failed", err);
  }
  return ok();
#else
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE, "CUDA event destruction is unavailable in this build");
#endif
}

extern "C" cuteafd_status_t cuteafd_cuda_event_record(void* cuda_event, void* cuda_stream) {
  if (cuda_event == nullptr) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "CUDA event is null");
  }
#if CUTEAFD_NATIVE_ENABLE_CUDA
  cudaError_t err = cudaEventRecord(reinterpret_cast<cudaEvent_t>(cuda_event),
                                    reinterpret_cast<cudaStream_t>(cuda_stream));
  if (err != cudaSuccess) {
    return fail_cuda(CUTEAFD_STATUS_INTERNAL_ERROR, "cudaEventRecord failed", err);
  }
  return ok();
#else
  if (cuda_stream != nullptr) {
    return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE, "CUDA event recording is unavailable in this build");
  }
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE, "CUDA event recording is unavailable in this build");
#endif
}

extern "C" cuteafd_status_t cuteafd_cuda_event_synchronize(void* cuda_event) {
  if (cuda_event == nullptr) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT,
                "cuteafd_cuda_event_synchronize requires a non-null event");
  }
#if CUTEAFD_NATIVE_ENABLE_CUDA
  const cudaError_t err = cudaEventSynchronize(reinterpret_cast<cudaEvent_t>(cuda_event));
  if (err != cudaSuccess) {
    return fail_cuda(CUTEAFD_STATUS_CUDA_UNAVAILABLE, "cudaEventSynchronize failed", err);
  }
  return ok();
#else
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA event synchronization is unavailable in this build");
#endif
}

extern "C" cuteafd_status_t cuteafd_cuda_event_elapsed_ms(void* start_event, void* end_event,
                                                       float* out_ms) {
  if (start_event == nullptr || end_event == nullptr) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "CUDA event elapsed input event is null");
  }
  if (out_ms == nullptr) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "CUDA event elapsed output pointer is null");
  }
  *out_ms = 0.0f;
#if CUTEAFD_NATIVE_ENABLE_CUDA
  cudaError_t err = cudaEventElapsedTime(out_ms, reinterpret_cast<cudaEvent_t>(start_event),
                                         reinterpret_cast<cudaEvent_t>(end_event));
  if (err != cudaSuccess) {
    return fail_cuda(CUTEAFD_STATUS_INTERNAL_ERROR, "cudaEventElapsedTime failed", err);
  }
  return ok();
#else
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE, "CUDA event elapsed timing is unavailable in this build");
#endif
}

extern "C" cuteafd_status_t cuteafd_cuda_graph_begin_capture(void* cuda_stream) {
  if (cuda_stream == nullptr) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "CUDA graph capture stream is null");
  }
#if CUTEAFD_NATIVE_ENABLE_CUDA
  // Runtime graph slots are owned per host thread. Thread-local capture avoids
  // invalidating a graph capture when unrelated test/request threads enqueue
  // CUDA work on their own streams.
  cudaError_t err =
      cudaStreamBeginCapture(reinterpret_cast<cudaStream_t>(cuda_stream),
                             cudaStreamCaptureModeThreadLocal);
  if (err != cudaSuccess) {
    return fail_cuda(CUTEAFD_STATUS_INTERNAL_ERROR, "cudaStreamBeginCapture failed", err);
  }
  return ok();
#else
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA graph capture is unavailable in this build");
#endif
}

extern "C" cuteafd_status_t cuteafd_cuda_graph_end_capture(void* cuda_stream,
                                                        void** out_cuda_graph_exec) {
  if (out_cuda_graph_exec == nullptr) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "CUDA graph exec output pointer is null");
  }
  *out_cuda_graph_exec = nullptr;
  cuteafd_cuda_graph_capture_info_t capture = {};
  cuteafd_status_t status = cuteafd_cuda_graph_end_capture_retained(cuda_stream, &capture);
  if (status != CUTEAFD_STATUS_OK) {
    return status;
  }
#if CUTEAFD_NATIVE_ENABLE_CUDA
  cudaError_t destroy_err = cudaGraphDestroy(reinterpret_cast<cudaGraph_t>(capture.graph));
  if (destroy_err != cudaSuccess) {
    cudaGraphExecDestroy(reinterpret_cast<cudaGraphExec_t>(capture.graph_exec));
    return fail_cuda(CUTEAFD_STATUS_INTERNAL_ERROR, "cudaGraphDestroy failed", destroy_err);
  }
  *out_cuda_graph_exec = capture.graph_exec;
  return ok();
#else
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA graph capture is unavailable in this build");
#endif
}

extern "C" cuteafd_status_t cuteafd_cuda_graph_end_capture_retained(
    void* cuda_stream, cuteafd_cuda_graph_capture_info_t* out) {
  if (cuda_stream == nullptr) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "CUDA graph capture stream is null");
  }
  if (out == nullptr) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "CUDA graph capture output pointer is null");
  }
  *out = {};
#if CUTEAFD_NATIVE_ENABLE_CUDA
  cudaGraph_t graph = nullptr;
  cudaError_t err = cudaStreamEndCapture(reinterpret_cast<cudaStream_t>(cuda_stream), &graph);
  if (err != cudaSuccess) {
    return fail_cuda(CUTEAFD_STATUS_INTERNAL_ERROR, "cudaStreamEndCapture failed", err);
  }
  if (graph == nullptr) {
    return fail(CUTEAFD_STATUS_INTERNAL_ERROR, "cudaStreamEndCapture returned a null graph");
  }
  cudaGraphExec_t graph_exec = nullptr;
  err = cudaGraphInstantiate(&graph_exec, graph, 0);
  if (err != cudaSuccess) {
    cudaGraphDestroy(graph);
    return fail_cuda(CUTEAFD_STATUS_INTERNAL_ERROR, "cudaGraphInstantiate failed", err);
  }
  if (graph_exec == nullptr) {
    cudaGraphDestroy(graph);
    return fail(CUTEAFD_STATUS_INTERNAL_ERROR, "cudaGraphInstantiate returned a null graph exec");
  }
  cuteafd_status_t status = fill_cuda_graph_capture_info(graph, graph_exec, out);
  if (status != CUTEAFD_STATUS_OK) {
    cudaGraphExecDestroy(graph_exec);
    cudaGraphDestroy(graph);
    return status;
  }
  return ok();
#else
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA graph capture is unavailable in this build");
#endif
}

extern "C" cuteafd_status_t cuteafd_cuda_graph_launch(void* cuda_graph_exec, void* cuda_stream) {
  if (cuda_graph_exec == nullptr) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "CUDA graph exec is null");
  }
  if (cuda_stream == nullptr) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "CUDA graph launch stream is null");
  }
#if CUTEAFD_NATIVE_ENABLE_CUDA
  cudaError_t err = cudaGraphLaunch(reinterpret_cast<cudaGraphExec_t>(cuda_graph_exec),
                                    reinterpret_cast<cudaStream_t>(cuda_stream));
  if (err != cudaSuccess) {
    return fail_cuda(CUTEAFD_STATUS_INTERNAL_ERROR, "cudaGraphLaunch failed", err);
  }
  return ok();
#else
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA graph launch is unavailable in this build");
#endif
}

extern "C" cuteafd_status_t cuteafd_cuda_graph_exec_update(void* cuda_graph_exec, void* cuda_graph) {
  if (cuda_graph_exec == nullptr) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "CUDA graph exec update target is null");
  }
  if (cuda_graph == nullptr) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "CUDA graph exec update graph is null");
  }
#if CUTEAFD_NATIVE_ENABLE_CUDA
  cudaGraphExecUpdateResultInfo update_info = {};
  cudaError_t err = cudaGraphExecUpdate(reinterpret_cast<cudaGraphExec_t>(cuda_graph_exec),
                                        reinterpret_cast<cudaGraph_t>(cuda_graph),
                                        &update_info);
  if (err != cudaSuccess) {
    const cuteafd_status_t status =
        fail_cuda(CUTEAFD_STATUS_INTERNAL_ERROR, "cudaGraphExecUpdate failed", err);
    // Callers may replace an incompatible exec with the freshly instantiated
    // graph. Do not leak that recovered update error into the next kernel's
    // launch-status check on this host thread.
    (void)cudaGetLastError();
    return status;
  }
  if (update_info.result != cudaGraphExecUpdateSuccess) {
    return fail(CUTEAFD_STATUS_INTERNAL_ERROR,
                "cudaGraphExecUpdate did not accept graph update; result=" +
                    std::to_string(static_cast<int>(update_info.result)));
  }
  return ok();
#else
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA graph exec update is unavailable in this build");
#endif
}

extern "C" cuteafd_status_t cuteafd_cuda_graph_destroy(void* cuda_graph) {
  if (cuda_graph == nullptr) {
    return ok();
  }
#if CUTEAFD_NATIVE_ENABLE_CUDA
  cudaError_t err = cudaGraphDestroy(reinterpret_cast<cudaGraph_t>(cuda_graph));
  if (err != cudaSuccess) {
    return fail_cuda(CUTEAFD_STATUS_INTERNAL_ERROR, "cudaGraphDestroy failed", err);
  }
  return ok();
#else
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA graph destruction is unavailable in this build");
#endif
}

extern "C" cuteafd_status_t cuteafd_cuda_graph_exec_destroy(void* cuda_graph_exec) {
  if (cuda_graph_exec == nullptr) {
    return ok();
  }
#if CUTEAFD_NATIVE_ENABLE_CUDA
  cudaError_t err = cudaGraphExecDestroy(reinterpret_cast<cudaGraphExec_t>(cuda_graph_exec));
  if (err != cudaSuccess) {
    return fail_cuda(CUTEAFD_STATUS_INTERNAL_ERROR, "cudaGraphExecDestroy failed", err);
  }
  return ok();
#else
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA graph exec destruction is unavailable in this build");
#endif
}

extern "C" cuteafd_status_t cuteafd_copy_h2d(cuteafd_device_buffer_t dst, const void* src, size_t bytes) {
  if (bytes == 0) {
    return ok();
  }
  if (dst.ptr == nullptr) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "destination device buffer is null");
  }
  if (src == nullptr) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "source host pointer is null");
  }
  if (bytes > dst.bytes) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "copy exceeds destination device buffer size");
  }
#if CUTEAFD_NATIVE_ENABLE_CUDA
  cudaError_t err = cudaMemcpy(dst.ptr, src, bytes, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    return fail_cuda(CUTEAFD_STATUS_COPY_FAILED, "cudaMemcpy host-to-device failed", err);
  }
#else
  std::memcpy(dst.ptr, src, bytes);
#endif
  return ok();
}

extern "C" cuteafd_status_t cuteafd_copy_h2d_async(cuteafd_device_buffer_t dst, const void* src,
                                               size_t bytes, void* cuda_stream) {
  if (bytes == 0) {
    return ok();
  }
  if (dst.ptr == nullptr) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "destination device buffer is null");
  }
  if (src == nullptr) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "source host pointer is null");
  }
  if (bytes > dst.bytes) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "copy exceeds destination device buffer size");
  }
#if CUTEAFD_NATIVE_ENABLE_CUDA
  cudaError_t err = cudaMemcpyAsync(dst.ptr, src, bytes, cudaMemcpyHostToDevice,
                                    reinterpret_cast<cudaStream_t>(cuda_stream));
  if (err != cudaSuccess) {
    return fail_cuda(CUTEAFD_STATUS_COPY_FAILED, "cudaMemcpyAsync host-to-device failed", err);
  }
#else
  if (cuda_stream != nullptr) {
    return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
                "CUDA async host-to-device copy is unavailable in this build");
  }
  std::memcpy(dst.ptr, src, bytes);
#endif
  return ok();
}

extern "C" cuteafd_status_t cuteafd_copy_h2d_batch_async(
    const cuteafd_device_buffer_t* dsts, const void* const* srcs, const size_t* bytes,
    size_t count, void* cuda_stream) {
  if (count == 0) {
    return ok();
  }
  if (dsts == nullptr || srcs == nullptr || bytes == nullptr) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "batched H2D copy arrays are null");
  }
  if (cuda_stream == nullptr) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT,
                "batched H2D copy requires a non-default CUDA stream");
  }
  std::vector<void*> destination_ptrs;
  destination_ptrs.reserve(count);
  for (size_t index = 0; index < count; ++index) {
    if (bytes[index] == 0) {
      return fail(CUTEAFD_STATUS_INVALID_ARGUMENT,
                  "batched H2D copy contains an empty entry");
    }
    if (dsts[index].ptr == nullptr || srcs[index] == nullptr) {
      return fail(CUTEAFD_STATUS_INVALID_ARGUMENT,
                  "batched H2D copy contains a null pointer");
    }
    if (bytes[index] > dsts[index].bytes) {
      return fail(CUTEAFD_STATUS_INVALID_ARGUMENT,
                  "batched H2D copy exceeds a destination device buffer");
    }
    destination_ptrs.push_back(dsts[index].ptr);
  }
#if CUTEAFD_NATIVE_ENABLE_CUDA
#if CUDART_VERSION >= 12080
  cudaMemcpyAttributes attributes{};
  attributes.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
  size_t attributes_index = 0;
  cudaError_t err = cudaMemcpyBatchAsync(
      destination_ptrs.data(), srcs, bytes, count, &attributes, &attributes_index, 1,
      reinterpret_cast<cudaStream_t>(cuda_stream));
  if (err != cudaSuccess) {
    return fail_cuda(CUTEAFD_STATUS_COPY_FAILED, "cudaMemcpyBatchAsync host-to-device failed",
                     err);
  }
#else
  for (size_t index = 0; index < count; ++index) {
    cudaError_t err = cudaMemcpyAsync(
        destination_ptrs[index], srcs[index], bytes[index], cudaMemcpyHostToDevice,
        reinterpret_cast<cudaStream_t>(cuda_stream));
    if (err != cudaSuccess) {
      return fail_cuda(CUTEAFD_STATUS_COPY_FAILED,
                       "cudaMemcpyAsync batched host-to-device fallback failed", err);
    }
  }
#endif
#else
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA batched host-to-device copy is unavailable in this build");
#endif
  return ok();
}

extern "C" cuteafd_status_t cuteafd_copy_h2d_2d_async(
    cuteafd_device_buffer_t dst, size_t dst_pitch_bytes, const void* src,
    size_t src_pitch_bytes, size_t width_bytes, size_t rows, void* cuda_stream) {
  if (width_bytes == 0 || rows == 0) {
    return ok();
  }
  if (dst.ptr == nullptr) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "2D copy destination device buffer is null");
  }
  if (src == nullptr) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "2D copy source host pointer is null");
  }
  if (dst_pitch_bytes < width_bytes || src_pitch_bytes < width_bytes) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "2D copy pitch is smaller than row width");
  }
  if (rows - 1 > (std::numeric_limits<size_t>::max() - width_bytes) / dst_pitch_bytes ||
      rows - 1 > (std::numeric_limits<size_t>::max() - width_bytes) / src_pitch_bytes) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "2D copy byte span overflows size_t");
  }
  const size_t dst_required = (rows - 1) * dst_pitch_bytes + width_bytes;
  if (dst_required > dst.bytes) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "2D copy exceeds destination device buffer size");
  }
#if CUTEAFD_NATIVE_ENABLE_CUDA
  cudaError_t err = cudaMemcpy2DAsync(
      dst.ptr, dst_pitch_bytes, src, src_pitch_bytes, width_bytes, rows,
      cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(cuda_stream));
  if (err != cudaSuccess) {
    return fail_cuda(CUTEAFD_STATUS_COPY_FAILED, "cudaMemcpy2DAsync host-to-device failed", err);
  }
#else
  if (cuda_stream != nullptr) {
    return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
                "CUDA async 2D host-to-device copy is unavailable in this build");
  }
  for (size_t row = 0; row < rows; ++row) {
    std::memcpy(static_cast<uint8_t*>(dst.ptr) + row * dst_pitch_bytes,
                static_cast<const uint8_t*>(src) + row * src_pitch_bytes, width_bytes);
  }
#endif
  return ok();
}

extern "C" cuteafd_status_t cuteafd_copy_d2h(void* dst, cuteafd_device_buffer_t src, size_t bytes) {
  if (bytes == 0) {
    return ok();
  }
  if (dst == nullptr) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "destination host pointer is null");
  }
  if (src.ptr == nullptr) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "source device buffer is null");
  }
  if (bytes > src.bytes) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "copy exceeds source device buffer size");
  }
#if CUTEAFD_NATIVE_ENABLE_CUDA
  cudaError_t err = cudaMemcpy(dst, src.ptr, bytes, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    return fail_cuda(CUTEAFD_STATUS_COPY_FAILED, "cudaMemcpy device-to-host failed", err);
  }
#else
  std::memcpy(dst, src.ptr, bytes);
#endif
  return ok();
}

extern "C" cuteafd_status_t cuteafd_copy_d2d(cuteafd_device_buffer_t dst, cuteafd_device_buffer_t src,
                                         size_t bytes) {
  if (bytes == 0) {
    return ok();
  }
  if (dst.ptr == nullptr) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "destination device buffer is null");
  }
  if (src.ptr == nullptr) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "source device buffer is null");
  }
  if (bytes > dst.bytes) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "copy exceeds destination device buffer size");
  }
  if (bytes > src.bytes) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "copy exceeds source device buffer size");
  }
#if CUTEAFD_NATIVE_ENABLE_CUDA
  cudaError_t err = cudaMemcpy(dst.ptr, src.ptr, bytes, cudaMemcpyDeviceToDevice);
  if (err != cudaSuccess) {
    return fail_cuda(CUTEAFD_STATUS_COPY_FAILED, "cudaMemcpy device-to-device failed", err);
  }
  // D2D cudaMemcpy returns before completion. Consumers use nonblocking
  // streams, which do not wait for the default stream's copy. This API promises
  // a completed copy; callers needing stream ordering use copy_d2d_async.
  err = cudaStreamSynchronize(nullptr);
  if (err != cudaSuccess) {
    return fail_cuda(CUTEAFD_STATUS_COPY_FAILED, "device-to-device copy synchronization failed", err);
  }
#else
  std::memcpy(dst.ptr, src.ptr, bytes);
#endif
  return ok();
}

extern "C" cuteafd_status_t cuteafd_copy_d2h_async(void* dst, cuteafd_device_buffer_t src,
                                               size_t bytes, void* cuda_stream) {
  if (bytes == 0) {
    return ok();
  }
  if (dst == nullptr) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "destination host pointer is null");
  }
  if (src.ptr == nullptr) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "source device buffer is null");
  }
  if (bytes > src.bytes) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "copy exceeds source device buffer size");
  }
#if CUTEAFD_NATIVE_ENABLE_CUDA
  cudaError_t err = cudaMemcpyAsync(dst, src.ptr, bytes, cudaMemcpyDeviceToHost,
                                    reinterpret_cast<cudaStream_t>(cuda_stream));
  if (err != cudaSuccess) {
    return fail_cuda(CUTEAFD_STATUS_COPY_FAILED, "cudaMemcpyAsync device-to-host failed", err);
  }
#else
  if (cuda_stream != nullptr) {
    return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
                "CUDA async device-to-host copy is unavailable in this build");
  }
  std::memcpy(dst, src.ptr, bytes);
#endif
  return ok();
}

extern "C" cuteafd_status_t cuteafd_copy_d2d_async(cuteafd_device_buffer_t dst,
                                               cuteafd_device_buffer_t src, size_t bytes,
                                               void* cuda_stream) {
  if (bytes == 0) {
    return ok();
  }
  if (dst.ptr == nullptr) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "destination device buffer is null");
  }
  if (src.ptr == nullptr) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "source device buffer is null");
  }
  if (bytes > dst.bytes) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "copy exceeds destination device buffer size");
  }
  if (bytes > src.bytes) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "copy exceeds source device buffer size");
  }
#if CUTEAFD_NATIVE_ENABLE_CUDA
  cudaError_t err = cudaMemcpyAsync(dst.ptr, src.ptr, bytes, cudaMemcpyDeviceToDevice,
                                    reinterpret_cast<cudaStream_t>(cuda_stream));
  if (err != cudaSuccess) {
    return fail_cuda(CUTEAFD_STATUS_COPY_FAILED, "cudaMemcpyAsync device-to-device failed", err);
  }
#else
  if (cuda_stream != nullptr) {
    return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
                "CUDA async device-to-device copy is unavailable in this build");
  }
  std::memcpy(dst.ptr, src.ptr, bytes);
#endif
  return ok();
}

extern "C" cuteafd_status_t cuteafd_copy_d2d_2d_async(
    cuteafd_device_buffer_t dst, size_t dst_pitch_bytes, cuteafd_device_buffer_t src,
    size_t src_pitch_bytes, size_t width_bytes, size_t rows, void* cuda_stream) {
  if (width_bytes == 0 || rows == 0) {
    return ok();
  }
  if (dst.ptr == nullptr) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "2D copy destination device buffer is null");
  }
  if (src.ptr == nullptr) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "2D copy source device buffer is null");
  }
  if (dst_pitch_bytes < width_bytes || src_pitch_bytes < width_bytes) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "2D copy pitch is smaller than row width");
  }
  if (rows - 1 > (std::numeric_limits<size_t>::max() - width_bytes) / dst_pitch_bytes ||
      rows - 1 > (std::numeric_limits<size_t>::max() - width_bytes) / src_pitch_bytes) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "2D copy byte span overflows size_t");
  }
  const size_t dst_required = (rows - 1) * dst_pitch_bytes + width_bytes;
  const size_t src_required = (rows - 1) * src_pitch_bytes + width_bytes;
  if (dst_required > dst.bytes) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "2D copy exceeds destination device buffer size");
  }
  if (src_required > src.bytes) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "2D copy exceeds source device buffer size");
  }
#if CUTEAFD_NATIVE_ENABLE_CUDA
  cudaError_t err = cudaMemcpy2DAsync(
      dst.ptr, dst_pitch_bytes, src.ptr, src_pitch_bytes, width_bytes, rows,
      cudaMemcpyDeviceToDevice, reinterpret_cast<cudaStream_t>(cuda_stream));
  if (err != cudaSuccess) {
    return fail_cuda(CUTEAFD_STATUS_COPY_FAILED, "cudaMemcpy2DAsync device-to-device failed", err);
  }
#else
  if (cuda_stream != nullptr) {
    return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
                "CUDA async 2D device-to-device copy is unavailable in this build");
  }
  for (size_t row = 0; row < rows; ++row) {
    std::memcpy(static_cast<uint8_t*>(dst.ptr) + row * dst_pitch_bytes,
                static_cast<const uint8_t*>(src.ptr) + row * src_pitch_bytes, width_bytes);
  }
#endif
  return ok();
}

extern "C" cuteafd_status_t cuteafd_last_error(char* out, size_t out_len) {
  if (out == nullptr || out_len == 0) {
    return CUTEAFD_STATUS_INVALID_ARGUMENT;
  }
  if (g_last_error.size() + 1 > out_len) {
    const size_t copy_len = out_len - 1;
    if (copy_len > 0) {
      std::memcpy(out, g_last_error.data(), copy_len);
    }
    out[out_len - 1] = '\0';
    return CUTEAFD_STATUS_BUFFER_TOO_SMALL;
  }
  std::memcpy(out, g_last_error.c_str(), g_last_error.size() + 1);
  return CUTEAFD_STATUS_OK;
}

extern "C" cuteafd_status_t cuteafd_nccl_unique_id_bytes(size_t* out_bytes) {
  if (out_bytes == nullptr) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "NCCL unique ID byte output is null");
  }
#if CUTEAFD_NATIVE_ENABLE_NCCL
  *out_bytes = sizeof(ncclUniqueId);
  return ok();
#else
  *out_bytes = 0;
  return fail(CUTEAFD_STATUS_NCCL_UNAVAILABLE, "NCCL is unavailable in this native build");
#endif
}

extern "C" cuteafd_status_t cuteafd_nccl_get_unique_id(void* out, size_t out_bytes) {
#if CUTEAFD_NATIVE_ENABLE_NCCL
  if (out == nullptr) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "NCCL unique ID output is null");
  }
  if (out_bytes != sizeof(ncclUniqueId)) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT,
                "NCCL unique ID output has an unexpected byte size");
  }
  ncclUniqueId unique_id{};
  const ncclResult_t result = ncclGetUniqueId(&unique_id);
  if (result != ncclSuccess) {
    return fail_nccl("ncclGetUniqueId failed", result);
  }
  std::memcpy(out, &unique_id, sizeof(unique_id));
  return ok();
#else
  (void)out;
  (void)out_bytes;
  return fail(CUTEAFD_STATUS_NCCL_UNAVAILABLE, "NCCL is unavailable in this native build");
#endif
}

extern "C" cuteafd_status_t cuteafd_nccl_comm_init_rank(const void* unique_id,
                                                      size_t unique_id_bytes, int world_size,
                                                      int rank, void** out_handle) {
#if CUTEAFD_NATIVE_ENABLE_NCCL
  if (unique_id == nullptr || out_handle == nullptr) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT,
                "NCCL communicator unique ID or output handle is null");
  }
  *out_handle = nullptr;
  if (unique_id_bytes != sizeof(ncclUniqueId)) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT,
                "NCCL communicator unique ID has an unexpected byte size");
  }
  if (world_size <= 1 || rank < 0 || rank >= world_size) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "NCCL communicator rank configuration is invalid");
  }
  ncclUniqueId id{};
  std::memcpy(&id, unique_id, sizeof(id));
  auto* handle = new (std::nothrow) NcclCommHandle();
  if (handle == nullptr) {
    return fail(CUTEAFD_STATUS_ALLOCATION_FAILED, "allocating NCCL communicator handle failed");
  }
  const ncclResult_t result = ncclCommInitRank(&handle->comm, world_size, id, rank);
  if (result != ncclSuccess) {
    delete handle;
    return fail_nccl("ncclCommInitRank failed", result);
  }
  handle->world_size = world_size;
  handle->rank = rank;
  *out_handle = handle;
  return ok();
#else
  (void)unique_id;
  (void)unique_id_bytes;
  (void)world_size;
  (void)rank;
  if (out_handle != nullptr) {
    *out_handle = nullptr;
  }
  return fail(CUTEAFD_STATUS_NCCL_UNAVAILABLE, "NCCL is unavailable in this native build");
#endif
}

extern "C" cuteafd_status_t cuteafd_nccl_gather_u8_async(
    void* opaque_handle, cuteafd_device_buffer_t send, cuteafd_device_buffer_t recv, size_t bytes,
    int root, void* cuda_stream) {
#if CUTEAFD_NATIVE_ENABLE_NCCL
  auto* handle = static_cast<NcclCommHandle*>(opaque_handle);
  if (handle == nullptr || handle->comm == nullptr || cuda_stream == nullptr) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT,
                "NCCL gather handle or CUDA stream is null");
  }
  if (root < 0 || root >= handle->world_size || bytes == 0 || send.ptr == nullptr ||
      send.bytes < bytes) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "NCCL gather send contract is invalid");
  }
  const size_t peer_count = static_cast<size_t>(handle->world_size - 1);
  if (handle->rank == root) {
    if (bytes > std::numeric_limits<size_t>::max() / peer_count || recv.ptr == nullptr ||
        recv.bytes < bytes * peer_count) {
      return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "NCCL gather receive buffer is too small");
    }
  }
  ncclResult_t result = ncclGroupStart();
  if (result != ncclSuccess) {
    return fail_nccl("ncclGroupStart failed", result);
  }
  if (handle->rank == root) {
    size_t recv_index = 0;
    for (int peer = 0; peer < handle->world_size; ++peer) {
      if (peer == root) {
        continue;
      }
      void* peer_recv = static_cast<unsigned char*>(recv.ptr) + recv_index * bytes;
      result = ncclRecv(peer_recv, bytes, ncclUint8, peer, handle->comm,
                        reinterpret_cast<cudaStream_t>(cuda_stream));
      if (result != ncclSuccess) {
        break;
      }
      ++recv_index;
    }
  } else {
    result = ncclSend(send.ptr, bytes, ncclUint8, root, handle->comm,
                      reinterpret_cast<cudaStream_t>(cuda_stream));
  }
  const ncclResult_t group_result = ncclGroupEnd();
  if (result != ncclSuccess) {
    return fail_nccl("NCCL gather operation failed", result);
  }
  if (group_result != ncclSuccess) {
    return fail_nccl("ncclGroupEnd failed", group_result);
  }
  return ok();
#else
  (void)opaque_handle;
  (void)send;
  (void)recv;
  (void)bytes;
  (void)root;
  (void)cuda_stream;
  return fail(CUTEAFD_STATUS_NCCL_UNAVAILABLE, "NCCL is unavailable in this native build");
#endif
}

extern "C" cuteafd_status_t cuteafd_nccl_row_all_to_all_u8_async(
    void* opaque_handle, cuteafd_device_buffer_t send, cuteafd_device_buffer_t recv, size_t rows,
    size_t row_stride_bytes, void* cuda_stream) {
#if CUTEAFD_NATIVE_ENABLE_NCCL
  auto* handle = static_cast<NcclCommHandle*>(opaque_handle);
  if (handle == nullptr || handle->comm == nullptr || cuda_stream == nullptr) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT,
                "NCCL row all-to-all handle or CUDA stream is null");
  }
  const size_t world_size = static_cast<size_t>(handle->world_size);
  const size_t rank = static_cast<size_t>(handle->rank);
  if (rows < world_size || row_stride_bytes == 0 ||
      rows > std::numeric_limits<size_t>::max() / row_stride_bytes) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT,
                "NCCL row all-to-all shape is invalid");
  }
  const size_t send_bytes = rows * row_stride_bytes;
  const size_t base_rows = rows / world_size;
  const size_t extra_rows = rows % world_size;
  const size_t local_rows = base_rows + (rank < extra_rows ? 1 : 0);
  if (local_rows > std::numeric_limits<size_t>::max() / row_stride_bytes) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT,
                "NCCL row all-to-all local byte count overflowed");
  }
  const size_t local_bytes = local_rows * row_stride_bytes;
  const size_t peer_count = world_size - 1;
  if (local_bytes > std::numeric_limits<size_t>::max() / peer_count || send.ptr == nullptr ||
      send.bytes < send_bytes || recv.ptr == nullptr || recv.bytes < local_bytes * peer_count) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT,
                "NCCL row all-to-all device buffer is too small");
  }

  ncclResult_t result = ncclGroupStart();
  if (result != ncclSuccess) {
    return fail_nccl("ncclGroupStart failed", result);
  }
  size_t recv_index = 0;
  for (size_t peer = 0; peer < world_size; ++peer) {
    if (peer == rank) {
      continue;
    }
    const size_t peer_rows = base_rows + (peer < extra_rows ? 1 : 0);
    const size_t peer_row_start = peer * base_rows + std::min(peer, extra_rows);
    const size_t peer_bytes = peer_rows * row_stride_bytes;
    const void* peer_send = static_cast<const unsigned char*>(send.ptr) +
                            peer_row_start * row_stride_bytes;
    void* peer_recv = static_cast<unsigned char*>(recv.ptr) + recv_index * local_bytes;
    result = ncclSend(peer_send, peer_bytes, ncclUint8, static_cast<int>(peer), handle->comm,
                      reinterpret_cast<cudaStream_t>(cuda_stream));
    if (result != ncclSuccess) {
      break;
    }
    result = ncclRecv(peer_recv, local_bytes, ncclUint8, static_cast<int>(peer), handle->comm,
                      reinterpret_cast<cudaStream_t>(cuda_stream));
    if (result != ncclSuccess) {
      break;
    }
    ++recv_index;
  }
  const ncclResult_t group_result = ncclGroupEnd();
  if (result != ncclSuccess) {
    return fail_nccl("NCCL row all-to-all operation failed", result);
  }
  if (group_result != ncclSuccess) {
    return fail_nccl("ncclGroupEnd failed", group_result);
  }
  return ok();
#else
  (void)opaque_handle;
  (void)send;
  (void)recv;
  (void)rows;
  (void)row_stride_bytes;
  (void)cuda_stream;
  return fail(CUTEAFD_STATUS_NCCL_UNAVAILABLE, "NCCL is unavailable in this native build");
#endif
}

extern "C" cuteafd_status_t cuteafd_nccl_all_reduce_bf16_async(
    void* opaque_handle, cuteafd_device_buffer_t send, cuteafd_device_buffer_t recv, size_t values,
    void* cuda_stream) {
#if CUTEAFD_NATIVE_ENABLE_NCCL
  auto* handle = static_cast<NcclCommHandle*>(opaque_handle);
  if (handle == nullptr || handle->comm == nullptr || cuda_stream == nullptr) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT,
                "NCCL BF16 all-reduce handle or CUDA stream is null");
  }
  if (values == 0 || values > std::numeric_limits<size_t>::max() / sizeof(uint16_t)) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "NCCL BF16 all-reduce value count is invalid");
  }
  const size_t bytes = values * sizeof(uint16_t);
  if (send.ptr == nullptr || send.bytes < bytes || recv.ptr == nullptr || recv.bytes < bytes) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT,
                "NCCL BF16 all-reduce device buffer is too small");
  }
  const ncclResult_t result =
      ncclAllReduce(send.ptr, recv.ptr, values, ncclBfloat16, ncclSum, handle->comm,
                    reinterpret_cast<cudaStream_t>(cuda_stream));
  if (result != ncclSuccess) {
    return fail_nccl("NCCL BF16 all-reduce failed", result);
  }
  return ok();
#else
  (void)opaque_handle;
  (void)send;
  (void)recv;
  (void)values;
  (void)cuda_stream;
  return fail(CUTEAFD_STATUS_NCCL_UNAVAILABLE, "NCCL is unavailable in this native build");
#endif
}

extern "C" cuteafd_status_t cuteafd_nccl_reduce_bf16_async(
    void* opaque_handle, cuteafd_device_buffer_t send, cuteafd_device_buffer_t recv, size_t values,
    int root, void* cuda_stream) {
#if CUTEAFD_NATIVE_ENABLE_NCCL
  auto* handle = static_cast<NcclCommHandle*>(opaque_handle);
  if (handle == nullptr || handle->comm == nullptr || cuda_stream == nullptr) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT,
                "NCCL BF16 reduce handle or CUDA stream is null");
  }
  if (root < 0 || root >= handle->world_size || values == 0 ||
      values > std::numeric_limits<size_t>::max() / sizeof(uint16_t)) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "NCCL BF16 reduce contract is invalid");
  }
  const size_t bytes = values * sizeof(uint16_t);
  if (send.ptr == nullptr || send.bytes < bytes ||
      (handle->rank == root && (recv.ptr == nullptr || recv.bytes < bytes))) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "NCCL BF16 reduce device buffer is too small");
  }
  const ncclResult_t result =
      ncclReduce(send.ptr, recv.ptr, values, ncclBfloat16, ncclSum, root, handle->comm,
                 reinterpret_cast<cudaStream_t>(cuda_stream));
  if (result != ncclSuccess) {
    return fail_nccl("NCCL BF16 reduce failed", result);
  }
  return ok();
#else
  (void)opaque_handle;
  (void)send;
  (void)recv;
  (void)values;
  (void)root;
  (void)cuda_stream;
  return fail(CUTEAFD_STATUS_NCCL_UNAVAILABLE, "NCCL is unavailable in this native build");
#endif
}

extern "C" cuteafd_status_t cuteafd_nccl_comm_destroy(void* opaque_handle) {
#if CUTEAFD_NATIVE_ENABLE_NCCL
  auto* handle = static_cast<NcclCommHandle*>(opaque_handle);
  if (handle == nullptr) {
    return ok();
  }
  const ncclResult_t result = ncclCommDestroy(handle->comm);
  delete handle;
  if (result != ncclSuccess) {
    return fail_nccl("ncclCommDestroy failed", result);
  }
  return ok();
#else
  (void)opaque_handle;
  return fail(CUTEAFD_STATUS_NCCL_UNAVAILABLE, "NCCL is unavailable in this native build");
#endif
}

extern "C" cuteafd_status_t cuteafd_rdma_device_info(cuteafd_rdma_device_info_t* out) {
  if (out == nullptr) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "RDMA device info output pointer is null");
  }
  std::memset(out, 0, sizeof(*out));
  out->rdma_enabled = CUTEAFD_NATIVE_ENABLE_RDMA ? 1 : 0;

#if CUTEAFD_NATIVE_ENABLE_RDMA
  int device_count = 0;
  ibv_device** devices = ibv_get_device_list(&device_count);
  if (devices == nullptr) {
    return fail(CUTEAFD_STATUS_RDMA_UNAVAILABLE, "ibv_get_device_list failed");
  }
  out->device_count = device_count;
  if (device_count <= 0) {
    set_fixed_string(out->first_device_name, sizeof(out->first_device_name),
                     "no-rdma-devices");
    set_fixed_string(out->first_device_transport, sizeof(out->first_device_transport),
                     "libibverbs");
    set_fixed_string(out->status, sizeof(out->status), "no RDMA devices reported by libibverbs");
    ibv_free_device_list(devices);
    return ok();
  }
  set_first_rdma_device_info(devices[0], out);
  ibv_free_device_list(devices);
  return ok();
#else
  set_fixed_string(out->first_device_name, sizeof(out->first_device_name), "rdma-disabled-build");
  set_fixed_string(out->first_device_transport, sizeof(out->first_device_transport),
                   "libibverbs-disabled");
  set_fixed_string(out->status, sizeof(out->status),
                   "native library built with CUTEAFD_ENABLE_RDMA=OFF");
  return ok();
#endif
}

extern "C" cuteafd_status_t cuteafd_rdma_plan_host_buffer_registration(
    const void* ptr, size_t bytes, size_t alignment, cuteafd_rdma_host_buffer_plan_t* out) {
  return compute_host_buffer_plan(ptr, bytes, alignment, out);
}

extern "C" cuteafd_status_t cuteafd_rdma_register_host_buffer_probe(
    void* ptr, size_t bytes, cuteafd_rdma_register_probe_t* out) {
  if (out == nullptr) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "RDMA register probe output pointer is null");
  }
  if (ptr == nullptr) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "RDMA register probe buffer pointer is null");
  }
  if (bytes == 0) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "RDMA register probe byte size is zero");
  }
  std::memset(out, 0, sizeof(*out));
  out->bytes = bytes;

#if CUTEAFD_NATIVE_ENABLE_RDMA
  int device_count = 0;
  ibv_device** devices = ibv_get_device_list(&device_count);
  if (devices == nullptr) {
    return fail(CUTEAFD_STATUS_RDMA_UNAVAILABLE, "ibv_get_device_list failed");
  }
  if (device_count <= 0) {
    ibv_free_device_list(devices);
    return fail(CUTEAFD_STATUS_RDMA_UNAVAILABLE,
                "no RDMA devices available for host-buffer registration probe");
  }
  ibv_device* device = devices[0];
  const char* device_name = ibv_get_device_name(device);
  set_fixed_string(out->device_name, sizeof(out->device_name),
                   device_name != nullptr ? device_name : "unknown-rdma-device");

  ibv_context* context = ibv_open_device(device);
  if (context == nullptr) {
    ibv_free_device_list(devices);
    return fail(CUTEAFD_STATUS_RDMA_UNAVAILABLE,
                "ibv_open_device failed for host-buffer registration probe");
  }
  ibv_pd* pd = ibv_alloc_pd(context);
  if (pd == nullptr) {
    ibv_close_device(context);
    ibv_free_device_list(devices);
    return fail(CUTEAFD_STATUS_RDMA_UNAVAILABLE,
                "ibv_alloc_pd failed for host-buffer registration probe");
  }
  const int access = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;
  ibv_mr* mr = ibv_reg_mr(pd, ptr, bytes, access);
  if (mr == nullptr) {
    ibv_dealloc_pd(pd);
    ibv_close_device(context);
    ibv_free_device_list(devices);
    return fail(CUTEAFD_STATUS_RDMA_UNAVAILABLE,
                "ibv_reg_mr failed for host-buffer registration probe");
  }

  out->registered = 1;
  out->lkey = mr->lkey;
  out->rkey = mr->rkey;
  const int dereg_status = ibv_dereg_mr(mr);
  const int dealloc_status = ibv_dealloc_pd(pd);
  const int close_status = ibv_close_device(context);
  ibv_free_device_list(devices);
  if (dereg_status != 0 || dealloc_status != 0 || close_status != 0) {
    return fail(CUTEAFD_STATUS_INTERNAL_ERROR,
                "RDMA host-buffer registration probe cleanup failed");
  }
  return ok();
#else
  set_fixed_string(out->device_name, sizeof(out->device_name), "rdma-disabled-build");
  return fail(CUTEAFD_STATUS_RDMA_UNAVAILABLE,
              "RDMA host-buffer registration probe requires CUTEAFD_ENABLE_RDMA=ON");
#endif
}

extern "C" cuteafd_status_t cuteafd_rdma_create_rc_qp_probe(uint32_t port_num, uint32_t send_wr,
                                                        uint32_t recv_wr, uint32_t max_sge,
                                                        cuteafd_rdma_rc_qp_probe_t* out) {
  if (out == nullptr) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "RDMA RC QP probe output pointer is null");
  }
  if (port_num == 0) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "RDMA RC QP probe port number is zero");
  }
  if (send_wr == 0 || recv_wr == 0) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT,
                "RDMA RC QP probe work-request counts must be non-zero");
  }
  if (max_sge == 0) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "RDMA RC QP probe max_sge is zero");
  }
  std::memset(out, 0, sizeof(*out));
  out->rdma_enabled = CUTEAFD_NATIVE_ENABLE_RDMA ? 1 : 0;
  out->port_num = port_num;
  out->requested_send_wr = send_wr;
  out->requested_recv_wr = recv_wr;
  out->requested_max_sge = max_sge;

#if CUTEAFD_NATIVE_ENABLE_RDMA
  int device_count = 0;
  ibv_device** devices = ibv_get_device_list(&device_count);
  if (devices == nullptr) {
    return fail(CUTEAFD_STATUS_RDMA_UNAVAILABLE, "ibv_get_device_list failed");
  }
  if (device_count <= 0) {
    ibv_free_device_list(devices);
    return fail(CUTEAFD_STATUS_RDMA_UNAVAILABLE,
                "no RDMA devices available for RC QP creation probe");
  }

  ibv_device* device = devices[0];
  const char* device_name = ibv_get_device_name(device);
  set_fixed_string(out->device_name, sizeof(out->device_name),
                   device_name != nullptr ? device_name : "unknown-rdma-device");

  ibv_context* context = ibv_open_device(device);
  if (context == nullptr) {
    ibv_free_device_list(devices);
    return fail(CUTEAFD_STATUS_RDMA_UNAVAILABLE, "ibv_open_device failed for RC QP probe");
  }

  ibv_port_attr port_attr = {};
  if (ibv_query_port(context, static_cast<uint8_t>(port_num), &port_attr) != 0) {
    ibv_close_device(context);
    ibv_free_device_list(devices);
    return fail(CUTEAFD_STATUS_RDMA_UNAVAILABLE, "ibv_query_port failed for RC QP probe");
  }

  ibv_pd* pd = ibv_alloc_pd(context);
  if (pd == nullptr) {
    ibv_close_device(context);
    ibv_free_device_list(devices);
    return fail(CUTEAFD_STATUS_RDMA_UNAVAILABLE, "ibv_alloc_pd failed for RC QP probe");
  }

  const int cq_depth = static_cast<int>(std::max(send_wr, recv_wr));
  ibv_cq* cq = ibv_create_cq(context, cq_depth, nullptr, nullptr, 0);
  if (cq == nullptr) {
    ibv_dealloc_pd(pd);
    ibv_close_device(context);
    ibv_free_device_list(devices);
    return fail(CUTEAFD_STATUS_RDMA_UNAVAILABLE, "ibv_create_cq failed for RC QP probe");
  }

  ibv_qp_init_attr qp_attr = {};
  qp_attr.send_cq = cq;
  qp_attr.recv_cq = cq;
  qp_attr.qp_type = IBV_QPT_RC;
  qp_attr.cap.max_send_wr = send_wr;
  qp_attr.cap.max_recv_wr = recv_wr;
  qp_attr.cap.max_send_sge = max_sge;
  qp_attr.cap.max_recv_sge = max_sge;
  qp_attr.cap.max_inline_data = 0;

  ibv_qp* qp = ibv_create_qp(pd, &qp_attr);
  if (qp == nullptr) {
    ibv_destroy_cq(cq);
    ibv_dealloc_pd(pd);
    ibv_close_device(context);
    ibv_free_device_list(devices);
    return fail(CUTEAFD_STATUS_RDMA_UNAVAILABLE, "ibv_create_qp failed for RC QP probe");
  }

  out->created = 1;
  out->qp_num = qp->qp_num;
  out->lid = port_attr.lid;
  out->active_mtu = static_cast<uint32_t>(port_attr.active_mtu);
  out->actual_max_send_wr = qp_attr.cap.max_send_wr;
  out->actual_max_recv_wr = qp_attr.cap.max_recv_wr;
  out->actual_max_send_sge = qp_attr.cap.max_send_sge;
  out->actual_max_recv_sge = qp_attr.cap.max_recv_sge;
  out->actual_max_inline_data = qp_attr.cap.max_inline_data;
  set_fixed_string(out->status, sizeof(out->status), "RC QP resources created and destroyed");

  const int destroy_qp_status = ibv_destroy_qp(qp);
  const int destroy_cq_status = ibv_destroy_cq(cq);
  const int dealloc_pd_status = ibv_dealloc_pd(pd);
  const int close_status = ibv_close_device(context);
  ibv_free_device_list(devices);
  if (destroy_qp_status != 0 || destroy_cq_status != 0 || dealloc_pd_status != 0 ||
      close_status != 0) {
    return fail(CUTEAFD_STATUS_INTERNAL_ERROR, "RDMA RC QP probe cleanup failed");
  }
  return ok();
#else
  set_fixed_string(out->device_name, sizeof(out->device_name), "rdma-disabled-build");
  set_fixed_string(out->status, sizeof(out->status),
                   "RC QP probe requires CUTEAFD_ENABLE_RDMA=ON");
  return fail(CUTEAFD_STATUS_RDMA_UNAVAILABLE,
              "RDMA RC QP creation probe requires CUTEAFD_ENABLE_RDMA=ON");
#endif
}

extern "C" cuteafd_status_t cuteafd_rdma_rc_send_recv_loopback_probe(
    uint32_t port_num, size_t bytes, cuteafd_rdma_rc_send_recv_probe_t* out) {
  if (out == nullptr) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "RDMA RC send/recv probe output pointer is null");
  }
  if (port_num == 0 || port_num > 255) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "RDMA RC send/recv probe port number is invalid");
  }
  if (bytes == 0) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "RDMA RC send/recv probe byte size is zero");
  }
  if (bytes > std::numeric_limits<uint32_t>::max()) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "RDMA RC send/recv probe byte size exceeds u32");
  }
  std::memset(out, 0, sizeof(*out));
  out->rdma_enabled = CUTEAFD_NATIVE_ENABLE_RDMA ? 1 : 0;
  out->port_num = port_num;
  out->bytes = bytes;

#if CUTEAFD_NATIVE_ENABLE_RDMA
  int device_count = 0;
  ibv_device** devices = ibv_get_device_list(&device_count);
  if (devices == nullptr) {
    return fail(CUTEAFD_STATUS_RDMA_UNAVAILABLE, "ibv_get_device_list failed");
  }
  if (device_count <= 0) {
    ibv_free_device_list(devices);
    return fail(CUTEAFD_STATUS_RDMA_UNAVAILABLE,
                "no RDMA devices available for RC send/recv loopback probe");
  }

  ibv_context* context = nullptr;
  ibv_pd* pd = nullptr;
  ibv_cq* cq = nullptr;
  ibv_qp* sender_qp = nullptr;
  ibv_qp* receiver_qp = nullptr;
  ibv_mr* send_mr = nullptr;
  ibv_mr* recv_mr = nullptr;
  unsigned char* send_buffer = nullptr;
  unsigned char* recv_buffer = nullptr;

  auto cleanup = [&]() {
    if (receiver_qp != nullptr) {
      ibv_destroy_qp(receiver_qp);
    }
    if (sender_qp != nullptr) {
      ibv_destroy_qp(sender_qp);
    }
    if (recv_mr != nullptr) {
      ibv_dereg_mr(recv_mr);
    }
    if (send_mr != nullptr) {
      ibv_dereg_mr(send_mr);
    }
    if (cq != nullptr) {
      ibv_destroy_cq(cq);
    }
    if (pd != nullptr) {
      ibv_dealloc_pd(pd);
    }
    if (context != nullptr) {
      ibv_close_device(context);
    }
    if (devices != nullptr) {
      ibv_free_device_list(devices);
    }
    std::free(send_buffer);
    std::free(recv_buffer);
  };

  ibv_device* device = devices[0];
  const char* device_name = ibv_get_device_name(device);
  set_fixed_string(out->device_name, sizeof(out->device_name),
                   device_name != nullptr ? device_name : "unknown-rdma-device");

  context = ibv_open_device(device);
  if (context == nullptr) {
    cleanup();
    return fail(CUTEAFD_STATUS_RDMA_UNAVAILABLE, "ibv_open_device failed for RC send/recv probe");
  }

  ibv_port_attr port_attr = {};
  if (ibv_query_port(context, static_cast<uint8_t>(port_num), &port_attr) != 0) {
    cleanup();
    return fail(CUTEAFD_STATUS_RDMA_UNAVAILABLE, "ibv_query_port failed for RC send/recv probe");
  }

  pd = ibv_alloc_pd(context);
  if (pd == nullptr) {
    cleanup();
    return fail(CUTEAFD_STATUS_RDMA_UNAVAILABLE, "ibv_alloc_pd failed for RC send/recv probe");
  }
  cq = ibv_create_cq(context, 4, nullptr, nullptr, 0);
  if (cq == nullptr) {
    cleanup();
    return fail(CUTEAFD_STATUS_RDMA_UNAVAILABLE, "ibv_create_cq failed for RC send/recv probe");
  }

  ibv_qp_init_attr qp_attr = {};
  qp_attr.send_cq = cq;
  qp_attr.recv_cq = cq;
  qp_attr.qp_type = IBV_QPT_RC;
  qp_attr.cap.max_send_wr = 4;
  qp_attr.cap.max_recv_wr = 4;
  qp_attr.cap.max_send_sge = 1;
  qp_attr.cap.max_recv_sge = 1;

  sender_qp = ibv_create_qp(pd, &qp_attr);
  if (sender_qp == nullptr) {
    cleanup();
    return fail(CUTEAFD_STATUS_RDMA_UNAVAILABLE, "ibv_create_qp sender failed");
  }
  receiver_qp = ibv_create_qp(pd, &qp_attr);
  if (receiver_qp == nullptr) {
    cleanup();
    return fail(CUTEAFD_STATUS_RDMA_UNAVAILABLE, "ibv_create_qp receiver failed");
  }
  out->sender_qp_num = sender_qp->qp_num;
  out->receiver_qp_num = receiver_qp->qp_num;

  send_buffer = static_cast<unsigned char*>(std::malloc(bytes));
  recv_buffer = static_cast<unsigned char*>(std::malloc(bytes));
  if (send_buffer == nullptr || recv_buffer == nullptr) {
    cleanup();
    return fail(CUTEAFD_STATUS_ALLOCATION_FAILED,
                "allocating RC send/recv loopback host buffers failed");
  }
  for (size_t idx = 0; idx < bytes; ++idx) {
    send_buffer[idx] = static_cast<unsigned char>((idx * 31 + 7) & 0xff);
    recv_buffer[idx] = 0;
  }

  send_mr = ibv_reg_mr(pd, send_buffer, bytes, IBV_ACCESS_LOCAL_WRITE);
  if (send_mr == nullptr) {
    cleanup();
    return fail(CUTEAFD_STATUS_RDMA_UNAVAILABLE, "ibv_reg_mr sender failed");
  }
  recv_mr = ibv_reg_mr(pd, recv_buffer, bytes, IBV_ACCESS_LOCAL_WRITE);
  if (recv_mr == nullptr) {
    cleanup();
    return fail(CUTEAFD_STATUS_RDMA_UNAVAILABLE, "ibv_reg_mr receiver failed");
  }

  constexpr uint32_t sender_psn = 0x111111;
  constexpr uint32_t receiver_psn = 0x222222;
  cuteafd_status_t status = modify_rc_qp_to_init(sender_qp, port_num);
  if (status != CUTEAFD_STATUS_OK) {
    cleanup();
    return status;
  }
  status = modify_rc_qp_to_init(receiver_qp, port_num);
  if (status != CUTEAFD_STATUS_OK) {
    cleanup();
    return status;
  }
  status = modify_rc_qp_to_rtr(context, sender_qp, port_attr, port_num, receiver_qp->qp_num,
                               receiver_psn, port_attr.lid, nullptr, 0);
  if (status != CUTEAFD_STATUS_OK) {
    cleanup();
    return status;
  }
  status = modify_rc_qp_to_rtr(context, receiver_qp, port_attr, port_num, sender_qp->qp_num,
                               sender_psn, port_attr.lid, nullptr, 0);
  if (status != CUTEAFD_STATUS_OK) {
    cleanup();
    return status;
  }
  status = modify_rc_qp_to_rts(sender_qp, sender_psn);
  if (status != CUTEAFD_STATUS_OK) {
    cleanup();
    return status;
  }
  status = modify_rc_qp_to_rts(receiver_qp, receiver_psn);
  if (status != CUTEAFD_STATUS_OK) {
    cleanup();
    return status;
  }

  ibv_sge recv_sge = {};
  recv_sge.addr = reinterpret_cast<uintptr_t>(recv_buffer);
  recv_sge.length = static_cast<uint32_t>(bytes);
  recv_sge.lkey = recv_mr->lkey;
  ibv_recv_wr recv_wr = {};
  recv_wr.wr_id = 1;
  recv_wr.sg_list = &recv_sge;
  recv_wr.num_sge = 1;
  ibv_recv_wr* bad_recv = nullptr;
  if (ibv_post_recv(receiver_qp, &recv_wr, &bad_recv) != 0) {
    cleanup();
    return fail(CUTEAFD_STATUS_RDMA_UNAVAILABLE, "ibv_post_recv failed for RC loopback probe");
  }

  ibv_sge send_sge = {};
  send_sge.addr = reinterpret_cast<uintptr_t>(send_buffer);
  send_sge.length = static_cast<uint32_t>(bytes);
  send_sge.lkey = send_mr->lkey;
  ibv_send_wr send_wr = {};
  send_wr.wr_id = 2;
  send_wr.sg_list = &send_sge;
  send_wr.num_sge = 1;
  send_wr.opcode = IBV_WR_SEND;
  send_wr.send_flags = IBV_SEND_SIGNALED;
  ibv_send_wr* bad_send = nullptr;
  if (ibv_post_send(sender_qp, &send_wr, &bad_send) != 0) {
    cleanup();
    return fail(CUTEAFD_STATUS_RDMA_UNAVAILABLE, "ibv_post_send failed for RC loopback probe");
  }

  uint32_t completions = 0;
  constexpr uint32_t max_poll_iterations = 1'000'000;
  for (uint32_t iteration = 0; iteration < max_poll_iterations && completions < 2; ++iteration) {
    out->poll_iterations = iteration + 1;
    ibv_wc wc = {};
    const int polled = ibv_poll_cq(cq, 1, &wc);
    if (polled < 0) {
      cleanup();
      return fail(CUTEAFD_STATUS_INTERNAL_ERROR, "ibv_poll_cq failed for RC loopback probe");
    }
    if (polled == 0) {
      continue;
    }
    if (wc.status != IBV_WC_SUCCESS) {
      cleanup();
      return fail(CUTEAFD_STATUS_RDMA_UNAVAILABLE,
                  "RC loopback probe completion returned non-success status");
    }
    if (wc.wr_id == 1) {
      out->recv_completions += 1;
    } else if (wc.wr_id == 2) {
      out->send_completions += 1;
    }
    completions += 1;
  }
  if (completions != 2 || out->send_completions != 1 || out->recv_completions != 1) {
    cleanup();
    return fail(CUTEAFD_STATUS_RDMA_UNAVAILABLE,
                "RC loopback probe timed out waiting for send/recv completions");
  }

  out->payload_matches = std::memcmp(send_buffer, recv_buffer, bytes) == 0 ? 1 : 0;
  if (!out->payload_matches) {
    cleanup();
    return fail(CUTEAFD_STATUS_INTERNAL_ERROR, "RC loopback probe payload mismatch");
  }
  out->completed = 1;
  set_fixed_string(out->status, sizeof(out->status), "RC send/recv loopback completed");
  cleanup();
  return ok();
#else
  set_fixed_string(out->device_name, sizeof(out->device_name), "rdma-disabled-build");
  set_fixed_string(out->status, sizeof(out->status),
                   "RC send/recv loopback requires CUTEAFD_ENABLE_RDMA=ON");
  return fail(CUTEAFD_STATUS_RDMA_UNAVAILABLE,
              "RDMA RC send/recv loopback probe requires CUTEAFD_ENABLE_RDMA=ON");
#endif
}

extern "C" cuteafd_status_t cuteafd_rdma_rc_protocol_v2_loopback_probe(
    uint32_t port_num, const void* request_frame, size_t request_bytes,
    const void* response_frame, size_t response_bytes,
    cuteafd_rdma_rc_protocol_v2_loopback_probe_t* out) {
  if (out == nullptr) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT,
                "RDMA RC ProtocolV2 loopback probe output pointer is null");
  }
  if (port_num == 0 || port_num > 255) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT,
                "RDMA RC ProtocolV2 loopback probe port number is invalid");
  }
  if (request_bytes == 0 || response_bytes == 0) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT,
                "RDMA RC ProtocolV2 loopback probe frame byte size is zero");
  }
  if (request_bytes > std::numeric_limits<uint32_t>::max() ||
      response_bytes > std::numeric_limits<uint32_t>::max()) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT,
                "RDMA RC ProtocolV2 loopback probe frame byte size exceeds u32");
  }
  std::memset(out, 0, sizeof(*out));
  out->rdma_enabled = CUTEAFD_NATIVE_ENABLE_RDMA ? 1 : 0;
  out->port_num = port_num;
  out->request_bytes = request_bytes;
  out->response_bytes = response_bytes;

  cuteafd_status_t status = validate_protocol_v2_frame(request_frame, request_bytes, 1, "request");
  if (status != CUTEAFD_STATUS_OK) {
    return status;
  }
  status = validate_protocol_v2_frame(response_frame, response_bytes, 2, "response");
  if (status != CUTEAFD_STATUS_OK) {
    return status;
  }

#if CUTEAFD_NATIVE_ENABLE_RDMA
  int device_count = 0;
  ibv_device** devices = ibv_get_device_list(&device_count);
  if (devices == nullptr) {
    return fail(CUTEAFD_STATUS_RDMA_UNAVAILABLE, "ibv_get_device_list failed");
  }
  if (device_count <= 0) {
    ibv_free_device_list(devices);
    return fail(CUTEAFD_STATUS_RDMA_UNAVAILABLE,
                "no RDMA devices available for RC ProtocolV2 loopback probe");
  }

  ibv_context* context = nullptr;
  ibv_pd* pd = nullptr;
  ibv_cq* cq = nullptr;
  ibv_qp* client_qp = nullptr;
  ibv_qp* server_qp = nullptr;
  ibv_mr* client_request_send_mr = nullptr;
  ibv_mr* server_request_recv_mr = nullptr;
  ibv_mr* server_response_send_mr = nullptr;
  ibv_mr* client_response_recv_mr = nullptr;
  unsigned char* client_request_send_buffer = nullptr;
  unsigned char* server_request_recv_buffer = nullptr;
  unsigned char* server_response_send_buffer = nullptr;
  unsigned char* client_response_recv_buffer = nullptr;

  auto cleanup = [&]() {
    if (server_qp != nullptr) {
      ibv_destroy_qp(server_qp);
    }
    if (client_qp != nullptr) {
      ibv_destroy_qp(client_qp);
    }
    if (client_response_recv_mr != nullptr) {
      ibv_dereg_mr(client_response_recv_mr);
    }
    if (server_response_send_mr != nullptr) {
      ibv_dereg_mr(server_response_send_mr);
    }
    if (server_request_recv_mr != nullptr) {
      ibv_dereg_mr(server_request_recv_mr);
    }
    if (client_request_send_mr != nullptr) {
      ibv_dereg_mr(client_request_send_mr);
    }
    if (cq != nullptr) {
      ibv_destroy_cq(cq);
    }
    if (pd != nullptr) {
      ibv_dealloc_pd(pd);
    }
    if (context != nullptr) {
      ibv_close_device(context);
    }
    if (devices != nullptr) {
      ibv_free_device_list(devices);
    }
    std::free(client_request_send_buffer);
    std::free(server_request_recv_buffer);
    std::free(server_response_send_buffer);
    std::free(client_response_recv_buffer);
  };

  ibv_device* device = devices[0];
  const char* device_name = ibv_get_device_name(device);
  set_fixed_string(out->device_name, sizeof(out->device_name),
                   device_name != nullptr ? device_name : "unknown-rdma-device");

  context = ibv_open_device(device);
  if (context == nullptr) {
    cleanup();
    return fail(CUTEAFD_STATUS_RDMA_UNAVAILABLE,
                "ibv_open_device failed for RC ProtocolV2 loopback probe");
  }

  ibv_port_attr port_attr = {};
  if (ibv_query_port(context, static_cast<uint8_t>(port_num), &port_attr) != 0) {
    cleanup();
    return fail(CUTEAFD_STATUS_RDMA_UNAVAILABLE,
                "ibv_query_port failed for RC ProtocolV2 loopback probe");
  }

  pd = ibv_alloc_pd(context);
  if (pd == nullptr) {
    cleanup();
    return fail(CUTEAFD_STATUS_RDMA_UNAVAILABLE,
                "ibv_alloc_pd failed for RC ProtocolV2 loopback probe");
  }
  cq = ibv_create_cq(context, 8, nullptr, nullptr, 0);
  if (cq == nullptr) {
    cleanup();
    return fail(CUTEAFD_STATUS_RDMA_UNAVAILABLE,
                "ibv_create_cq failed for RC ProtocolV2 loopback probe");
  }

  ibv_qp_init_attr qp_attr = {};
  qp_attr.send_cq = cq;
  qp_attr.recv_cq = cq;
  qp_attr.qp_type = IBV_QPT_RC;
  qp_attr.cap.max_send_wr = 4;
  qp_attr.cap.max_recv_wr = 4;
  qp_attr.cap.max_send_sge = 1;
  qp_attr.cap.max_recv_sge = 1;

  client_qp = ibv_create_qp(pd, &qp_attr);
  if (client_qp == nullptr) {
    cleanup();
    return fail(CUTEAFD_STATUS_RDMA_UNAVAILABLE,
                "ibv_create_qp client failed for ProtocolV2 loopback probe");
  }
  server_qp = ibv_create_qp(pd, &qp_attr);
  if (server_qp == nullptr) {
    cleanup();
    return fail(CUTEAFD_STATUS_RDMA_UNAVAILABLE,
                "ibv_create_qp server failed for ProtocolV2 loopback probe");
  }
  out->client_qp_num = client_qp->qp_num;
  out->server_qp_num = server_qp->qp_num;

  client_request_send_buffer = static_cast<unsigned char*>(std::malloc(request_bytes));
  server_request_recv_buffer = static_cast<unsigned char*>(std::malloc(request_bytes));
  server_response_send_buffer = static_cast<unsigned char*>(std::malloc(response_bytes));
  client_response_recv_buffer = static_cast<unsigned char*>(std::malloc(response_bytes));
  if (client_request_send_buffer == nullptr || server_request_recv_buffer == nullptr ||
      server_response_send_buffer == nullptr || client_response_recv_buffer == nullptr) {
    cleanup();
    return fail(CUTEAFD_STATUS_ALLOCATION_FAILED,
                "allocating RC ProtocolV2 loopback host buffers failed");
  }
  std::memcpy(client_request_send_buffer, request_frame, request_bytes);
  std::memset(server_request_recv_buffer, 0, request_bytes);
  std::memcpy(server_response_send_buffer, response_frame, response_bytes);
  std::memset(client_response_recv_buffer, 0, response_bytes);

  client_request_send_mr =
      ibv_reg_mr(pd, client_request_send_buffer, request_bytes, IBV_ACCESS_LOCAL_WRITE);
  if (client_request_send_mr == nullptr) {
    cleanup();
    return fail(CUTEAFD_STATUS_RDMA_UNAVAILABLE,
                "ibv_reg_mr client request send failed for ProtocolV2 loopback probe");
  }
  server_request_recv_mr =
      ibv_reg_mr(pd, server_request_recv_buffer, request_bytes, IBV_ACCESS_LOCAL_WRITE);
  if (server_request_recv_mr == nullptr) {
    cleanup();
    return fail(CUTEAFD_STATUS_RDMA_UNAVAILABLE,
                "ibv_reg_mr server request recv failed for ProtocolV2 loopback probe");
  }
  server_response_send_mr =
      ibv_reg_mr(pd, server_response_send_buffer, response_bytes, IBV_ACCESS_LOCAL_WRITE);
  if (server_response_send_mr == nullptr) {
    cleanup();
    return fail(CUTEAFD_STATUS_RDMA_UNAVAILABLE,
                "ibv_reg_mr server response send failed for ProtocolV2 loopback probe");
  }
  client_response_recv_mr =
      ibv_reg_mr(pd, client_response_recv_buffer, response_bytes, IBV_ACCESS_LOCAL_WRITE);
  if (client_response_recv_mr == nullptr) {
    cleanup();
    return fail(CUTEAFD_STATUS_RDMA_UNAVAILABLE,
                "ibv_reg_mr client response recv failed for ProtocolV2 loopback probe");
  }

  constexpr uint32_t client_psn = 0x313131;
  constexpr uint32_t server_psn = 0x414141;
  status = modify_rc_qp_to_init(client_qp, port_num);
  if (status != CUTEAFD_STATUS_OK) {
    cleanup();
    return status;
  }
  status = modify_rc_qp_to_init(server_qp, port_num);
  if (status != CUTEAFD_STATUS_OK) {
    cleanup();
    return status;
  }
  status = modify_rc_qp_to_rtr(context, client_qp, port_attr, port_num, server_qp->qp_num,
                               server_psn, port_attr.lid, nullptr, 0);
  if (status != CUTEAFD_STATUS_OK) {
    cleanup();
    return status;
  }
  status = modify_rc_qp_to_rtr(context, server_qp, port_attr, port_num, client_qp->qp_num,
                               client_psn, port_attr.lid, nullptr, 0);
  if (status != CUTEAFD_STATUS_OK) {
    cleanup();
    return status;
  }
  status = modify_rc_qp_to_rts(client_qp, client_psn);
  if (status != CUTEAFD_STATUS_OK) {
    cleanup();
    return status;
  }
  status = modify_rc_qp_to_rts(server_qp, server_psn);
  if (status != CUTEAFD_STATUS_OK) {
    cleanup();
    return status;
  }

  ibv_sge server_request_recv_sge = {};
  server_request_recv_sge.addr = reinterpret_cast<uintptr_t>(server_request_recv_buffer);
  server_request_recv_sge.length = static_cast<uint32_t>(request_bytes);
  server_request_recv_sge.lkey = server_request_recv_mr->lkey;
  ibv_recv_wr server_request_recv_wr = {};
  server_request_recv_wr.wr_id = 1;
  server_request_recv_wr.sg_list = &server_request_recv_sge;
  server_request_recv_wr.num_sge = 1;
  ibv_recv_wr* bad_recv = nullptr;
  if (ibv_post_recv(server_qp, &server_request_recv_wr, &bad_recv) != 0) {
    cleanup();
    return fail(CUTEAFD_STATUS_RDMA_UNAVAILABLE,
                "ibv_post_recv server request failed for ProtocolV2 loopback probe");
  }

  ibv_sge client_response_recv_sge = {};
  client_response_recv_sge.addr = reinterpret_cast<uintptr_t>(client_response_recv_buffer);
  client_response_recv_sge.length = static_cast<uint32_t>(response_bytes);
  client_response_recv_sge.lkey = client_response_recv_mr->lkey;
  ibv_recv_wr client_response_recv_wr = {};
  client_response_recv_wr.wr_id = 2;
  client_response_recv_wr.sg_list = &client_response_recv_sge;
  client_response_recv_wr.num_sge = 1;
  if (ibv_post_recv(client_qp, &client_response_recv_wr, &bad_recv) != 0) {
    cleanup();
    return fail(CUTEAFD_STATUS_RDMA_UNAVAILABLE,
                "ibv_post_recv client response failed for ProtocolV2 loopback probe");
  }

  ibv_sge client_request_send_sge = {};
  client_request_send_sge.addr = reinterpret_cast<uintptr_t>(client_request_send_buffer);
  client_request_send_sge.length = static_cast<uint32_t>(request_bytes);
  client_request_send_sge.lkey = client_request_send_mr->lkey;
  ibv_send_wr client_request_send_wr = {};
  client_request_send_wr.wr_id = 3;
  client_request_send_wr.sg_list = &client_request_send_sge;
  client_request_send_wr.num_sge = 1;
  client_request_send_wr.opcode = IBV_WR_SEND;
  client_request_send_wr.send_flags = IBV_SEND_SIGNALED;
  ibv_send_wr* bad_send = nullptr;
  if (ibv_post_send(client_qp, &client_request_send_wr, &bad_send) != 0) {
    cleanup();
    return fail(CUTEAFD_STATUS_RDMA_UNAVAILABLE,
                "ibv_post_send client request failed for ProtocolV2 loopback probe");
  }

  ibv_sge server_response_send_sge = {};
  server_response_send_sge.addr = reinterpret_cast<uintptr_t>(server_response_send_buffer);
  server_response_send_sge.length = static_cast<uint32_t>(response_bytes);
  server_response_send_sge.lkey = server_response_send_mr->lkey;
  ibv_send_wr server_response_send_wr = {};
  server_response_send_wr.wr_id = 4;
  server_response_send_wr.sg_list = &server_response_send_sge;
  server_response_send_wr.num_sge = 1;
  server_response_send_wr.opcode = IBV_WR_SEND;
  server_response_send_wr.send_flags = IBV_SEND_SIGNALED;
  if (ibv_post_send(server_qp, &server_response_send_wr, &bad_send) != 0) {
    cleanup();
    return fail(CUTEAFD_STATUS_RDMA_UNAVAILABLE,
                "ibv_post_send server response failed for ProtocolV2 loopback probe");
  }

  uint32_t completions = 0;
  constexpr uint32_t max_poll_iterations = 1'000'000;
  for (uint32_t iteration = 0; iteration < max_poll_iterations && completions < 4; ++iteration) {
    out->poll_iterations = iteration + 1;
    ibv_wc wc = {};
    const int polled = ibv_poll_cq(cq, 1, &wc);
    if (polled < 0) {
      cleanup();
      return fail(CUTEAFD_STATUS_INTERNAL_ERROR,
                  "ibv_poll_cq failed for RC ProtocolV2 loopback probe");
    }
    if (polled == 0) {
      continue;
    }
    if (wc.status != IBV_WC_SUCCESS) {
      cleanup();
      return fail(CUTEAFD_STATUS_RDMA_UNAVAILABLE,
                  "RC ProtocolV2 loopback completion returned non-success status");
    }
    if (wc.wr_id == 1 || wc.wr_id == 2) {
      out->recv_completions += 1;
    } else if (wc.wr_id == 3 || wc.wr_id == 4) {
      out->send_completions += 1;
    }
    completions += 1;
  }
  if (completions != 4 || out->send_completions != 2 || out->recv_completions != 2) {
    cleanup();
    return fail(CUTEAFD_STATUS_RDMA_UNAVAILABLE,
                "RC ProtocolV2 loopback probe timed out waiting for completions");
  }

  out->request_payload_matches =
      std::memcmp(request_frame, server_request_recv_buffer, request_bytes) == 0 ? 1 : 0;
  out->response_payload_matches =
      std::memcmp(response_frame, client_response_recv_buffer, response_bytes) == 0 ? 1 : 0;
  if (!out->request_payload_matches || !out->response_payload_matches) {
    cleanup();
    return fail(CUTEAFD_STATUS_INTERNAL_ERROR, "RC ProtocolV2 loopback payload mismatch");
  }
  out->completed = 1;
  set_fixed_string(out->status, sizeof(out->status),
                   "RC ProtocolV2 request/response loopback completed");
  cleanup();
  return ok();
#else
  set_fixed_string(out->device_name, sizeof(out->device_name), "rdma-disabled-build");
  set_fixed_string(out->status, sizeof(out->status),
                   "RC ProtocolV2 loopback requires CUTEAFD_ENABLE_RDMA=ON");
  return fail(CUTEAFD_STATUS_RDMA_UNAVAILABLE,
              "RDMA RC ProtocolV2 loopback probe requires CUTEAFD_ENABLE_RDMA=ON");
#endif
}

extern "C" cuteafd_status_t cuteafd_rdma_rc_endpoint_create(
    uint32_t port_num, uint32_t local_psn, size_t send_frame_bytes, size_t recv_frame_bytes,
    size_t send_registered_span_bytes, size_t recv_registered_span_bytes, uint32_t max_send_wr,
    uint32_t max_recv_wr, uint32_t max_sge, cuteafd_rdma_rc_endpoint_info_t* out) {
  return cuteafd_rdma_rc_endpoint_create_with_buffer_flags(
      port_num, local_psn, send_frame_bytes, recv_frame_bytes,
      send_registered_span_bytes, recv_registered_span_bytes, max_send_wr, max_recv_wr,
      max_sge, CUTEAFD_HOST_BUFFER_FLAG_NONE, out);
}

static cuteafd_status_t create_rdma_rc_endpoint_with_buffer_flags(
    const char* requested_device_name, uint32_t port_num, uint32_t local_psn,
    int requested_gid_index,
    size_t send_frame_bytes, size_t recv_frame_bytes, size_t send_registered_span_bytes,
    size_t recv_registered_span_bytes, uint32_t max_send_wr, uint32_t max_recv_wr,
    uint32_t max_sge, uint64_t host_buffer_flags, cuteafd_rdma_rc_endpoint_info_t* out) {
  if (out == nullptr) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "RDMA RC endpoint output pointer is null");
  }
  if (port_num == 0 || port_num > 255) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "RDMA RC endpoint port number is invalid");
  }
  if (local_psn > 0x00ff'ffff) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "RDMA RC endpoint PSN exceeds 24 bits");
  }
  if (send_frame_bytes == 0 || recv_frame_bytes == 0 || send_registered_span_bytes == 0 ||
      recv_registered_span_bytes == 0) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "RDMA RC endpoint frame/span byte size is zero");
  }
  if (send_frame_bytes > send_registered_span_bytes ||
      recv_frame_bytes > recv_registered_span_bytes) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT,
                "RDMA RC endpoint frame bytes exceed registered span bytes");
  }
  if (send_registered_span_bytes > std::numeric_limits<uint32_t>::max() ||
      recv_registered_span_bytes > std::numeric_limits<uint32_t>::max()) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT,
                "RDMA RC endpoint registered span byte size exceeds u32");
  }
  if (max_send_wr == 0 || max_recv_wr == 0 || max_sge == 0) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT,
                "RDMA RC endpoint work-request capacities must be non-zero");
  }
  constexpr uint64_t kMappedHostFlags =
      CUTEAFD_HOST_BUFFER_FLAG_PINNED | CUTEAFD_HOST_BUFFER_FLAG_MAPPED;
  if (host_buffer_flags != CUTEAFD_HOST_BUFFER_FLAG_NONE &&
      host_buffer_flags != kMappedHostFlags) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT,
                "RDMA RC endpoint host buffers must be either ordinary or pinned and mapped");
  }
#if !CUTEAFD_NATIVE_ENABLE_CUDA
  if (host_buffer_flags != CUTEAFD_HOST_BUFFER_FLAG_NONE) {
    return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
                "CUDA-mapped RDMA RC endpoint buffers require a CUDA build");
  }
#endif
  std::memset(out, 0, sizeof(*out));
  out->rdma_enabled = CUTEAFD_NATIVE_ENABLE_RDMA ? 1 : 0;
  out->port_num = port_num;
  out->psn = local_psn;
  out->send_frame_bytes = send_frame_bytes;
  out->recv_frame_bytes = recv_frame_bytes;
  out->send_registered_span_bytes = send_registered_span_bytes;
  out->recv_registered_span_bytes = recv_registered_span_bytes;
  out->max_send_wr = max_send_wr;
  out->max_recv_wr = max_recv_wr;
  out->max_sge = max_sge;

#if CUTEAFD_NATIVE_ENABLE_RDMA
  int device_count = 0;
  ibv_device** devices = ibv_get_device_list(&device_count);
  if (devices == nullptr) {
    return fail(CUTEAFD_STATUS_RDMA_UNAVAILABLE, "ibv_get_device_list failed");
  }
  if (device_count <= 0) {
    ibv_free_device_list(devices);
    return fail(CUTEAFD_STATUS_RDMA_UNAVAILABLE, "no RDMA devices available for RC endpoint");
  }

  CuteafdRdmaRcEndpointHandle* endpoint = new (std::nothrow) CuteafdRdmaRcEndpointHandle();
  if (endpoint == nullptr) {
    ibv_free_device_list(devices);
    return fail(CUTEAFD_STATUS_ALLOCATION_FAILED, "allocating RDMA RC endpoint handle failed");
  }
  endpoint->port_num = port_num;
  endpoint->psn = local_psn;
  endpoint->send_frame_bytes = send_frame_bytes;
  endpoint->recv_frame_bytes = recv_frame_bytes;
  endpoint->send_registered_span_bytes = send_registered_span_bytes;
  endpoint->recv_registered_span_bytes = recv_registered_span_bytes;
  endpoint->host_buffer_flags = host_buffer_flags;

  ibv_device* device = nullptr;
  if (requested_device_name == nullptr || requested_device_name[0] == '\0') {
    device = devices[0];
  } else {
    for (int index = 0; index < device_count; ++index) {
      const char* candidate_name = ibv_get_device_name(devices[index]);
      if (candidate_name != nullptr && std::strcmp(candidate_name, requested_device_name) == 0) {
        device = devices[index];
        break;
      }
    }
    if (device == nullptr) {
      char message[256];
      std::snprintf(message, sizeof(message), "RDMA RC endpoint device %s was not found",
                    requested_device_name);
      delete endpoint;
      ibv_free_device_list(devices);
      return fail(CUTEAFD_STATUS_RDMA_UNAVAILABLE, message);
    }
  }
  const char* device_name = ibv_get_device_name(device);
  set_fixed_string(out->device_name, sizeof(out->device_name),
                   device_name != nullptr ? device_name : "unknown-rdma-device");

  endpoint->context = ibv_open_device(device);
  ibv_free_device_list(devices);
  if (endpoint->context == nullptr) {
    destroy_rdma_rc_endpoint(endpoint);
    return fail(CUTEAFD_STATUS_RDMA_UNAVAILABLE, "ibv_open_device failed for RC endpoint");
  }

  if (ibv_query_port(endpoint->context, static_cast<uint8_t>(port_num), &endpoint->port_attr) !=
      0) {
    destroy_rdma_rc_endpoint(endpoint);
    return fail(CUTEAFD_STATUS_RDMA_UNAVAILABLE, "ibv_query_port failed for RC endpoint");
  }
  endpoint->lid = endpoint->port_attr.lid;
  out->lid = endpoint->lid;
  out->active_mtu = endpoint->port_attr.active_mtu;

  ibv_gid local_gid = {};
  uint32_t local_gid_index = 0;
  cuteafd_status_t status =
      requested_gid_index < 0
          ? select_rc_gid(endpoint->context, endpoint->port_attr, port_num, &local_gid,
                          &local_gid_index)
          : (ibv_query_gid(endpoint->context, static_cast<uint8_t>(port_num),
                           requested_gid_index, &local_gid) == 0
                 ? CUTEAFD_STATUS_OK
                 : fail(CUTEAFD_STATUS_RDMA_UNAVAILABLE, "selected RoCE GID is unavailable"));
  if (requested_gid_index >= 0) local_gid_index = static_cast<uint32_t>(requested_gid_index);
  if (status != CUTEAFD_STATUS_OK) {
    destroy_rdma_rc_endpoint(endpoint);
    return status;
  }
  endpoint->gid_index = local_gid_index;
  gid_to_hex(local_gid, out->gid_hex, sizeof(out->gid_hex));

  endpoint->pd = ibv_alloc_pd(endpoint->context);
  if (endpoint->pd == nullptr) {
    destroy_rdma_rc_endpoint(endpoint);
    return fail(CUTEAFD_STATUS_RDMA_UNAVAILABLE, "ibv_alloc_pd failed for RC endpoint");
  }

  endpoint->send_channel = ibv_create_comp_channel(endpoint->context);
  if (endpoint->send_channel == nullptr) {
    destroy_rdma_rc_endpoint(endpoint);
    return fail(CUTEAFD_STATUS_RDMA_UNAVAILABLE,
                "ibv_create_comp_channel send failed for RC endpoint");
  }
  endpoint->recv_channel = ibv_create_comp_channel(endpoint->context);
  if (endpoint->recv_channel == nullptr) {
    destroy_rdma_rc_endpoint(endpoint);
    return fail(CUTEAFD_STATUS_RDMA_UNAVAILABLE,
                "ibv_create_comp_channel recv failed for RC endpoint");
  }

  const int send_cq_depth = static_cast<int>(std::max<uint32_t>(max_send_wr, 4));
  endpoint->send_cq =
      ibv_create_cq(endpoint->context, send_cq_depth, nullptr, endpoint->send_channel, 0);
  if (endpoint->send_cq == nullptr) {
    destroy_rdma_rc_endpoint(endpoint);
    return fail(CUTEAFD_STATUS_RDMA_UNAVAILABLE, "ibv_create_cq send failed for RC endpoint");
  }
  const int recv_cq_depth = static_cast<int>(std::max<uint32_t>(max_recv_wr, 4));
  endpoint->recv_cq =
      ibv_create_cq(endpoint->context, recv_cq_depth, nullptr, endpoint->recv_channel, 0);
  if (endpoint->recv_cq == nullptr) {
    destroy_rdma_rc_endpoint(endpoint);
    return fail(CUTEAFD_STATUS_RDMA_UNAVAILABLE, "ibv_create_cq recv failed for RC endpoint");
  }

  ibv_qp_init_attr qp_attr = {};
  qp_attr.send_cq = endpoint->send_cq;
  qp_attr.recv_cq = endpoint->recv_cq;
  qp_attr.qp_type = IBV_QPT_RC;
  qp_attr.cap.max_send_wr = max_send_wr;
  qp_attr.cap.max_recv_wr = max_recv_wr;
  qp_attr.cap.max_send_sge = max_sge;
  qp_attr.cap.max_recv_sge = max_sge;
  qp_attr.cap.max_inline_data = 0;

  endpoint->qp = ibv_create_qp(endpoint->pd, &qp_attr);
  if (endpoint->qp == nullptr) {
    destroy_rdma_rc_endpoint(endpoint);
    return fail(CUTEAFD_STATUS_RDMA_UNAVAILABLE, "ibv_create_qp failed for RC endpoint");
  }
  out->qp_num = endpoint->qp->qp_num;
  out->max_send_wr = qp_attr.cap.max_send_wr;
  out->max_recv_wr = qp_attr.cap.max_recv_wr;
  out->max_sge = std::min(qp_attr.cap.max_send_sge, qp_attr.cap.max_recv_sge);

  if (host_buffer_flags == kMappedHostFlags) {
#if CUTEAFD_NATIVE_ENABLE_CUDA
    void* send_buffer = nullptr;
    cudaError_t cuda_status = cudaHostAlloc(
        &send_buffer, send_registered_span_bytes, cudaHostAllocPortable | cudaHostAllocMapped);
    if (cuda_status != cudaSuccess) {
      destroy_rdma_rc_endpoint(endpoint);
      return fail_cuda(CUTEAFD_STATUS_ALLOCATION_FAILED,
                       "cudaHostAlloc RDMA RC endpoint send buffer failed", cuda_status);
    }
    endpoint->send_buffer = static_cast<unsigned char*>(send_buffer);
    void* recv_buffer = nullptr;
    cuda_status = cudaHostAlloc(
        &recv_buffer, recv_registered_span_bytes, cudaHostAllocPortable | cudaHostAllocMapped);
    if (cuda_status != cudaSuccess) {
      destroy_rdma_rc_endpoint(endpoint);
      return fail_cuda(CUTEAFD_STATUS_ALLOCATION_FAILED,
                       "cudaHostAlloc RDMA RC endpoint recv buffer failed", cuda_status);
    }
    endpoint->recv_buffer = static_cast<unsigned char*>(recv_buffer);
#endif
  } else {
    endpoint->send_buffer = static_cast<unsigned char*>(std::malloc(send_registered_span_bytes));
    endpoint->recv_buffer = static_cast<unsigned char*>(std::malloc(recv_registered_span_bytes));
  }
  if (endpoint->send_buffer == nullptr || endpoint->recv_buffer == nullptr) {
    destroy_rdma_rc_endpoint(endpoint);
    return fail(CUTEAFD_STATUS_ALLOCATION_FAILED,
                "allocating RDMA RC endpoint registered buffers failed");
  }
  std::memset(endpoint->send_buffer, 0, send_registered_span_bytes);
  std::memset(endpoint->recv_buffer, 0, recv_registered_span_bytes);

  endpoint->send_mr =
      ibv_reg_mr(endpoint->pd, endpoint->send_buffer, send_registered_span_bytes,
                 IBV_ACCESS_LOCAL_WRITE);
  if (endpoint->send_mr == nullptr) {
    destroy_rdma_rc_endpoint(endpoint);
    return fail(CUTEAFD_STATUS_RDMA_UNAVAILABLE, "ibv_reg_mr send buffer failed for RC endpoint");
  }
  endpoint->recv_mr =
      ibv_reg_mr(endpoint->pd, endpoint->recv_buffer, recv_registered_span_bytes,
                 IBV_ACCESS_LOCAL_WRITE);
  if (endpoint->recv_mr == nullptr) {
    destroy_rdma_rc_endpoint(endpoint);
    return fail(CUTEAFD_STATUS_RDMA_UNAVAILABLE, "ibv_reg_mr recv buffer failed for RC endpoint");
  }

  status = modify_rc_qp_to_init(endpoint->qp, port_num);
  if (status != CUTEAFD_STATUS_OK) {
    destroy_rdma_rc_endpoint(endpoint);
    return status;
  }

  out->handle = endpoint;
  char status_message[128];
  std::snprintf(status_message, sizeof(status_message), "RDMA RC endpoint created gid_index=%u",
                static_cast<unsigned>(endpoint->gid_index));
  set_fixed_string(out->status, sizeof(out->status), status_message);
  return ok();
#else
  set_fixed_string(out->device_name, sizeof(out->device_name), "rdma-disabled-build");
  set_fixed_string(out->status, sizeof(out->status),
                   "RDMA RC endpoint requires CUTEAFD_ENABLE_RDMA=ON");
  return fail(CUTEAFD_STATUS_RDMA_UNAVAILABLE,
              "RDMA RC endpoint requires CUTEAFD_ENABLE_RDMA=ON");
#endif
}

extern "C" cuteafd_status_t cuteafd_rdma_rc_endpoint_create_with_buffer_flags(
    uint32_t port_num, uint32_t local_psn, size_t send_frame_bytes, size_t recv_frame_bytes,
    size_t send_registered_span_bytes, size_t recv_registered_span_bytes, uint32_t max_send_wr,
    uint32_t max_recv_wr, uint32_t max_sge, uint64_t host_buffer_flags,
    cuteafd_rdma_rc_endpoint_info_t* out) {
  return create_rdma_rc_endpoint_with_buffer_flags(
      nullptr, port_num, local_psn, -1, send_frame_bytes, recv_frame_bytes,
      send_registered_span_bytes, recv_registered_span_bytes, max_send_wr, max_recv_wr, max_sge,
      host_buffer_flags, out);
}

extern "C" cuteafd_status_t cuteafd_rdma_rc_endpoint_create_on_device_with_buffer_flags(
    const char* device_name, uint32_t port_num, uint32_t local_psn, size_t send_frame_bytes,
    size_t recv_frame_bytes, size_t send_registered_span_bytes,
    size_t recv_registered_span_bytes, uint32_t max_send_wr, uint32_t max_recv_wr,
    uint32_t max_sge, uint64_t host_buffer_flags, cuteafd_rdma_rc_endpoint_info_t* out) {
  if (device_name == nullptr || device_name[0] == '\0') {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT,
                "RDMA RC endpoint requested device name is empty");
  }
  return create_rdma_rc_endpoint_with_buffer_flags(
      device_name, port_num, local_psn, -1, send_frame_bytes, recv_frame_bytes,
      send_registered_span_bytes, recv_registered_span_bytes, max_send_wr, max_recv_wr, max_sge,
      host_buffer_flags, out);
}

extern "C" cuteafd_status_t cuteafd_rdma_rc_endpoint_create_on_gid_with_buffer_flags(
    const char* device_name, uint32_t port_num, uint32_t gid_index, uint32_t local_psn,
    size_t send_frame_bytes, size_t recv_frame_bytes, size_t send_registered_span_bytes,
    size_t recv_registered_span_bytes, uint32_t max_send_wr, uint32_t max_recv_wr,
    uint32_t max_sge, uint64_t host_buffer_flags, cuteafd_rdma_rc_endpoint_info_t* out) {
  if (device_name == nullptr || device_name[0] == '\0' || gid_index > 255) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "invalid RDMA device or GID index");
  }
  return create_rdma_rc_endpoint_with_buffer_flags(
      device_name, port_num, local_psn, static_cast<int>(gid_index), send_frame_bytes,
      recv_frame_bytes, send_registered_span_bytes, recv_registered_span_bytes,
      max_send_wr, max_recv_wr, max_sge, host_buffer_flags, out);
}

extern "C" cuteafd_status_t cuteafd_rdma_rc_endpoint_buffer_view(
    void* handle, int receive_buffer, cuteafd_rdma_rc_endpoint_buffer_view_t* out) {
  if (handle == nullptr) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "RDMA RC endpoint handle is null");
  }
  if (out == nullptr) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "RDMA RC endpoint buffer view is null");
  }
  if (receive_buffer != 0 && receive_buffer != 1) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT,
                "RDMA RC endpoint buffer view selector must be send or receive");
  }
  std::memset(out, 0, sizeof(*out));
  out->device_id = -1;
#if CUTEAFD_NATIVE_ENABLE_RDMA
  auto* endpoint = static_cast<CuteafdRdmaRcEndpointHandle*>(handle);
  out->host_ptr = receive_buffer != 0 ? endpoint->recv_buffer : endpoint->send_buffer;
  out->bytes = receive_buffer != 0 ? endpoint->recv_registered_span_bytes
                                   : endpoint->send_registered_span_bytes;
  out->host_flags = endpoint->host_buffer_flags;
#if CUTEAFD_NATIVE_ENABLE_CUDA
  if ((endpoint->host_buffer_flags & CUTEAFD_HOST_BUFFER_FLAG_MAPPED) != 0) {
    cudaError_t cuda_status = cudaHostGetDevicePointer(&out->device_ptr, out->host_ptr, 0);
    if (cuda_status != cudaSuccess) {
      return fail_cuda(CUTEAFD_STATUS_INTERNAL_ERROR,
                       "cudaHostGetDevicePointer RDMA RC endpoint buffer failed", cuda_status);
    }
    cuda_status = cudaGetDevice(&out->device_id);
    if (cuda_status != cudaSuccess) {
      return fail_cuda(CUTEAFD_STATUS_INTERNAL_ERROR,
                       "cudaGetDevice RDMA RC endpoint buffer failed", cuda_status);
    }
  }
#endif
  return ok();
#else
  (void)receive_buffer;
  return fail(CUTEAFD_STATUS_RDMA_UNAVAILABLE,
              "RDMA RC endpoint buffer views require CUTEAFD_ENABLE_RDMA=ON");
#endif
}

static cuteafd_status_t connect_rdma_rc_endpoint(void* handle, uint32_t remote_qp_num,
                                                 uint32_t remote_psn, uint32_t remote_lid,
                                                 const char* remote_gid_hex,
                                                 uint32_t flow_label) {
  if (handle == nullptr) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "RDMA RC endpoint handle is null");
  }
  if (remote_qp_num == 0) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "RDMA RC endpoint remote QP number is zero");
  }
  if (remote_psn > 0x00ff'ffff) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "RDMA RC endpoint remote PSN exceeds 24 bits");
  }
  if (flow_label > 0x000f'ffff) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "RDMA RC endpoint flow label exceeds 20 bits");
  }
#if CUTEAFD_NATIVE_ENABLE_RDMA
  auto* endpoint = static_cast<CuteafdRdmaRcEndpointHandle*>(handle);
  ibv_gid remote_gid = {};
  cuteafd_status_t status = gid_from_hex(remote_gid_hex, &remote_gid);
  if (status != CUTEAFD_STATUS_OK) {
    return status;
  }
  status = modify_rc_qp_to_rtr(endpoint->context, endpoint->qp, endpoint->port_attr,
                               endpoint->port_num, remote_qp_num, remote_psn, remote_lid,
                               &remote_gid, endpoint->gid_index, flow_label);
  if (status != CUTEAFD_STATUS_OK) {
    return status;
  }
  return modify_rc_qp_to_rts(endpoint->qp, endpoint->psn);
#else
  (void)remote_qp_num;
  (void)remote_psn;
  (void)remote_lid;
  (void)remote_gid_hex;
  return fail(CUTEAFD_STATUS_RDMA_UNAVAILABLE,
              "RDMA RC endpoint connect requires CUTEAFD_ENABLE_RDMA=ON");
#endif
}

extern "C" cuteafd_status_t cuteafd_rdma_rc_endpoint_connect(void* handle, uint32_t remote_qp_num,
                                                          uint32_t remote_psn,
                                                          uint32_t remote_lid,
                                                          const char* remote_gid_hex) {
  return connect_rdma_rc_endpoint(handle, remote_qp_num, remote_psn, remote_lid, remote_gid_hex,
                                  0);
}

extern "C" cuteafd_status_t cuteafd_rdma_rc_endpoint_connect_flow_label(
    void* handle, uint32_t remote_qp_num, uint32_t remote_psn, uint32_t remote_lid,
    const char* remote_gid_hex, uint32_t flow_label) {
  return connect_rdma_rc_endpoint(handle, remote_qp_num, remote_psn, remote_lid, remote_gid_hex,
                                  flow_label);
}

extern "C" cuteafd_status_t cuteafd_rdma_rc_endpoint_expose_send_read(void* handle, size_t bytes,
                                                                   uint64_t* remote_addr,
                                                                   uint32_t* rkey) {
  if (handle == nullptr || remote_addr == nullptr || rkey == nullptr) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT,
                "RDMA RC endpoint handle or read exposure output is null");
  }
#if CUTEAFD_NATIVE_ENABLE_RDMA
  auto* endpoint = static_cast<CuteafdRdmaRcEndpointHandle*>(handle);
  if (bytes == 0 || bytes > endpoint->send_registered_span_bytes) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT,
                "RDMA RC endpoint read exposure exceeds the send buffer");
  }
  if (endpoint->read_mr != nullptr) {
    ibv_dereg_mr(endpoint->read_mr);
    endpoint->read_mr = nullptr;
  }
  endpoint->read_mr = ibv_reg_mr(endpoint->pd, endpoint->send_buffer, bytes,
                                 IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ);
  if (endpoint->read_mr == nullptr) {
    return fail(CUTEAFD_STATUS_RDMA_UNAVAILABLE,
                "ibv_reg_mr failed for the RC endpoint read exposure");
  }
  *remote_addr = reinterpret_cast<uintptr_t>(endpoint->send_buffer);
  *rkey = endpoint->read_mr->rkey;
  return ok();
#else
  (void)bytes;
  return fail(CUTEAFD_STATUS_RDMA_UNAVAILABLE,
              "RDMA RC endpoint read exposure requires CUTEAFD_ENABLE_RDMA=ON");
#endif
}

extern "C" cuteafd_status_t cuteafd_rdma_rc_endpoint_read_wait(void* handle, size_t offset_bytes,
                                                            size_t bytes, uint64_t remote_addr,
                                                            uint32_t rkey, uint32_t timeout_ms) {
  if (handle == nullptr) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "RDMA RC endpoint handle is null");
  }
  if (bytes == 0 || bytes > std::numeric_limits<uint32_t>::max() || remote_addr == 0) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "RDMA RC endpoint read is invalid");
  }
#if CUTEAFD_NATIVE_ENABLE_RDMA
  auto* endpoint = static_cast<CuteafdRdmaRcEndpointHandle*>(handle);
  if (offset_bytes > endpoint->recv_registered_span_bytes ||
      bytes > endpoint->recv_registered_span_bytes - offset_bytes) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT,
                "RDMA RC endpoint read exceeds the receive buffer");
  }
  constexpr uint64_t kReadWrId = 0x7256'5244;  // "RD"
  ibv_sge sge = {};
  sge.addr = reinterpret_cast<uintptr_t>(endpoint->recv_buffer + offset_bytes);
  sge.length = static_cast<uint32_t>(bytes);
  sge.lkey = endpoint->recv_mr->lkey;
  ibv_send_wr wr = {};
  wr.wr_id = kReadWrId;
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_RDMA_READ;
  wr.send_flags = IBV_SEND_SIGNALED;
  wr.wr.rdma.remote_addr = remote_addr;
  wr.wr.rdma.rkey = rkey;
  ibv_send_wr* bad = nullptr;
  if (ibv_post_send(endpoint->qp, &wr, &bad) != 0) {
    return fail(CUTEAFD_STATUS_RDMA_UNAVAILABLE, "ibv_post_send of an RC read failed");
  }
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(std::max<uint32_t>(timeout_ms, 1));
  for (;;) {
    ibv_wc wc = {};
    const int polled = ibv_poll_cq(endpoint->send_cq, 1, &wc);
    if (polled < 0) {
      return fail(CUTEAFD_STATUS_INTERNAL_ERROR, "ibv_poll_cq send failed for an RC read");
    }
    if (polled == 1) {
      if (wc.status != IBV_WC_SUCCESS) {
        char message[256];
        std::snprintf(message, sizeof(message),
                      "RDMA RC read completion returned non-success status status=%u (%s) "
                      "vendor_err=%u",
                      static_cast<unsigned>(wc.status), ibv_wc_status_str(wc.status),
                      static_cast<unsigned>(wc.vendor_err));
        return fail(CUTEAFD_STATUS_RDMA_UNAVAILABLE, message);
      }
      if (wc.opcode == IBV_WC_RDMA_READ && wc.wr_id == kReadWrId) {
        return ok();
      }
      // Not ours: leave it for the endpoint's ordinary send polling.
      endpoint->pending_send_completions += 1;
      continue;
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      return fail(CUTEAFD_STATUS_RDMA_UNAVAILABLE, "RDMA RC endpoint timed out waiting for a read");
    }
  }
#else
  (void)offset_bytes;
  (void)rkey;
  (void)timeout_ms;
  return fail(CUTEAFD_STATUS_RDMA_UNAVAILABLE,
              "RDMA RC endpoint read requires CUTEAFD_ENABLE_RDMA=ON");
#endif
}

extern "C" cuteafd_status_t cuteafd_rdma_rc_endpoint_post_recv_at(
    void* handle, size_t offset_bytes, size_t bytes, uint64_t wr_id) {
  if (handle == nullptr) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "RDMA RC endpoint handle is null");
  }
  if (bytes == 0 || bytes > std::numeric_limits<uint32_t>::max()) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "RDMA RC endpoint recv byte size is invalid");
  }
#if CUTEAFD_NATIVE_ENABLE_RDMA
  auto* endpoint = static_cast<CuteafdRdmaRcEndpointHandle*>(handle);
  if (bytes > endpoint->recv_frame_bytes) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT,
                "RDMA RC endpoint recv bytes exceed frame capacity");
  }
  if (offset_bytes > endpoint->recv_registered_span_bytes ||
      bytes > endpoint->recv_registered_span_bytes - offset_bytes) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT,
                "RDMA RC endpoint recv slot exceeds registered span");
  }
  ibv_sge recv_sge[2] = {};
  recv_sge[0].addr = reinterpret_cast<uintptr_t>(endpoint->recv_buffer + offset_bytes);
  recv_sge[0].length = static_cast<uint32_t>(bytes);
  recv_sge[0].lkey = endpoint->recv_mr->lkey;
  ibv_recv_wr recv_wr = {};
  recv_wr.wr_id = wr_id;
  recv_wr.sg_list = recv_sge;
  recv_wr.num_sge = 1;
  if (endpoint->landing_mr != nullptr) {
    // Header to the host slot, payload straight into device memory.
    recv_sge[0].length = static_cast<uint32_t>(std::min(bytes, endpoint->landing_header_bytes));
    recv_sge[1].addr = reinterpret_cast<uintptr_t>(endpoint->landing_ptr);
    recv_sge[1].length = static_cast<uint32_t>(endpoint->landing_bytes);
    recv_sge[1].lkey = endpoint->landing_mr->lkey;
    recv_wr.num_sge = 2;
  }
  ibv_recv_wr* bad_recv = nullptr;
  if (ibv_post_recv(endpoint->qp, &recv_wr, &bad_recv) != 0) {
    return fail(CUTEAFD_STATUS_RDMA_UNAVAILABLE, "ibv_post_recv failed for RC endpoint");
  }
  return ok();
#else
  (void)offset_bytes;
  (void)bytes;
  (void)wr_id;
  return fail(CUTEAFD_STATUS_RDMA_UNAVAILABLE,
              "RDMA RC endpoint recv requires CUTEAFD_ENABLE_RDMA=ON");
#endif
}

#if CUTEAFD_NATIVE_ENABLE_RDMA && CUTEAFD_NATIVE_ENABLE_CUDA
namespace {
// Registers device memory with the NIC through a dma-buf export of its whole
// allocation (no nvidia-peermem needed); the MR keeps the export alive.
cuteafd_status_t register_device_dmabuf_current(ibv_pd* pd, void* ptr, size_t bytes, ibv_mr** out,
                                                int access);

// The export needs the owning device's context current; sessions may connect
// on a thread (a prefill lane) that never selected it.
cuteafd_status_t register_device_dmabuf(ibv_pd* pd, void* ptr, size_t bytes, ibv_mr** out,
                                        int access = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_RELAXED_ORDERING) {
  int ordinal = -1;
  if (cuPointerGetAttribute(&ordinal, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
                            reinterpret_cast<CUdeviceptr>(ptr)) != CUDA_SUCCESS || ordinal < 0) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "GPU landing range is not device memory");
  }
  int previous = -1;
  if (cudaGetDevice(&previous) != cudaSuccess) {
    previous = -1;
  }
  if (ordinal != previous && cudaSetDevice(ordinal) != cudaSuccess) {
    return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE, "cannot select the GPU landing range's device");
  }
  const cuteafd_status_t status = register_device_dmabuf_current(pd, ptr, bytes, out, access);
  const std::string message = g_last_error;
  if (ordinal != previous && previous >= 0) {
    cudaSetDevice(previous);
  }
  return status == CUTEAFD_STATUS_OK ? ok() : fail(status, message);
}

cuteafd_status_t register_device_dmabuf_current(ibv_pd* pd, void* ptr, size_t bytes, ibv_mr** out,
                                                int access) {
  CUdeviceptr base = 0;
  size_t size = 0;
  CUresult result = cuMemGetAddressRange(&base, &size, reinterpret_cast<CUdeviceptr>(ptr));
  if (result != CUDA_SUCCESS) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT,
                "cuMemGetAddressRange failed for the GPU landing range (" +
                    std::to_string(static_cast<int>(result)) + ")");
  }
  const uint64_t offset = reinterpret_cast<uint64_t>(ptr) - static_cast<uint64_t>(base);
  if (offset > size || bytes > size - offset) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "GPU landing range exceeds its allocation");
  }
  int fd = -1;
  result = cuMemGetHandleForAddressRange(&fd, base, size, CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0);
  if (result != CUDA_SUCCESS || fd < 0) {
    return fail(CUTEAFD_STATUS_RDMA_UNAVAILABLE,
                "cuMemGetHandleForAddressRange(DMA_BUF_FD) failed (" +
                    std::to_string(static_cast<int>(result)) + ")");
  }
  ibv_mr* mr = ibv_reg_dmabuf_mr(pd, offset, bytes, reinterpret_cast<uint64_t>(ptr), fd, access);
  const int saved_errno = errno;
  close(fd);
  if (mr == nullptr) {
    return fail(CUTEAFD_STATUS_RDMA_UNAVAILABLE,
                std::string("ibv_reg_dmabuf_mr failed: ") + std::strerror(saved_errno));
  }
  *out = mr;
  return ok();
}
}  // namespace
#endif

extern "C" cuteafd_status_t cuteafd_rdma_rc_endpoint_set_recv_landing(
    void* handle, void* device_ptr, size_t bytes, size_t header_bytes) {
  if (handle == nullptr) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "RDMA RC endpoint handle is null");
  }
#if CUTEAFD_NATIVE_ENABLE_RDMA && CUTEAFD_NATIVE_ENABLE_CUDA
  auto* endpoint = static_cast<CuteafdRdmaRcEndpointHandle*>(handle);
  if (endpoint->landing_mr != nullptr) {
    ibv_dereg_mr(endpoint->landing_mr);
    endpoint->landing_mr = nullptr;
    endpoint->landing_ptr = nullptr;
    endpoint->landing_bytes = 0;
    endpoint->landing_header_bytes = 0;
  }
  if (device_ptr == nullptr || bytes == 0) {
    return ok();
  }
  if (bytes > std::numeric_limits<uint32_t>::max() || header_bytes == 0 ||
      header_bytes > endpoint->recv_frame_bytes) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "GPU landing extent is invalid");
  }
  ibv_mr* mr = nullptr;
  const cuteafd_status_t status = register_device_dmabuf(endpoint->pd, device_ptr, bytes, &mr);
  if (status != CUTEAFD_STATUS_OK) {
    return status;
  }
  endpoint->landing_mr = mr;
  endpoint->landing_ptr = static_cast<unsigned char*>(device_ptr);
  endpoint->landing_bytes = bytes;
  endpoint->landing_header_bytes = header_bytes;
  return ok();
#else
  (void)device_ptr;
  (void)bytes;
  (void)header_bytes;
  return fail(CUTEAFD_STATUS_RDMA_UNAVAILABLE,
              "GPU landing requires CUTEAFD_ENABLE_RDMA=ON and CUDA");
#endif
}

extern "C" cuteafd_status_t cuteafd_rdma_rc_endpoint_expose_device(void* handle, void* device_ptr,
                                                               size_t bytes, uint32_t* rkey) {
  if (handle == nullptr || rkey == nullptr) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "RDMA RC endpoint handle or rkey output is null");
  }
#if CUTEAFD_NATIVE_ENABLE_RDMA && CUTEAFD_NATIVE_ENABLE_CUDA
  auto* endpoint = static_cast<CuteafdRdmaRcEndpointHandle*>(handle);
  if (endpoint->exposed_mr != nullptr) {
    ibv_dereg_mr(endpoint->exposed_mr);
    endpoint->exposed_mr = nullptr;
  }
  if (device_ptr == nullptr || bytes == 0) {
    *rkey = 0;
    return ok();
  }
  // No relaxed ordering: a completion flag written after the data must not
  // become visible before it.
  ibv_mr* mr = nullptr;
  const cuteafd_status_t status = register_device_dmabuf(endpoint->pd, device_ptr, bytes, &mr,
                                                         IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
  if (status != CUTEAFD_STATUS_OK) {
    return status;
  }
  endpoint->exposed_mr = mr;
  *rkey = mr->rkey;
  return ok();
#else
  (void)device_ptr;
  (void)bytes;
  return fail(CUTEAFD_STATUS_RDMA_UNAVAILABLE, "exposing device memory requires RDMA and CUDA");
#endif
}

extern "C" cuteafd_status_t cuteafd_rdma_rc_endpoint_post_write_flagged(
    void* handle, size_t offset_bytes, size_t bytes, uint64_t remote_addr, uint32_t rkey,
    uint64_t flag_value, uint64_t flag_remote_addr, uint32_t flag_rkey, uint64_t wr_id) {
  if (handle == nullptr) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "RDMA RC endpoint handle is null");
  }
  if (bytes > std::numeric_limits<uint32_t>::max() || remote_addr == 0 || flag_remote_addr == 0) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "RDMA RC endpoint write is invalid");
  }
#if CUTEAFD_NATIVE_ENABLE_RDMA
  auto* endpoint = static_cast<CuteafdRdmaRcEndpointHandle*>(handle);
  if (offset_bytes > endpoint->send_registered_span_bytes ||
      bytes > endpoint->send_registered_span_bytes - offset_bytes) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "RDMA RC endpoint write source exceeds the send buffer");
  }
  constexpr uint32_t kFlagSlots = 64;
  if (endpoint->flag_source == nullptr) {
    endpoint->flag_source = static_cast<uint64_t*>(std::calloc(kFlagSlots, sizeof(uint64_t)));
    if (endpoint->flag_source == nullptr) {
      return fail(CUTEAFD_STATUS_ALLOCATION_FAILED, "RDMA RC flag source allocation failed");
    }
    endpoint->flag_source_mr = ibv_reg_mr(endpoint->pd, endpoint->flag_source,
                                          kFlagSlots * sizeof(uint64_t), IBV_ACCESS_LOCAL_WRITE);
    if (endpoint->flag_source_mr == nullptr) {
      return fail(CUTEAFD_STATUS_RDMA_UNAVAILABLE, "ibv_reg_mr failed for the RC flag source");
    }
  }
  uint64_t* flag = endpoint->flag_source + (endpoint->flag_cursor++ % kFlagSlots);
  *flag = flag_value;
  ibv_sge data_sge = {};
  data_sge.addr = reinterpret_cast<uintptr_t>(endpoint->send_buffer + offset_bytes);
  data_sge.length = static_cast<uint32_t>(bytes);
  data_sge.lkey = endpoint->send_mr->lkey;
  ibv_sge flag_sge = {};
  flag_sge.addr = reinterpret_cast<uintptr_t>(flag);
  flag_sge.length = sizeof(uint64_t);
  flag_sge.lkey = endpoint->flag_source_mr->lkey;
  // The data write, then the flag write: RC executes them in order and the
  // flag (not relaxed-ordered) lands after the data. Only the flag signals.
  ibv_send_wr flag_wr = {};
  flag_wr.wr_id = wr_id;
  flag_wr.sg_list = &flag_sge;
  flag_wr.num_sge = 1;
  flag_wr.opcode = IBV_WR_RDMA_WRITE;
  flag_wr.send_flags = IBV_SEND_SIGNALED;
  flag_wr.wr.rdma.remote_addr = flag_remote_addr;
  flag_wr.wr.rdma.rkey = flag_rkey;
  ibv_send_wr data_wr = {};
  data_wr.wr_id = wr_id;
  data_wr.sg_list = &data_sge;
  data_wr.num_sge = 1;
  data_wr.opcode = IBV_WR_RDMA_WRITE;
  data_wr.wr.rdma.remote_addr = remote_addr;
  data_wr.wr.rdma.rkey = rkey;
  data_wr.next = &flag_wr;
  ibv_send_wr* first = bytes > 0 ? &data_wr : &flag_wr;
  ibv_send_wr* bad_send = nullptr;
  if (ibv_post_send(endpoint->qp, first, &bad_send) != 0) {
    return fail(CUTEAFD_STATUS_RDMA_UNAVAILABLE, "ibv_post_send of a flagged write failed for RC endpoint");
  }
  return ok();
#else
  (void)offset_bytes; (void)bytes; (void)remote_addr; (void)rkey; (void)flag_value;
  (void)flag_remote_addr; (void)flag_rkey; (void)wr_id;
  return fail(CUTEAFD_STATUS_RDMA_UNAVAILABLE, "RDMA RC endpoint write requires CUTEAFD_ENABLE_RDMA=ON");
#endif
}

extern "C" cuteafd_status_t cuteafd_rdma_gpu_landing_probe(
    const char* device_name, uint32_t port_num, size_t bytes, uint32_t iterations,
    cuteafd_rdma_gpu_landing_probe_t* out) {
  if (out == nullptr) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "GPU landing probe output is null");
  }
  std::memset(out, 0, sizeof(*out));
  out->cuda_device = -1;
  out->writes_ordering = -1;
#if CUTEAFD_NATIVE_ENABLE_RDMA && CUTEAFD_NATIVE_ENABLE_CUDA
  auto finish = [&](cuteafd_status_t status, const std::string& message) {
    set_fixed_string(out->status, sizeof(out->status), message.c_str());
    return status == CUTEAFD_STATUS_OK ? ok() : fail(status, message);
  };
  int cuda_device = 0;
  if (cudaGetDevice(&cuda_device) != cudaSuccess || cudaFree(nullptr) != cudaSuccess) {
    return finish(CUTEAFD_STATUS_CUDA_UNAVAILABLE, "no CUDA device for the GPU landing probe");
  }
  out->cuda_device = cuda_device;
  CUdevice dev = 0;
  if (cuDeviceGet(&dev, cuda_device) == CUDA_SUCCESS) {
    cuDeviceGetAttribute(&out->dma_buf_supported, CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED, dev);
    cuDeviceGetAttribute(&out->gpudirect_rdma_supported,
                         CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED, dev);
    cuDeviceGetAttribute(&out->writes_ordering,
                         CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING, dev);
  }
  if (!out->dma_buf_supported) {
    return finish(CUTEAFD_STATUS_RDMA_UNAVAILABLE, "CUDA reports no dma-buf support");
  }
  if (port_num == 0 || port_num > 255 || bytes == 0 || bytes > (size_t{1} << 30)) {
    return finish(CUTEAFD_STATUS_INVALID_ARGUMENT, "GPU landing probe arguments are invalid");
  }
  constexpr size_t kHeader = 96;
  const size_t frame = kHeader + bytes;
  int device_count = 0;
  ibv_device** devices = ibv_get_device_list(&device_count);
  ibv_device* device = nullptr;
  for (int index = 0; devices != nullptr && index < device_count; ++index) {
    const char* name = ibv_get_device_name(devices[index]);
    if (device_name == nullptr || device_name[0] == '\0' ||
        (name != nullptr && std::strcmp(name, device_name) == 0)) {
      device = devices[index];
      break;
    }
  }
  if (device == nullptr) {
    if (devices != nullptr) {
      ibv_free_device_list(devices);
    }
    return finish(CUTEAFD_STATUS_RDMA_UNAVAILABLE, "no RDMA device for the GPU landing probe");
  }
  set_fixed_string(out->device_name, sizeof(out->device_name), ibv_get_device_name(device));
  ibv_context* context = ibv_open_device(device);
  ibv_free_device_list(devices);
  ibv_pd* pd = nullptr;
  ibv_cq* cq = nullptr;
  ibv_qp* qps[2] = {};
  ibv_mr *send_mr = nullptr, *recv_mr = nullptr, *gpu_mr = nullptr;
  unsigned char *send = nullptr, *recv = nullptr;
  void* gpu = nullptr;
  auto cleanup = [&]() {
    for (ibv_qp* qp : qps) {
      if (qp != nullptr) ibv_destroy_qp(qp);
    }
    for (ibv_mr* mr : {send_mr, recv_mr, gpu_mr}) {
      if (mr != nullptr) ibv_dereg_mr(mr);
    }
    if (cq != nullptr) ibv_destroy_cq(cq);
    if (pd != nullptr) ibv_dealloc_pd(pd);
    if (context != nullptr) ibv_close_device(context);
    if (send != nullptr) cudaFreeHost(send);
    if (recv != nullptr) cudaFreeHost(recv);
    if (gpu != nullptr) cudaFree(gpu);
  };
  auto bail = [&](cuteafd_status_t status, const std::string& message) {
    cleanup();
    return finish(status, message);
  };
  ibv_port_attr port_attr = {};
  if (context == nullptr || ibv_query_port(context, static_cast<uint8_t>(port_num), &port_attr) != 0 ||
      (pd = ibv_alloc_pd(context)) == nullptr ||
      (cq = ibv_create_cq(context, 16, nullptr, nullptr, 0)) == nullptr) {
    return bail(CUTEAFD_STATUS_RDMA_UNAVAILABLE, "RDMA device setup failed for the landing probe");
  }
  if (cudaMalloc(&gpu, bytes) != cudaSuccess ||
      cudaHostAlloc(reinterpret_cast<void**>(&send), frame, cudaHostAllocPortable) != cudaSuccess ||
      cudaHostAlloc(reinterpret_cast<void**>(&recv), frame, cudaHostAllocPortable) != cudaSuccess) {
    return bail(CUTEAFD_STATUS_ALLOCATION_FAILED, "GPU landing probe allocation failed");
  }
  const cuteafd_status_t registered = register_device_dmabuf(pd, gpu, bytes, &gpu_mr);
  if (registered != CUTEAFD_STATUS_OK) {
    const std::string message = g_last_error;
    return bail(registered, message);
  }
  out->registered = 1;
  for (size_t index = 0; index < frame; ++index) {
    send[index] = static_cast<unsigned char>(index * 131 + 7);
  }
  send_mr = ibv_reg_mr(pd, send, frame, IBV_ACCESS_LOCAL_WRITE);
  recv_mr = ibv_reg_mr(pd, recv, frame, IBV_ACCESS_LOCAL_WRITE);
  ibv_qp_init_attr qp_attr = {};
  qp_attr.send_cq = cq;
  qp_attr.recv_cq = cq;
  qp_attr.qp_type = IBV_QPT_RC;
  qp_attr.cap.max_send_wr = 4;
  qp_attr.cap.max_recv_wr = 4;
  qp_attr.cap.max_send_sge = 2;
  qp_attr.cap.max_recv_sge = 2;
  if (send_mr == nullptr || recv_mr == nullptr ||
      (qps[0] = ibv_create_qp(pd, &qp_attr)) == nullptr ||
      (qps[1] = ibv_create_qp(pd, &qp_attr)) == nullptr) {
    return bail(CUTEAFD_STATUS_RDMA_UNAVAILABLE, "GPU landing probe QP setup failed");
  }
  ibv_gid gid = {};
  uint32_t gid_index = 0;
  if (select_rc_gid(context, port_attr, port_num, &gid, &gid_index) != CUTEAFD_STATUS_OK) {
    return bail(CUTEAFD_STATUS_RDMA_UNAVAILABLE, "GPU landing probe found no GID");
  }
  for (int side = 0; side < 2; ++side) {
    if (modify_rc_qp_to_init(qps[side], port_num) != CUTEAFD_STATUS_OK ||
        modify_rc_qp_to_rtr(context, qps[side], port_attr, port_num, qps[1 - side]->qp_num, 0,
                            port_attr.lid, &gid, gid_index) != CUTEAFD_STATUS_OK ||
        modify_rc_qp_to_rts(qps[side], 0) != CUTEAFD_STATUS_OK) {
      return bail(CUTEAFD_STATUS_RDMA_UNAVAILABLE, "GPU landing probe QP connect failed");
    }
  }
  // One SEND per round, received as [header -> host][payload -> host or GPU].
  auto round = [&](bool land, double* seconds) -> bool {
    ibv_sge recv_sge[2] = {};
    recv_sge[0] = {reinterpret_cast<uint64_t>(recv), static_cast<uint32_t>(kHeader), recv_mr->lkey};
    recv_sge[1] = land ? ibv_sge{reinterpret_cast<uint64_t>(gpu), static_cast<uint32_t>(bytes), gpu_mr->lkey}
                       : ibv_sge{reinterpret_cast<uint64_t>(recv + kHeader), static_cast<uint32_t>(bytes),
                                 recv_mr->lkey};
    ibv_recv_wr recv_wr = {};
    recv_wr.sg_list = recv_sge;
    recv_wr.num_sge = 2;
    ibv_recv_wr* bad_recv = nullptr;
    ibv_sge send_sge = {reinterpret_cast<uint64_t>(send), static_cast<uint32_t>(frame), send_mr->lkey};
    ibv_send_wr send_wr = {};
    send_wr.sg_list = &send_sge;
    send_wr.num_sge = 1;
    send_wr.opcode = IBV_WR_SEND;
    send_wr.send_flags = IBV_SEND_SIGNALED;
    ibv_send_wr* bad_send = nullptr;
    if (ibv_post_recv(qps[1], &recv_wr, &bad_recv) != 0) return false;
    const auto started = std::chrono::steady_clock::now();
    if (ibv_post_send(qps[0], &send_wr, &bad_send) != 0) return false;
    int completions = 0;
    while (completions < 2) {
      ibv_wc wc = {};
      const int polled = ibv_poll_cq(cq, 1, &wc);
      if (polled < 0 || (polled == 1 && wc.status != IBV_WC_SUCCESS)) return false;
      completions += polled;
      if (std::chrono::steady_clock::now() - started > std::chrono::seconds(5)) return false;
    }
    *seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    return true;
  };
  const uint32_t rounds = std::max<uint32_t>(1, iterations);
  for (int land = 0; land < 2; ++land) {
    std::vector<double> times;
    for (uint32_t index = 0; index <= rounds; ++index) {
      double seconds = 0.0;
      if (!round(land != 0, &seconds)) {
        return bail(CUTEAFD_STATUS_RDMA_UNAVAILABLE, "GPU landing probe transfer failed");
      }
      if (index > 0) times.push_back(seconds);  // the first round warms up
    }
    std::sort(times.begin(), times.end());
    const double gbps = static_cast<double>(bytes) / times[times.size() / 2] / 1e9;
    (land ? out->gpu_gbps : out->host_gbps) = gbps;
  }
  std::vector<unsigned char> check(bytes);
  if (cudaMemcpy(check.data(), gpu, bytes, cudaMemcpyDeviceToHost) != cudaSuccess ||
      std::memcmp(check.data(), send + kHeader, bytes) != 0) {
    return bail(CUTEAFD_STATUS_INTERNAL_ERROR, "GPU landing probe payload mismatch");
  }
  cleanup();
  return finish(CUTEAFD_STATUS_OK, "dma-buf landing verified");
#else
  (void)device_name;
  (void)port_num;
  (void)bytes;
  (void)iterations;
  set_fixed_string(out->status, sizeof(out->status), "built without RDMA or CUDA");
  return fail(CUTEAFD_STATUS_RDMA_UNAVAILABLE, "GPU landing requires CUTEAFD_ENABLE_RDMA=ON and CUDA");
#endif
}

extern "C" cuteafd_status_t cuteafd_rdma_rc_endpoint_post_recv(void* handle, size_t bytes,
                                                            uint64_t wr_id) {
  return cuteafd_rdma_rc_endpoint_post_recv_at(handle, 0, bytes, wr_id);
}

extern "C" cuteafd_status_t cuteafd_rdma_rc_endpoint_post_send_at(
    void* handle, size_t offset_bytes, size_t bytes, uint64_t wr_id) {
  if (handle == nullptr) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "RDMA RC endpoint handle is null");
  }
  if (bytes == 0 || bytes > std::numeric_limits<uint32_t>::max()) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "RDMA RC endpoint send byte size is invalid");
  }
#if CUTEAFD_NATIVE_ENABLE_RDMA
  auto* endpoint = static_cast<CuteafdRdmaRcEndpointHandle*>(handle);
  if (bytes > endpoint->send_frame_bytes) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT,
                "RDMA RC endpoint send bytes exceed frame capacity");
  }
  if (offset_bytes > endpoint->send_registered_span_bytes ||
      bytes > endpoint->send_registered_span_bytes - offset_bytes) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT,
                "RDMA RC endpoint send slot exceeds registered span");
  }
  ibv_sge send_sge = {};
  send_sge.addr = reinterpret_cast<uintptr_t>(endpoint->send_buffer + offset_bytes);
  send_sge.length = static_cast<uint32_t>(bytes);
  send_sge.lkey = endpoint->send_mr->lkey;
  ibv_send_wr send_wr = {};
  send_wr.wr_id = wr_id;
  send_wr.sg_list = &send_sge;
  send_wr.num_sge = 1;
  send_wr.opcode = IBV_WR_SEND;
  send_wr.send_flags = IBV_SEND_SIGNALED;
  ibv_send_wr* bad_send = nullptr;
  if (ibv_post_send(endpoint->qp, &send_wr, &bad_send) != 0) {
    return fail(CUTEAFD_STATUS_RDMA_UNAVAILABLE, "ibv_post_send failed for RC endpoint");
  }
  return ok();
#else
  (void)offset_bytes;
  (void)bytes;
  (void)wr_id;
  return fail(CUTEAFD_STATUS_RDMA_UNAVAILABLE,
              "RDMA RC endpoint send requires CUTEAFD_ENABLE_RDMA=ON");
#endif
}

extern "C" cuteafd_status_t cuteafd_rdma_rc_endpoint_send_parts_at(
    void* handle, const void* prefix, size_t prefix_bytes, const void* payload,
    size_t payload_bytes, size_t offset_bytes, uint64_t wr_id) {
  if (handle == nullptr) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "RDMA RC endpoint handle is null");
  }
  if ((prefix_bytes != 0 && prefix == nullptr) ||
      (payload_bytes != 0 && payload == nullptr)) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT,
                "RDMA RC endpoint send part pointer is null");
  }
  if (prefix_bytes > std::numeric_limits<size_t>::max() - payload_bytes) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "RDMA RC endpoint send byte size overflows");
  }
  const size_t bytes = prefix_bytes + payload_bytes;
  if (bytes == 0 || bytes > std::numeric_limits<uint32_t>::max()) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "RDMA RC endpoint send byte size is invalid");
  }
#if CUTEAFD_NATIVE_ENABLE_RDMA
  auto* endpoint = static_cast<CuteafdRdmaRcEndpointHandle*>(handle);
  if (bytes > endpoint->send_frame_bytes) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT,
                "RDMA RC endpoint send bytes exceed frame capacity");
  }
  if (offset_bytes > endpoint->send_registered_span_bytes ||
      bytes > endpoint->send_registered_span_bytes - offset_bytes) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT,
                "RDMA RC endpoint send slot exceeds registered span");
  }
  if (prefix_bytes != 0) {
    std::memcpy(endpoint->send_buffer + offset_bytes, prefix, prefix_bytes);
  }
  if (payload_bytes != 0) {
    std::memcpy(endpoint->send_buffer + offset_bytes + prefix_bytes, payload, payload_bytes);
  }
  return cuteafd_rdma_rc_endpoint_post_send_at(handle, offset_bytes, bytes, wr_id);
#else
  (void)prefix;
  (void)prefix_bytes;
  (void)payload;
  (void)payload_bytes;
  (void)offset_bytes;
  (void)wr_id;
  return fail(CUTEAFD_STATUS_RDMA_UNAVAILABLE,
              "RDMA RC endpoint send requires CUTEAFD_ENABLE_RDMA=ON");
#endif
}

extern "C" cuteafd_status_t cuteafd_rdma_rc_endpoint_send_at(
    void* handle, const void* frame, size_t offset_bytes, size_t bytes, uint64_t wr_id) {
  return cuteafd_rdma_rc_endpoint_send_parts_at(handle, frame, bytes, nullptr, 0,
                                               offset_bytes, wr_id);
}

extern "C" cuteafd_status_t cuteafd_rdma_rc_endpoint_send(void* handle, const void* frame,
                                                       size_t bytes, uint64_t wr_id) {
  return cuteafd_rdma_rc_endpoint_send_at(handle, frame, 0, bytes, wr_id);
}

extern "C" cuteafd_status_t cuteafd_rdma_rc_endpoint_poll_with_timeout(
    void* handle, uint32_t expected_send_completions, uint32_t expected_recv_completions,
    uint32_t max_poll_iterations, uint32_t active_event_poll_timeout_ms,
    cuteafd_rdma_rc_completion_stats_t* out) {
  if (handle == nullptr) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "RDMA RC endpoint handle is null");
  }
  if (out == nullptr) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "RDMA RC endpoint poll output pointer is null");
  }
  if (expected_send_completions == 0 && expected_recv_completions == 0) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT,
                "RDMA RC endpoint poll expected completion count is zero");
  }
  // Kept in the ABI for compatibility. Wall-clock deadlines bound this wait.
  (void)max_poll_iterations;
  std::memset(out, 0, sizeof(*out));
  out->expected_send_completions = expected_send_completions;
  out->expected_recv_completions = expected_recv_completions;
#if CUTEAFD_NATIVE_ENABLE_RDMA
  auto* endpoint = static_cast<CuteafdRdmaRcEndpointHandle*>(handle);
  const int effective_active_event_poll_timeout_ms =
      static_cast<int>(std::min<uint32_t>(
          std::max<uint32_t>(active_event_poll_timeout_ms, 1),
          static_cast<uint32_t>(std::numeric_limits<int>::max())));
  const int event_poll_timeout_ms = effective_active_event_poll_timeout_ms;
  auto mark_recent_activity = [&]() {
    endpoint->busy_poll_until =
        std::chrono::steady_clock::now() + kRdmaRcEndpointRecentActivityBusyPollWindow;
  };
  drain_rdma_rc_cq_events(endpoint->send_channel);
  drain_rdma_rc_cq_events(endpoint->recv_channel);
  auto consume_pending = [&]() -> bool {
    bool consumed = false;
    while (out->send_completions < expected_send_completions &&
           endpoint->pending_send_completions > 0) {
      endpoint->pending_send_completions -= 1;
      out->send_completions += 1;
      consumed = true;
    }
    while (out->recv_completions < expected_recv_completions &&
           endpoint->pending_recv_completions > 0) {
      endpoint->pending_recv_completions -= 1;
      out->recv_completions += 1;
      consumed = true;
    }
    if (consumed) {
      mark_recent_activity();
    }
    return out->send_completions >= expected_send_completions &&
           out->recv_completions >= expected_recv_completions;
  };
  if (consume_pending()) {
    set_fixed_string(out->status, sizeof(out->status),
                     "RDMA RC endpoint poll completed from pending completions");
    return ok();
  }
  auto fail_completion = [](const char* cq_name, const ibv_wc& wc) -> cuteafd_status_t {
    char message[256];
    std::snprintf(message, sizeof(message),
                  "RDMA RC endpoint %s completion returned non-success status status=%u (%s) "
                  "opcode=%u wr_id=%llu vendor_err=%u",
                  cq_name, static_cast<unsigned>(wc.status), ibv_wc_status_str(wc.status),
                  static_cast<unsigned>(wc.opcode), static_cast<unsigned long long>(wc.wr_id),
                  static_cast<unsigned>(wc.vendor_err));
    return fail(CUTEAFD_STATUS_RDMA_UNAVAILABLE, message);
  };
  auto complete = [&]() -> bool {
    return consume_pending();
  };
  auto timeout = [&]() -> cuteafd_status_t {
    char message[256];
    std::snprintf(message, sizeof(message),
                  "RDMA RC endpoint timed out waiting for completions send=%u/%u recv=%u/%u "
                  "poll_iterations=%u",
                  static_cast<unsigned>(out->send_completions),
                  static_cast<unsigned>(expected_send_completions),
                  static_cast<unsigned>(out->recv_completions),
                  static_cast<unsigned>(expected_recv_completions),
                  static_cast<unsigned>(out->poll_iterations));
    return fail(CUTEAFD_STATUS_RDMA_UNAVAILABLE, message);
  };
  auto poll_one_cq = [&](ibv_cq* cq, const char* cq_name, ibv_wc_opcode expected_opcode,
                         uint32_t* completions, uint32_t expected) -> cuteafd_status_t {
    if (*completions >= expected) {
      return ok();
    }
    ibv_wc wc = {};
    const int polled = ibv_poll_cq(cq, 1, &wc);
    if (polled < 0) {
      return fail(CUTEAFD_STATUS_INTERNAL_ERROR,
                  std::string("ibv_poll_cq ") + cq_name + " failed for RC endpoint");
    }
    if (polled == 0) {
      return ok();
    }
    if (wc.status != IBV_WC_SUCCESS) {
      return fail_completion(cq_name, wc);
    }
    if (wc.opcode != expected_opcode) {
      return fail(CUTEAFD_STATUS_RDMA_UNAVAILABLE,
                  std::string("RDMA RC endpoint ") + cq_name +
                      " CQ completion had unexpected opcode");
    }
    *completions += 1;
    mark_recent_activity();
    return ok();
  };
  auto poll_incomplete_cqs_once = [&]() -> cuteafd_status_t {
    if (out->poll_iterations < std::numeric_limits<uint32_t>::max()) {
      out->poll_iterations += 1;
    }
    if (out->send_completions < expected_send_completions) {
      const cuteafd_status_t status =
          poll_one_cq(endpoint->send_cq, "send", IBV_WC_SEND, &out->send_completions,
                      expected_send_completions);
      if (status != CUTEAFD_STATUS_OK) {
        return status;
      }
    }
    if (out->recv_completions < expected_recv_completions) {
      const cuteafd_status_t status =
          poll_one_cq(endpoint->recv_cq, "recv", IBV_WC_RECV, &out->recv_completions,
                      expected_recv_completions);
      if (status != CUTEAFD_STATUS_OK) {
        return status;
      }
    }
    return ok();
  };
  const auto initial_busy_deadline =
      std::chrono::steady_clock::now() + kRdmaRcEndpointBusyPollBudget;
  // The recent-activity busy window never outlasts the caller's wait budget:
  // a worker sweeping several connections waits on one with a short budget
  // and must not spin on it for the whole 5 s window while others queue.
  const auto busy_deadline = std::max(
      initial_busy_deadline,
      std::min(endpoint->busy_poll_until,
               std::chrono::steady_clock::now() + std::chrono::milliseconds(event_poll_timeout_ms)));
  do {
    const cuteafd_status_t status = poll_incomplete_cqs_once();
    if (status != CUTEAFD_STATUS_OK) {
      return status;
    }
    if (complete()) {
      set_fixed_string(out->status, sizeof(out->status),
                       "RDMA RC endpoint poll completed after busy poll");
      return ok();
    }
  } while (std::chrono::steady_clock::now() < busy_deadline);

  auto request_notify = [](ibv_cq* cq, const char* cq_name) -> cuteafd_status_t {
    if (ibv_req_notify_cq(cq, 0) != 0) {
      return fail(CUTEAFD_STATUS_RDMA_UNAVAILABLE,
                  std::string("ibv_req_notify_cq ") + cq_name + " failed for RC endpoint");
    }
    return ok();
  };
  auto get_and_ack_cq_event = [](ibv_comp_channel* channel, ibv_cq* expected_cq,
                                 const char* cq_name) -> cuteafd_status_t {
    ibv_cq* event_cq = nullptr;
    void* event_context = nullptr;
    if (ibv_get_cq_event(channel, &event_cq, &event_context) != 0) {
      return fail(CUTEAFD_STATUS_INTERNAL_ERROR,
                  std::string("ibv_get_cq_event ") + cq_name + " failed for RC endpoint");
    }
    ibv_ack_cq_events(event_cq, 1);
    if (event_cq != expected_cq) {
      return fail(CUTEAFD_STATUS_RDMA_UNAVAILABLE,
                  std::string("RDMA RC endpoint ") + cq_name +
                      " completion channel returned an unexpected CQ");
    }
    return ok();
  };
  for (;;) {
    const bool arm_send = out->send_completions < expected_send_completions;
    const bool arm_recv = out->recv_completions < expected_recv_completions;
    if (!arm_send && !arm_recv) {
      set_fixed_string(out->status, sizeof(out->status), "RDMA RC endpoint poll completed");
      return ok();
    }
    if (arm_send) {
      const cuteafd_status_t status = request_notify(endpoint->send_cq, "send");
      if (status != CUTEAFD_STATUS_OK) {
        return status;
      }
    }
    if (arm_recv) {
      const cuteafd_status_t status = request_notify(endpoint->recv_cq, "recv");
      if (status != CUTEAFD_STATUS_OK) {
        return status;
      }
    }

    const cuteafd_status_t poll_status = poll_incomplete_cqs_once();
    if (poll_status != CUTEAFD_STATUS_OK) {
      return poll_status;
    }
    if (complete()) {
      drain_rdma_rc_cq_events(endpoint->send_channel);
      drain_rdma_rc_cq_events(endpoint->recv_channel);
      set_fixed_string(out->status, sizeof(out->status),
                       "RDMA RC endpoint poll completed after CQ notify drain");
      return ok();
    }

    pollfd fds[2] = {};
    ibv_comp_channel* channels[2] = {};
    ibv_cq* cqs[2] = {};
    const char* names[2] = {};
    int fd_count = 0;
    if (out->send_completions < expected_send_completions) {
      fds[fd_count].fd = endpoint->send_channel->fd;
      fds[fd_count].events = POLLIN;
      channels[fd_count] = endpoint->send_channel;
      cqs[fd_count] = endpoint->send_cq;
      names[fd_count] = "send";
      fd_count += 1;
    }
    if (out->recv_completions < expected_recv_completions) {
      fds[fd_count].fd = endpoint->recv_channel->fd;
      fds[fd_count].events = POLLIN;
      channels[fd_count] = endpoint->recv_channel;
      cqs[fd_count] = endpoint->recv_cq;
      names[fd_count] = "recv";
      fd_count += 1;
    }
    int ready = 0;
    do {
      ready = poll(fds, fd_count, event_poll_timeout_ms);
    } while (ready < 0 && errno == EINTR);
    if (ready < 0) {
      return fail(CUTEAFD_STATUS_INTERNAL_ERROR,
                  std::string("poll on RDMA RC completion channels failed: ") +
                      std::strerror(errno));
    }
    if (ready == 0) {
      return timeout();
    }
    for (int i = 0; i < fd_count; ++i) {
      if ((fds[i].revents & (POLLERR | POLLHUP | POLLNVAL)) != 0) {
        return fail(CUTEAFD_STATUS_RDMA_UNAVAILABLE,
                    std::string("RDMA RC endpoint ") + names[i] +
                        " completion channel returned an error event");
      }
      if ((fds[i].revents & POLLIN) != 0) {
        const cuteafd_status_t status = get_and_ack_cq_event(channels[i], cqs[i], names[i]);
        if (status != CUTEAFD_STATUS_OK) {
          return status;
        }
      }
    }
    const cuteafd_status_t post_event_poll_status = poll_incomplete_cqs_once();
    if (post_event_poll_status != CUTEAFD_STATUS_OK) {
      return post_event_poll_status;
    }
    if (complete()) {
      drain_rdma_rc_cq_events(endpoint->send_channel);
      drain_rdma_rc_cq_events(endpoint->recv_channel);
      set_fixed_string(out->status, sizeof(out->status),
                       "RDMA RC endpoint poll completed after CQ event wait");
      return ok();
    }
  }
#else
  (void)active_event_poll_timeout_ms;
  return fail(CUTEAFD_STATUS_RDMA_UNAVAILABLE,
              "RDMA RC endpoint poll requires CUTEAFD_ENABLE_RDMA=ON");
#endif
}

extern "C" cuteafd_status_t cuteafd_rdma_rc_endpoint_poll(
    void* handle, uint32_t expected_send_completions, uint32_t expected_recv_completions,
    uint32_t max_poll_iterations, cuteafd_rdma_rc_completion_stats_t* out) {
  return cuteafd_rdma_rc_endpoint_poll_with_timeout(
      handle, expected_send_completions, expected_recv_completions, max_poll_iterations,
      kRdmaRcEndpointActiveEventPollTimeoutMs, out);
}

extern "C" cuteafd_status_t cuteafd_rdma_rc_endpoint_try_poll(
    void* handle, uint32_t max_send_completions, uint32_t max_recv_completions,
    cuteafd_rdma_rc_completion_stats_t* out) {
  if (handle == nullptr) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "RDMA RC endpoint handle is null");
  }
  if (out == nullptr) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "RDMA RC endpoint poll output pointer is null");
  }
  if (max_send_completions == 0 && max_recv_completions == 0) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT,
                "RDMA RC endpoint try-poll completion count is zero");
  }
  std::memset(out, 0, sizeof(*out));
  out->expected_send_completions = max_send_completions;
  out->expected_recv_completions = max_recv_completions;
#if CUTEAFD_NATIVE_ENABLE_RDMA
  auto* endpoint = static_cast<CuteafdRdmaRcEndpointHandle*>(handle);
  auto fail_completion = [](const char* cq_name, const ibv_wc& wc) -> cuteafd_status_t {
    char message[256];
    std::snprintf(message, sizeof(message),
                  "RDMA RC endpoint %s completion returned non-success status status=%u (%s) "
                  "opcode=%u wr_id=%llu vendor_err=%u",
                  cq_name, static_cast<unsigned>(wc.status), ibv_wc_status_str(wc.status),
                  static_cast<unsigned>(wc.opcode), static_cast<unsigned long long>(wc.wr_id),
                  static_cast<unsigned>(wc.vendor_err));
    return fail(CUTEAFD_STATUS_RDMA_UNAVAILABLE, message);
  };
  auto poll_available = [&](ibv_cq* cq, const char* cq_name, ibv_wc_opcode expected_opcode,
                            uint32_t maximum, uint32_t* completed) -> cuteafd_status_t {
    while (*completed < maximum) {
      ibv_wc wc = {};
      const int polled = ibv_poll_cq(cq, 1, &wc);
      out->poll_iterations += 1;
      if (polled < 0) {
        return fail(CUTEAFD_STATUS_INTERNAL_ERROR,
                    std::string("ibv_poll_cq ") + cq_name + " failed for RC endpoint");
      }
      if (polled == 0) {
        break;
      }
      if (wc.status != IBV_WC_SUCCESS) {
        return fail_completion(cq_name, wc);
      }
      if (wc.opcode != expected_opcode) {
        return fail(CUTEAFD_STATUS_RDMA_UNAVAILABLE,
                    std::string("RDMA RC endpoint ") + cq_name +
                        " CQ completion had unexpected opcode");
      }
      *completed += 1;
    }
    return ok();
  };
  while (out->send_completions < max_send_completions &&
         endpoint->pending_send_completions > 0) {
    endpoint->pending_send_completions -= 1;
    out->send_completions += 1;
  }
  while (out->recv_completions < max_recv_completions &&
         endpoint->pending_recv_completions > 0) {
    endpoint->pending_recv_completions -= 1;
    out->recv_completions += 1;
  }
  cuteafd_status_t status =
      poll_available(endpoint->send_cq, "send", IBV_WC_SEND, max_send_completions,
                     &out->send_completions);
  if (status != CUTEAFD_STATUS_OK) {
    return status;
  }
  status = poll_available(endpoint->recv_cq, "recv", IBV_WC_RECV, max_recv_completions,
                          &out->recv_completions);
  if (status != CUTEAFD_STATUS_OK) {
    return status;
  }
  if (out->send_completions > 0 || out->recv_completions > 0) {
    endpoint->busy_poll_until =
        std::chrono::steady_clock::now() + kRdmaRcEndpointRecentActivityBusyPollWindow;
  }
  set_fixed_string(out->status, sizeof(out->status),
                   out->send_completions > 0 || out->recv_completions > 0
                       ? "RDMA RC endpoint try-poll completed"
                       : "RDMA RC endpoint try-poll would block");
  return ok();
#else
  return fail(CUTEAFD_STATUS_RDMA_UNAVAILABLE,
              "RDMA RC endpoint try-poll requires CUTEAFD_ENABLE_RDMA=ON");
#endif
}

extern "C" cuteafd_status_t cuteafd_rdma_rc_endpoint_copy_recv_at(
    void* handle, void* out, size_t out_bytes, size_t offset_bytes, size_t bytes) {
  if (handle == nullptr) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "RDMA RC endpoint handle is null");
  }
  if (out == nullptr) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "RDMA RC endpoint recv output pointer is null");
  }
  if (bytes == 0 || bytes > out_bytes) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT,
                "RDMA RC endpoint recv copy byte size exceeds output buffer");
  }
#if CUTEAFD_NATIVE_ENABLE_RDMA
  auto* endpoint = static_cast<CuteafdRdmaRcEndpointHandle*>(handle);
  if (bytes > endpoint->recv_frame_bytes) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT,
                "RDMA RC endpoint recv copy bytes exceed frame capacity");
  }
  if (offset_bytes > endpoint->recv_registered_span_bytes ||
      bytes > endpoint->recv_registered_span_bytes - offset_bytes) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT,
                "RDMA RC endpoint recv copy slot exceeds registered span");
  }
  std::memcpy(out, endpoint->recv_buffer + offset_bytes, bytes);
  return ok();
#else
  (void)out_bytes;
  (void)offset_bytes;
  (void)bytes;
  return fail(CUTEAFD_STATUS_RDMA_UNAVAILABLE,
              "RDMA RC endpoint recv copy requires CUTEAFD_ENABLE_RDMA=ON");
#endif
}

extern "C" cuteafd_status_t cuteafd_rdma_rc_endpoint_copy_recv(void* handle, void* out,
                                                            size_t out_bytes, size_t bytes) {
  return cuteafd_rdma_rc_endpoint_copy_recv_at(handle, out, out_bytes, 0, bytes);
}

extern "C" cuteafd_status_t cuteafd_rdma_rc_endpoint_register_region(void* handle, void* ptr,
                                                                size_t bytes, uint32_t* region) {
  if (handle == nullptr || ptr == nullptr || region == nullptr || bytes == 0) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "RDMA RC endpoint region is invalid");
  }
#if CUTEAFD_NATIVE_ENABLE_RDMA
  auto* endpoint = static_cast<CuteafdRdmaRcEndpointHandle*>(handle);
  if (endpoint->regions.size() >= 64) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "RDMA RC endpoint has 64 regions already");
  }
  ibv_mr* mr = ibv_reg_mr(endpoint->pd, ptr, bytes, IBV_ACCESS_LOCAL_WRITE);
  if (mr == nullptr) {
    return fail(CUTEAFD_STATUS_RDMA_UNAVAILABLE, "ibv_reg_mr failed for RC endpoint region");
  }
  endpoint->regions.push_back(mr);
  *region = static_cast<uint32_t>(endpoint->regions.size() - 1);
  return ok();
#else
  return fail(CUTEAFD_STATUS_RDMA_UNAVAILABLE,
              "RDMA RC endpoint regions require CUTEAFD_ENABLE_RDMA=ON");
#endif
}

extern "C" cuteafd_status_t cuteafd_rdma_rc_endpoint_post_send_slot_region(
    void* handle, size_t slot_offset, size_t slot_bytes, uint32_t region, size_t region_offset,
    size_t region_bytes, uint64_t wr_id) {
  if (handle == nullptr) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "RDMA RC endpoint handle is null");
  }
  if (slot_bytes == 0 || region_bytes == 0 ||
      slot_bytes + region_bytes > std::numeric_limits<uint32_t>::max()) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "RDMA RC endpoint gathered send size is invalid");
  }
#if CUTEAFD_NATIVE_ENABLE_RDMA
  auto* endpoint = static_cast<CuteafdRdmaRcEndpointHandle*>(handle);
  if (slot_bytes + region_bytes > endpoint->send_frame_bytes) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT,
                "RDMA RC endpoint send bytes exceed frame capacity");
  }
  if (slot_offset > endpoint->send_registered_span_bytes ||
      slot_bytes > endpoint->send_registered_span_bytes - slot_offset) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT,
                "RDMA RC endpoint send slot exceeds registered span");
  }
  if (region >= endpoint->regions.size()) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "RDMA RC endpoint region is not registered");
  }
  ibv_mr* mr = endpoint->regions[region];
  if (region_offset > mr->length || region_bytes > mr->length - region_offset) {
    return fail(CUTEAFD_STATUS_INVALID_ARGUMENT, "RDMA RC endpoint send exceeds its region");
  }
  ibv_sge sge[2] = {};
  sge[0].addr = reinterpret_cast<uintptr_t>(endpoint->send_buffer + slot_offset);
  sge[0].length = static_cast<uint32_t>(slot_bytes);
  sge[0].lkey = endpoint->send_mr->lkey;
  sge[1].addr = reinterpret_cast<uintptr_t>(mr->addr) + region_offset;
  sge[1].length = static_cast<uint32_t>(region_bytes);
  sge[1].lkey = mr->lkey;
  ibv_send_wr wr = {};
  wr.wr_id = wr_id;
  wr.sg_list = sge;
  wr.num_sge = 2;
  wr.opcode = IBV_WR_SEND;
  wr.send_flags = IBV_SEND_SIGNALED;
  ibv_send_wr* bad = nullptr;
  if (ibv_post_send(endpoint->qp, &wr, &bad) != 0) {
    return fail(CUTEAFD_STATUS_RDMA_UNAVAILABLE, "ibv_post_send failed for a gathered RC send");
  }
  return ok();
#else
  (void)slot_offset;
  (void)region;
  (void)region_offset;
  (void)wr_id;
  return fail(CUTEAFD_STATUS_RDMA_UNAVAILABLE,
              "RDMA RC endpoint regions require CUTEAFD_ENABLE_RDMA=ON");
#endif
}

extern "C" cuteafd_status_t cuteafd_rdma_rc_endpoint_quiesce(void* handle) {
  if (handle == nullptr) {
    return ok();
  }
#if CUTEAFD_NATIVE_ENABLE_RDMA
  auto* endpoint = static_cast<CuteafdRdmaRcEndpointHandle*>(handle);
  if (endpoint->qp != nullptr) {
    if (ibv_destroy_qp(endpoint->qp) != 0) {
      return fail(CUTEAFD_STATUS_RDMA_UNAVAILABLE,
                  "terminal ibv_destroy_qp failed; all registrations and landing owners must remain live");
    }
    endpoint->qp = nullptr;
  }
  // MRs, rings, registered egress and external landing ranges remain owned
  // until the caller has also drained every queued CUDA consumer/upload.
  return ok();
#else
  return fail(CUTEAFD_STATUS_RDMA_UNAVAILABLE,
              "RDMA RC endpoint quiesce requires CUTEAFD_ENABLE_RDMA=ON");
#endif
}

extern "C" cuteafd_status_t cuteafd_rdma_rc_endpoint_destroy(void* handle) {
  if (handle == nullptr) {
    return ok();
  }
#if CUTEAFD_NATIVE_ENABLE_RDMA
  destroy_rdma_rc_endpoint(static_cast<CuteafdRdmaRcEndpointHandle*>(handle));
  return ok();
#else
  return fail(CUTEAFD_STATUS_RDMA_UNAVAILABLE,
              "RDMA RC endpoint destroy requires CUTEAFD_ENABLE_RDMA=ON");
#endif
}

#if !CUTEAFD_NATIVE_ENABLE_CUDA
extern "C" cuteafd_status_t cuteafd_cuda_rmsnorm_f32(const float*, const float*, float*, int, int,
                                                 float) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE, "CUDA RMSNorm kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_rmsnorm_f32_async(const float*, const float*, float*, int, int,
                                                       float, void*) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE, "CUDA RMSNorm kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_rmsnorm_bf16(const uint16_t*, const uint16_t*, uint16_t*, int,
                                                  int, float) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA BF16 RMSNorm kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_rmsnorm_bf16_async(const uint16_t*, const uint16_t*,
                                                        uint16_t*, int, int, float, void*) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA BF16 RMSNorm kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_ds4_flash_rmsnorm_bf16_async(
    const uint16_t*, const uint16_t*, uint16_t*, int, int, float, void*) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "DeepSeek V4 Flash BF16 RMSNorm kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_ds4_rmsnorm_bf16_rne_async(
    const uint16_t*, const uint16_t*, uint16_t*, int, int, float, void*) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "DeepSeek V4 BF16 RNE RMSNorm kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_graph_update_rmsnorm_bf16_node(
    void*, void*, size_t, cuteafd_device_buffer_t, cuteafd_device_buffer_t, cuteafd_device_buffer_t, int,
    int, float) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA BF16 RMSNorm graph node update is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_graph_update_linear_bf16_node(
    void*, void*, size_t, cuteafd_device_buffer_t, cuteafd_device_buffer_t,
    const cuteafd_device_buffer_t*, cuteafd_device_buffer_t, size_t, size_t, size_t) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA BF16 linear graph node update is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_graph_update_router_topk_bf16_node(
    void*, void*, size_t, cuteafd_device_buffer_t, cuteafd_device_buffer_t, cuteafd_device_buffer_t,
    cuteafd_device_buffer_t, cuteafd_device_buffer_t, cuteafd_device_buffer_t, size_t, size_t, size_t,
    size_t) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA BF16 router top-k graph node update is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_graph_update_silu_gated_mlp_rows_bf16_down_stride_node(
    void*, void*, size_t, cuteafd_device_buffer_t, cuteafd_device_buffer_t, cuteafd_device_buffer_t,
    cuteafd_device_buffer_t, cuteafd_device_buffer_t, size_t, size_t, size_t, size_t) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA BF16 strided-down MLP graph node update is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_graph_update_residual_add_f32_delta_bf16_node(
    void*, void*, size_t, cuteafd_device_buffer_t, cuteafd_device_buffer_t, cuteafd_device_buffer_t,
    size_t) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA BF16 residual add from F32 delta graph node update is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_graph_update_residual_add_shared_f32_delta_bf16_node(
    void*, void*, size_t, cuteafd_device_buffer_t, cuteafd_device_buffer_t, cuteafd_device_buffer_t,
    cuteafd_device_buffer_t, size_t) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA BF16 residual add from shared plus F32 delta graph node update is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_graph_update_mla_kv_cache_unpack_bf16_node(
    void*, void*, size_t, cuteafd_device_buffer_t, cuteafd_device_buffer_t, cuteafd_device_buffer_t,
    cuteafd_device_buffer_t, size_t, size_t, size_t, size_t, size_t) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA MLA KV cache unpack graph node update is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_graph_update_mla_kv_projected_split_bf16_node(
    void*, void*, size_t, cuteafd_device_buffer_t, cuteafd_device_buffer_t, cuteafd_device_buffer_t, size_t,
    size_t, size_t, size_t) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA MLA KV projected split graph node update is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_layernorm_affine_f32_bf16(const float*, const uint16_t*,
                                                               const uint16_t*, float*, int, int,
                                                               float) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA F32/BF16 affine LayerNorm kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_layernorm_affine_f32_bf16_async(
    const float*, const uint16_t*, const uint16_t*, float*, int, int, float, void*) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA F32/BF16 affine LayerNorm kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_layernorm_affine_bf16(const uint16_t*, const uint16_t*,
                                                           const uint16_t*, uint16_t*, int, int,
                                                           float) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA BF16 affine LayerNorm kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_layernorm_affine_bf16_async(
    const uint16_t*, const uint16_t*, const uint16_t*, uint16_t*, int, int, float, void*) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA BF16 affine LayerNorm kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_silu_gated_mlp_f32(const float*, const float*, const float*,
                                                        const float*, float*, int, int) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE, "CUDA gated MLP kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_silu_gated_mlp_rows_f32(const float*, const float*,
                                                             const float*, const float*, float*,
                                                             size_t, size_t, size_t) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA row-batched gated MLP kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_silu_gated_mlp_rows_f32_async(
    const float*, const float*, const float*, const float*, float*, size_t, size_t, size_t, void*) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA row-batched gated MLP kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_silu_gated_mlp_rows_bf16(
    const uint16_t*, const uint16_t*, const uint16_t*, const uint16_t*, uint16_t*, size_t, size_t,
    size_t) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA BF16 row-batched gated MLP kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_silu_gated_mlp_rows_bf16_async(
    const uint16_t*, const uint16_t*, const uint16_t*, const uint16_t*, uint16_t*, size_t, size_t,
    size_t, void*) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA BF16 row-batched gated MLP kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_silu_gated_mlp_rows_bf16_down_stride(
    const uint16_t*, const uint16_t*, const uint16_t*, const uint16_t*, uint16_t*, size_t, size_t,
    size_t, size_t) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA BF16 row-batched strided-down gated MLP kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_silu_gated_mlp_rows_bf16_down_stride_async(
    const uint16_t*, const uint16_t*, const uint16_t*, const uint16_t*, uint16_t*, size_t, size_t,
    size_t, size_t, void*) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA BF16 row-batched strided-down gated MLP kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_silu_gated_mlp_rows_bf16_down_stride_staged(
    const uint16_t*, const uint16_t*, const uint16_t*, const uint16_t*, float*, uint16_t*, size_t,
    size_t, size_t, size_t) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA BF16 staged strided-down gated MLP kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_silu_gated_mlp_rows_bf16_down_stride_staged_async(
    const uint16_t*, const uint16_t*, const uint16_t*, const uint16_t*, float*, uint16_t*, size_t,
    size_t, size_t, size_t, void*) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA BF16 staged strided-down gated MLP kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_nvfp4_silu_gated_mlp_route_bf16_grouped_staged_accumulate_f32(
    const uint16_t*, const uint32_t*, const float*, const uint8_t*, const uint8_t*,
    const uint8_t*, const uint8_t*, const uint8_t*, const uint8_t*, float*, float*, size_t,
    size_t, size_t, size_t, size_t, size_t, size_t, size_t, float, float, float) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA staged accumulated NVFP4 routed expert MLP BF16 kernel is unavailable in this build");
}

extern "C" cuteafd_status_t
cuteafd_cuda_nvfp4_silu_gated_mlp_route_bf16_grouped_staged_accumulate_f32_async(
    const uint16_t*, const uint32_t*, const float*, const uint8_t*, const uint8_t*,
    const uint8_t*, const uint8_t*, const uint8_t*, const uint8_t*, float*, float*, size_t,
    size_t, size_t, size_t, size_t, size_t, size_t, size_t, float, float, float, void*) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA staged accumulated NVFP4 routed expert MLP BF16 kernel is unavailable in this build");
}

extern "C" cuteafd_status_t
cuteafd_cuda_nvfp4_silu_gated_mlp_route_bf16_batched_staged_accumulate_f32(
    const uint16_t*, const uint32_t*, const float*,
    const cuteafd_nvfp4_route_batched_metadata_t*, float*, float*, size_t, size_t, size_t, size_t,
    size_t, size_t) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA batched staged accumulated NVFP4 routed expert MLP BF16 kernel is unavailable in this build");
}

extern "C" cuteafd_status_t
cuteafd_cuda_nvfp4_silu_gated_mlp_route_bf16_batched_staged_accumulate_f32_async(
    const uint16_t*, const uint32_t*, const float*,
    const cuteafd_nvfp4_route_batched_metadata_t*, float*, float*, size_t, size_t, size_t, size_t,
    size_t, size_t, void*) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA batched staged accumulated NVFP4 routed expert MLP BF16 kernel is unavailable in this build");
}

extern "C" cuteafd_status_t
cuteafd_cuda_nvfp4_silu_gated_mlp_route_bf16_batched_staged_single_row_bf16(
    const uint16_t*, const uint32_t*, const float*,
    const cuteafd_nvfp4_route_batched_metadata_t*, float*, uint16_t*, size_t, size_t, size_t, size_t,
    size_t, size_t) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA batched staged single-row NVFP4 routed expert MLP BF16 output kernel is unavailable in this build");
}

extern "C" cuteafd_status_t
cuteafd_cuda_nvfp4_silu_gated_mlp_route_bf16_batched_staged_single_row_bf16_async(
    const uint16_t*, const uint32_t*, const float*,
    const cuteafd_nvfp4_route_batched_metadata_t*, float*, uint16_t*, size_t, size_t, size_t, size_t,
    size_t, size_t, void*) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA batched staged single-row NVFP4 routed expert MLP BF16 output kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_residual_add_f32(const float*, const float*, float*, size_t) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA residual add kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_residual_add_f32_async(const float*, const float*, float*,
                                                            size_t, void*) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA residual add kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_residual_add_bf16(const uint16_t*, const uint16_t*,
                                                       uint16_t*, size_t) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA BF16 residual add kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_residual_add_bf16_async(const uint16_t*, const uint16_t*,
                                                             uint16_t*, size_t, void*) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA BF16 residual add kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_residual_add_f32_delta_bf16(const uint16_t*, const float*,
                                                                 uint16_t*, size_t) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA BF16 residual add from F32 delta kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_residual_add_f32_delta_bf16_async(
    const uint16_t*, const float*, uint16_t*, size_t, void*) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA BF16 residual add from F32 delta kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_residual_add_shared_f32_delta_bf16(
    const uint16_t*, const uint16_t*, const float*, uint16_t*, size_t) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA BF16 residual add from shared plus F32 delta kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_residual_add_shared_f32_delta_bf16_async(
    const uint16_t*, const uint16_t*, const float*, uint16_t*, size_t, void*) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA BF16 residual add from shared plus F32 delta kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_residual_add_shared_fp8_e4m3_row_scaled_bf16_async(
    const uint16_t*, const uint16_t*, const uint8_t*, uint16_t*, size_t, void*) {
  return fail(
      CUTEAFD_STATUS_CUDA_UNAVAILABLE,
      "CUDA BF16 residual add from shared plus row-scaled FP8 delta kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_scheduler_mlp_delta_bf16(
    const uint16_t*, const uint16_t*, const uint16_t*, const uint16_t*, uint16_t*, size_t,
    size_t) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA scheduler BF16 MLP delta kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_scheduler_mlp_delta_bf16_async(
    const uint16_t*, const uint16_t*, const uint16_t*, const uint16_t*, uint16_t*, size_t, size_t,
    void*) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA scheduler BF16 MLP delta kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_summarize_bf16(const uint16_t*, size_t,
                                                    cuteafd_bf16_summary_t*) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA BF16 summary kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_summarize_bf16_async(const uint16_t*, size_t,
                                                          cuteafd_bf16_summary_t*, void*) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA BF16 summary kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_zero_f32(float*, size_t) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE, "CUDA F32 zero kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_zero_f32_async(float*, size_t, void*) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE, "CUDA F32 zero kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_accumulate_bf16_to_f32(const uint16_t*, float*, size_t) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA BF16-to-F32 accumulation kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_accumulate_bf16_to_f32_async(const uint16_t*, float*,
                                                                    size_t, void*) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA BF16-to-F32 accumulation kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_zero_bytes(void*, size_t) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE, "CUDA byte zero kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_zero_bytes_async(void*, size_t, void*) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE, "CUDA byte zero kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_f32_to_bf16(const float*, uint16_t*, size_t) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA F32-to-BF16 conversion kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_f32_to_bf16_async(const float*, uint16_t*, size_t, void*) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA F32-to-BF16 conversion kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_gather_rows_f32(const float*, const uint32_t*, float*, size_t,
                                                     size_t) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA row gather kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_gather_rows_f32_async(const float*, const uint32_t*, float*,
                                                           size_t, size_t, void*) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA row gather kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_gather_rows_f32_to_fp8_e4m3_row_scaled(
    const float*, const uint32_t*, uint8_t*, size_t, size_t, size_t) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA row-scaled FP8 E4M3 gather kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_gather_rows_f32_to_fp8_e4m3_row_scaled_async(
    const float*, const uint32_t*, uint8_t*, size_t, size_t, size_t, void*) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA row-scaled FP8 E4M3 gather kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_gather_rows_f32_to_nvfp4_e2m1_fp8_e4m3(
    const float*, const uint32_t*, uint8_t*, size_t, size_t, size_t) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA NVFP4 row gather kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_gather_rows_f32_to_nvfp4_e2m1_fp8_e4m3_async(
    const float*, const uint32_t*, uint8_t*, size_t, size_t, size_t, void*) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA NVFP4 row gather kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_gather_rows_bf16(const uint16_t*, const uint32_t*, uint16_t*,
                                                      size_t, size_t) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA BF16 row gather kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_gather_rows_bf16_async(const uint16_t*, const uint32_t*,
                                                            uint16_t*, size_t, size_t, void*) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA BF16 row gather kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_copy_row_prefix_bf16(const uint16_t*, uint16_t*, size_t,
                                                          size_t, size_t, size_t, size_t) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA BF16 row-prefix copy kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_copy_row_prefix_bf16_async(
    const uint16_t*, uint16_t*, size_t, size_t, size_t, size_t, size_t, void*) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA BF16 row-prefix copy kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_scatter_add_rows_f32(const float*, const uint32_t*, float*,
                                                          size_t, size_t) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA row scatter-add kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_scatter_add_rows_f32_async(const float*, const uint32_t*,
                                                                float*, size_t, size_t, void*) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA row scatter-add kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_scatter_add_rows_bf16_to_f32(const uint16_t*,
                                                                  const uint32_t*, float*,
                                                                  size_t, size_t) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA BF16-to-F32 row scatter-add kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_scatter_add_rows_bf16_to_f32_async(
    const uint16_t*, const uint32_t*, float*, size_t, size_t, void*) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA BF16-to-F32 row scatter-add kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_scatter_add_rows_fp8_e4m3_row_scaled_to_f32(
    const uint8_t*, size_t, const uint32_t*, float*, size_t, size_t) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA row-scaled FP8 E4M3-to-F32 scatter-add kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_scatter_add_rows_fp8_e4m3_row_scaled_to_f32_async(
    const uint8_t*, size_t, const uint32_t*, float*, size_t, size_t, void*) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA row-scaled FP8 E4M3-to-F32 scatter-add kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_scatter_add_rows_nvfp4_e2m1_fp8_e4m3_to_f32(
    const uint8_t*, size_t, const uint32_t*, float*, size_t, size_t) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA NVFP4-to-F32 row scatter-add kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_scatter_add_rows_nvfp4_e2m1_fp8_e4m3_to_f32_async(
    const uint8_t*, size_t, const uint32_t*, float*, size_t, size_t, void*) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA NVFP4-to-F32 row scatter-add kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_scatter_add_rows_bf16_weighted_to_f32(
    const uint16_t*, const uint32_t*, const float*, float*, size_t, size_t) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA weighted BF16-to-F32 row scatter-add kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_scatter_add_rows_bf16_weighted_to_f32_async(
    const uint16_t*, const uint32_t*, const float*, float*, size_t, size_t, void*) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA weighted BF16-to-F32 row scatter-add kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_kv_cache_write_bytes(const uint8_t*, uint8_t*, size_t,
                                                          size_t) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA KV cache byte write kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_kv_cache_write_bytes_async(const uint8_t*, uint8_t*,
                                                                size_t, size_t, void*) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA KV cache byte write kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_kv_cache_read_bytes(const uint8_t*, uint8_t*, size_t,
                                                         size_t) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA KV cache byte read kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_kv_cache_read_bytes_async(const uint8_t*, uint8_t*, size_t,
                                                               size_t, void*) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA KV cache byte read kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_kv_cache_write_blocks(const uint8_t*, uint8_t*,
                                                           const uint64_t*, const uint64_t*,
                                                           const uint64_t*, size_t) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA KV cache block write kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_kv_cache_write_blocks_async(
    const uint8_t*, uint8_t*, const uint64_t*, const uint64_t*, const uint64_t*, size_t, void*) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA KV cache block write kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_kv_cache_read_blocks(const uint8_t*, uint8_t*,
                                                          const uint64_t*, const uint64_t*,
                                                          const uint64_t*, size_t) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA KV cache block read kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_kv_cache_read_blocks_async(
    const uint8_t*, uint8_t*, const uint64_t*, const uint64_t*, const uint64_t*, size_t, void*) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA KV cache block read kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_mla_kv_cache_unpack_bf16(
    const uint8_t*, uint16_t*, uint16_t*, uint16_t*, size_t, size_t, size_t, size_t, size_t) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA MLA KV cache unpack kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_mla_kv_cache_unpack_bf16_async(
    const uint8_t*, uint16_t*, uint16_t*, uint16_t*, size_t, size_t, size_t, size_t, size_t,
    void*) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA MLA KV cache unpack kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_mla_kv_projected_split_bf16(
    const uint16_t*, uint16_t*, uint16_t*, size_t, size_t, size_t, size_t) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA MLA KV projected split kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_mla_kv_projected_split_bf16_async(
    const uint16_t*, uint16_t*, uint16_t*, size_t, size_t, size_t, size_t, void*) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA MLA KV projected split kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_mla_kv_prepare_bf16(
    const uint16_t*, const uint32_t*, const uint16_t*, uint16_t*, size_t, size_t, size_t, float,
    float) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA MLA KV prepare kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_mla_kv_prepare_bf16_async(
    const uint16_t*, const uint32_t*, const uint16_t*, uint16_t*, size_t, size_t, size_t, float,
    float, void*) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA MLA KV prepare kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_transpose_rows_heads_bf16(
    const uint16_t*, uint16_t*, size_t, size_t, size_t) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA BF16 rows/heads transpose kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_transpose_rows_heads_bf16_async(
    const uint16_t*, uint16_t*, size_t, size_t, size_t, void*) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA BF16 rows/heads transpose kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_transpose_heads_rows_bf16(
    const uint16_t*, uint16_t*, size_t, size_t, size_t) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA BF16 heads/rows transpose kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_transpose_heads_rows_bf16_async(
    const uint16_t*, uint16_t*, size_t, size_t, size_t, void*) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA BF16 heads/rows transpose kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_mla_compose_absorbed_query_bf16(
    const uint16_t*, const uint16_t*, uint16_t*, size_t, size_t, size_t,
    size_t) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA MLA absorbed-query compose kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_mla_compose_absorbed_query_bf16_async(
    const uint16_t*, const uint16_t*, uint16_t*, size_t, size_t, size_t,
    size_t, void*) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA MLA absorbed-query compose kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_generic_kv_page_table_init(
    int32_t*, size_t, size_t) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA generic KV page-table init kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_generic_kv_page_table_init_async(
    int32_t*, size_t, size_t, void*) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA generic KV page-table init kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_generic_kv_page_table_init_base(
    int32_t*, size_t, size_t, size_t) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA generic KV base page-table init kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_generic_kv_page_table_init_base_async(
    int32_t*, size_t, size_t, size_t, void*) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA generic KV base page-table init kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_generic_kv_page_table_init_offsets(
    int32_t*, const int32_t*, size_t, size_t) {
  return fail(
      CUTEAFD_STATUS_CUDA_UNAVAILABLE,
      "CUDA generic KV offset page-table init kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_generic_kv_page_table_init_offsets_async(
    int32_t*, const int32_t*, size_t, size_t, void*) {
  return fail(
      CUTEAFD_STATUS_CUDA_UNAVAILABLE,
      "CUDA generic KV offset page-table init kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_generic_kv_page_table_expand_indices(
    int32_t*, const uint32_t*, size_t, size_t, size_t) {
  return fail(
      CUTEAFD_STATUS_CUDA_UNAVAILABLE,
      "CUDA generic KV page-table expansion kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_generic_kv_page_table_expand_indices_async(
    int32_t*, const uint32_t*, size_t, size_t, size_t, void*) {
  return fail(
      CUTEAFD_STATUS_CUDA_UNAVAILABLE,
      "CUDA generic KV page-table expansion kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_mla_kv_pack_fp8_ds_mla(const uint16_t*, uint8_t*, size_t,
                                                            size_t, size_t) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA MLA KV FP8 DS pack kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_mla_kv_pack_fp8_ds_mla_async(
    const uint16_t*, uint8_t*, size_t, size_t, size_t, void*) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA MLA KV FP8 DS pack kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_mla_kv_unpack_fp8_ds_mla(const uint8_t*, uint16_t*, size_t,
                                                              size_t, size_t) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA MLA KV FP8 DS unpack kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_mla_kv_unpack_fp8_ds_mla_async(
    const uint8_t*, uint16_t*, size_t, size_t, size_t, void*) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA MLA KV FP8 DS unpack kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_mla_kv_pack_mxfp4_ds_mla(const uint16_t*, uint8_t*, size_t,
                                                              size_t, size_t) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA MLA KV MXFP4 DS pack kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_mla_kv_pack_mxfp4_ds_mla_async(
    const uint16_t*, uint8_t*, size_t, size_t, size_t, void*) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA MLA KV MXFP4 DS pack kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_mla_kv_unpack_mxfp4_ds_mla(const uint8_t*, uint16_t*,
                                                                size_t, size_t, size_t) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA MLA KV MXFP4 DS unpack kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_mla_kv_unpack_mxfp4_ds_mla_async(
    const uint8_t*, uint16_t*, size_t, size_t, size_t, void*) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA MLA KV MXFP4 DS unpack kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_router_topk_f32(const float*, const float*, const float*,
                                                     uint32_t*, float*, float*, size_t, size_t,
                                                     size_t, size_t) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA router top-k kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_router_topk_f32_async(const float*, const float*,
                                                           const float*, uint32_t*, float*,
                                                           float*, size_t, size_t, size_t, size_t,
                                                           void*) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA router top-k kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_router_topk_bf16(const uint16_t*, const uint16_t*,
                                                      const float*, uint32_t*, float*, float*,
                                                      size_t, size_t, size_t, size_t) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA BF16 router top-k kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_router_topk_bf16_async(
    const uint16_t*, const uint16_t*, const float*, uint32_t*, float*, float*, size_t, size_t,
    size_t, size_t, void*) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA BF16 router top-k kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_ds4_flash_router_topk_bf16(
    const uint16_t *, const uint16_t *, const float *, const int64_t *,
    const int64_t *, uint32_t *, float *, float *, size_t, int) {
  return fail(
      CUTEAFD_STATUS_CUDA_UNAVAILABLE,
      "CUDA DeepSeek-V4-Flash router kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_ds4_flash_router_topk_bf16_async(
    const uint16_t *, const uint16_t *, const float *, const int64_t *,
    const int64_t *, uint32_t *, float *, float *, size_t, int, void *) {
  return fail(
      CUTEAFD_STATUS_CUDA_UNAVAILABLE,
      "CUDA DeepSeek-V4-Flash router kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_ds4_flash_router_refine_topk_bf16_async(
    const uint16_t *, const uint16_t *, const float *, float *, uint32_t *,
    float *, float *, size_t, void *) {
  return fail(
      CUTEAFD_STATUS_CUDA_UNAVAILABLE,
      "CUDA DeepSeek-V4-Flash hybrid router refinement is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_router_topk_bf16_cub(
    const uint16_t*, const uint16_t*, const float*, float*, float*, uint32_t*, uint32_t*, int*,
    uint32_t*, float*, float*, void*, size_t, size_t, size_t, size_t, size_t) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA BF16 CUB router top-k kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_router_topk_bf16_cub_async(
    const uint16_t*, const uint16_t*, const float*, float*, float*, uint32_t*, uint32_t*, int*,
    uint32_t*, float*, float*, void*, size_t, size_t, size_t, size_t, size_t, void*) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA BF16 CUB router top-k kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_linear_f32(const float*, const float*, const float*, float*,
                                                 size_t, size_t, size_t) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA linear projection kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_linear_f32_async(const float*, const float*, const float*,
                                                      float*, size_t, size_t, size_t, void*) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA linear projection kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_linear_bf16(const uint16_t*, const uint16_t*,
                                                 const uint16_t*, uint16_t*, size_t, size_t,
                                                 size_t) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA BF16 linear projection kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_linear_bf16_async(const uint16_t*, const uint16_t*,
                                                       const uint16_t*, uint16_t*, size_t, size_t,
                                                       size_t, void*) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA BF16 linear projection kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_linear_bf16_cublas(const uint16_t*, const uint16_t*,
                                                        const uint16_t*, uint16_t*, size_t,
                                                        size_t, size_t) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA BF16 cuBLAS linear projection kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_linear_bf16_cublas_async(
    const uint16_t*, const uint16_t*, const uint16_t*, uint16_t*, size_t, size_t, size_t, void*) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA BF16 cuBLAS linear projection kernel is unavailable in this build");
}

extern "C" cuteafd_status_t
cuteafd_cuda_linear_bf16_m1_parity_batched_cublaslt_async(
    const uint16_t*, const uint16_t*, uint16_t*, size_t, size_t, size_t,
    void*) {
  return fail(
      CUTEAFD_STATUS_CUDA_UNAVAILABLE,
      "CUDA BF16 parity-batched cuBLASLt projection kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_quantize_bf16_w8a16_group256_async(
    const uint16_t*, int8_t*, float*, size_t, size_t, int, void*) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA W8A16 projection quantizer is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_quantize_bf16_w8a16_group256_packed_async(
    const uint16_t*, int8_t*, float*, size_t, size_t, void*) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA packed W8A16 projection quantizer is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_dequantize_w8a16_group256_bf16_async(
    const int8_t*, const float*, uint16_t*, size_t, size_t, void*) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA W8A16 projection dequantizer is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_linear_w8a16_group256_m1_simt_async(
    const uint16_t*, const int8_t*, const float*, uint16_t*, size_t, size_t,
    int, void*) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA W8A16 SIMT projection kernel is unavailable in this build");
}

extern "C" cuteafd_status_t
cuteafd_cuda_linear_w8a16_group256_m1_parity_batched_async(
    const uint16_t*, const int8_t*, const float*, uint16_t*, size_t, size_t,
    size_t, void*) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA W8A16 parity-batched SIMT projection kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_linear_w8a16_group256_m1_warp_packed_async(
    const uint16_t*, const int8_t*, const float*, uint16_t*, size_t, size_t,
    void*) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA W8A16 packed projection kernel is unavailable in this build");
}

extern "C" cuteafd_status_t
cuteafd_cuda_linear_w8a16_group256_m1_warp_packed_parity_batched_async(
    const uint16_t*, const int8_t*, const float*, uint16_t*, size_t, size_t,
    size_t, void*) {
  return fail(
      CUTEAFD_STATUS_CUDA_UNAVAILABLE,
      "CUDA W8A16 packed parity-batched projection kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_linear_w8a8_group256_wmma_async(
    const int8_t*, const float*, const int8_t*, const float*, uint16_t*,
    size_t, size_t, size_t, void*) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA W8A8 projection kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_linear_w8a16_group256_triton_file_async(
    const uint16_t*, const int8_t*, const float*, uint16_t*, size_t, size_t,
    size_t, const char*, const char*, size_t, size_t, size_t, size_t, void*) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA W8A16 Triton AOT launcher is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_causal_attention_f32(const float*, const float*, const float*,
                                                          float*, size_t, size_t, size_t, size_t,
                                                          float) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA causal attention kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_causal_attention_f32_async(
    const float*, const float*, const float*, float*, size_t, size_t, size_t, size_t, float,
    void*) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA causal attention kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_causal_attention_bf16(
    const uint16_t*, const uint16_t*, const uint16_t*, uint16_t*, size_t, size_t, size_t, size_t,
    float) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA BF16 causal attention kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_causal_attention_bf16_async(
    const uint16_t*, const uint16_t*, const uint16_t*, uint16_t*, size_t, size_t, size_t, size_t,
    float, void*) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA BF16 causal attention kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_rope_f32(const float*, const uint32_t*, float*, size_t,
                                              size_t, size_t, float) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE, "CUDA RoPE kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_rope_f32_async(const float*, const uint32_t*, float*, size_t,
                                                    size_t, size_t, float, void*) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE, "CUDA RoPE kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_rope_bf16(const uint16_t*, const uint32_t*, uint16_t*,
                                               size_t, size_t, size_t, float) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA BF16 RoPE kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_rope_bf16_async(const uint16_t*, const uint32_t*, uint16_t*,
                                                     size_t, size_t, size_t, float, void*) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA BF16 RoPE kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_ds4_dspark_prompt_metadata_async(
    uint32_t*, uint32_t*, float*, size_t, size_t, size_t, float, void*) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "DeepSeek dSpark prompt metadata kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_ds4_dspark_prepare_proposal_async(
    const uint16_t*, uint32_t*, uint16_t*, uint32_t*, uint32_t*, float*,
    int32_t*, int32_t*, size_t, size_t, size_t, size_t, size_t, size_t,
    size_t, float, void*) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "DeepSeek dSpark proposal preparation kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_mla_rope_attention_bf16(
    const uint16_t*, const uint16_t*, const uint16_t*, const uint16_t*, const uint16_t*, uint16_t*,
    size_t, size_t, size_t, size_t, size_t, float) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA BF16 MLA/RoPE attention kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_mla_rope_attention_bf16_async(
    const uint16_t*, const uint16_t*, const uint16_t*, const uint16_t*, const uint16_t*, uint16_t*,
    size_t, size_t, size_t, size_t, size_t, float, void*) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA BF16 MLA/RoPE attention kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_mla_rope_attention_bf16_suffix(
    const uint16_t*, const uint16_t*, const uint16_t*, const uint16_t*, const uint16_t*, uint16_t*,
    size_t, size_t, size_t, size_t, size_t, size_t, size_t, float) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA BF16 MLA/RoPE suffix attention kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_mla_rope_attention_bf16_suffix_async(
    const uint16_t*, const uint16_t*, const uint16_t*, const uint16_t*, const uint16_t*, uint16_t*,
    size_t, size_t, size_t, size_t, size_t, size_t, size_t, float, void*) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA BF16 MLA/RoPE suffix attention kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_graph_update_mla_rope_attention_bf16_suffix_node(
    void*, void*, size_t, cuteafd_device_buffer_t, cuteafd_device_buffer_t, cuteafd_device_buffer_t,
    cuteafd_device_buffer_t, cuteafd_device_buffer_t, cuteafd_device_buffer_t, size_t, size_t, size_t,
    size_t, size_t, size_t, size_t, float) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA BF16 MLA/RoPE suffix attention graph update is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_embedding_lookup_f32(const float*, const uint32_t*, float*,
                                                          size_t, size_t, size_t) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA embedding lookup kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_embedding_lookup_f32_async(const float*, const uint32_t*,
                                                                float*, size_t, size_t, size_t,
                                                                void*) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA embedding lookup kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_embedding_lookup_bf16(const uint16_t*, const uint32_t*,
                                                           uint16_t*, size_t, size_t, size_t) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA BF16 embedding lookup kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_embedding_lookup_bf16_async(const uint16_t*,
                                                                 const uint32_t*, uint16_t*,
                                                                 size_t, size_t, size_t, void*) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA BF16 embedding lookup kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_lm_head_argmax_bf16(const uint16_t*, const uint16_t*,
                                                         uint32_t*, float*, size_t, size_t,
                                                         size_t) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA BF16 LM-head argmax kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_lm_head_argmax_bf16_async(
    const uint16_t*, const uint16_t*, uint32_t*, float*, size_t, size_t, size_t, void*) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA BF16 LM-head argmax kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_lm_head_sample_topk_topp_bf16(
    const uint16_t*, const uint16_t*, const float*, uint32_t*, float*, size_t, size_t, size_t,
    float, size_t, float) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA BF16 LM-head top-k/top-p sampler kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_lm_head_sample_topk_topp_bf16_async(
    const uint16_t*, const uint16_t*, const float*, uint32_t*, float*, size_t, size_t, size_t,
    float, size_t, float, void*) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA BF16 LM-head top-k/top-p sampler kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_lm_head_argmax_sample_topk_topp_bf16_staged(
    const uint16_t*, const uint16_t*, const float*, uint32_t*, float*, uint32_t*, float*, float*,
    size_t, size_t, size_t, float, size_t, float) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA BF16 staged LM-head argmax sampler kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_lm_head_argmax_sample_topk_topp_bf16_staged_async(
    const uint16_t*, const uint16_t*, const float*, uint32_t*, float*, uint32_t*, float*, float*,
    size_t, size_t, size_t, float, size_t, float, void*) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA BF16 staged LM-head argmax sampler kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_lm_head_sample_topk_topp_bf16_cub(
    const uint16_t*, const uint16_t*, const float*, float*, float*, uint32_t*, uint32_t*, int*,
    uint32_t*, float*, void*, size_t, size_t, size_t, size_t, float, size_t, float) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA BF16 LM-head CUB top-k/top-p sampler kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_lm_head_sample_topk_topp_bf16_cub_async(
    const uint16_t*, const uint16_t*, const float*, float*, float*, uint32_t*, uint32_t*, int*,
    uint32_t*, float*, void*, size_t, size_t, size_t, size_t, float, size_t, float, void*) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA BF16 LM-head CUB top-k/top-p sampler kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_logits_argmax_f32(const float*, uint32_t*, float*, size_t,
                                                       size_t) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA logits argmax kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_logits_argmax_f32_async(const float*, uint32_t*, float*,
                                                             size_t, size_t, void*) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA logits argmax kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_logits_argmax_checked_f32_async(
    const float*, uint32_t*, float*, size_t, size_t, void*) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA checked logits argmax kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_logits_sample_topk_topp_f32(
    const float*, const float*, uint32_t*, float*, size_t, size_t, float, size_t, float) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA logits top-k/top-p sampler kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_logits_sample_topk_topp_f32_async(
    const float*, const float*, uint32_t*, float*, size_t, size_t, float, size_t, float, void*) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA logits top-k/top-p sampler kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_logits_sample_topk_topp_f32_cub(
    const float*, const float*, float*, uint32_t*, uint32_t*, int*, uint32_t*, float*, void*,
    size_t, size_t, size_t, float, size_t, float) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA CUB logits top-k/top-p sampler kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_logits_sample_topk_topp_f32_cub_async(
    const float*, const float*, float*, uint32_t*, uint32_t*, int*, uint32_t*, float*, void*,
    size_t, size_t, size_t, float, size_t, float, void*) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA CUB logits top-k/top-p sampler kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_pack_nibbles(const uint8_t*, uint8_t*, size_t) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE, "CUDA nibble pack kernel is unavailable in this build");
}

extern "C" cuteafd_status_t cuteafd_cuda_unpack_nibbles(const uint8_t*, uint8_t*, size_t) {
  return fail(CUTEAFD_STATUS_CUDA_UNAVAILABLE,
              "CUDA nibble unpack kernel is unavailable in this build");
}
#endif
