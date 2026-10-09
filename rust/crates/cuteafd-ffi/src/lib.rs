mod families;
mod shared;
pub use shared::{audio, fp8_moe, peer_exchange, programs, vision};
pub use shared::vocab_head::{VOCAB_HEAD_ROWS_MAX, VOCAB_HEAD_ROWS_PASS};
pub use shared::v41_device_ops::{V41Bf16Add, V41PeerCopy};
pub use families::deepseek_v41::v41_candidate_blocks::V41CandidateBlocks;
pub use families::deepseek_v41::v41_index_topk::V41IndexTopK;
pub use families::deepseek_v41::v41_sparse_attention::{V41SparseAttention, V41SparseBatch, V41SparseSource, V41SparseWindow};
pub use families::deepseek_v41::v41_index_scores::V41IndexScores;
pub use families::deepseek_v41::v41_kv::{V41Kv, V41KvStoreLayer, V41KvStoreLayers};
pub use families::deepseek_v41::v41_vision::V41VisionOps;
pub use families::deepseek_v41::v41_compressor::V41Compressor;
pub use families::deepseek_v41::v41_dspark_attention::{V41DsparkAttention,V41AttentionWindow};
pub use families::deepseek_v41::v41_grouped_output::{V41GroupedOutput, V41_GROUPED_OUTPUT_WORKSPACE};
pub use families::deepseek_v41::v41_attention_ops::V41AttentionOps;
pub use families::deepseek_v41::v41_dspark_cache::{V41DsparkCache, V41KvWrite};
pub use shared::v41_router::V41Router;
pub use families::deepseek_v41::v41_hc::V41Hc;
pub use families::deepseek_v41::v41_dspark::{V41DraftStep, V41DsparkConfidence, V41VocabularyProjection};
pub use families::deepseek_v41::v41_fp8::{V41Fp8Info, V41Fp8Kernel, V41SharedSwiGlu};
pub use families::deepseek_v41::v41_fp8_plan::{V41Fp8Plan, V41Fp8PlanInfo};
pub use shared::v41_exl3::{exl3_shard_widths, V41Exl3Info, V41Exl3Kernel, V41Exl3Layout, V41Exl3Routes};
pub use shared::v41_exl3_wire::V41Exl3Wire;
pub use shared::v41_experts::{
    V41ExpertInfo, V41ExpertInputQuantizer, V41ExpertKernel, V41ExpertLaunchArgs, V41ExpertPacker, V41ExpertPointer, V41ExpertOutputKind,
    V41CompactReducer, V41LocalExpertReducer, V41Tp2ExpertReducer, V41RouteReducer, V41_EXPERT_POINTER_COUNT,
    v41_pack_intermediate_supported, v41_rank_count_supported,
};
mod cuda_runtime;
pub mod memory_ledger;

static COORDINATOR_GPU_BUDGET: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static GPU_BUDGET_ALLOCATION: Mutex<()> = Mutex::new(());

/// Set once before coordinator startup. Worker processes never install this
/// ceiling; zero/unset preserves physical CUDA admission with no extra queries.
pub fn set_coordinator_gpu_budget(bytes: u64) -> Result<()> {
    anyhow::ensure!(bytes > 0, "coordinator GPU budget must be positive");
    COORDINATOR_GPU_BUDGET.compare_exchange(0, bytes, Ordering::Relaxed, Ordering::Relaxed)
        .map_err(|_| anyhow::anyhow!("coordinator GPU budget is already installed"))?;
    Ok(())
}

pub fn coordinator_gpu_budget() -> Option<cuteafd_core::serving_capacity::GpuMemoryBudget> {
    let bytes = COORDINATOR_GPU_BUDGET.load(Ordering::Relaxed);
    (bytes != 0).then_some(cuteafd_core::serving_capacity::GpuMemoryBudget(bytes))
}
pub use cuda_runtime::{select_copy_mechanism, CopyMechanism, CudaRuntime};
#[cfg(feature = "test-support")]
pub mod test_support;
#[cfg(all(target_os = "linux", any(test, feature = "test-support")))]
#[doc(hidden)]
pub mod native_library_lifetime_fixture;
#[cfg(all(test, target_os = "linux"))]
#[path = "native_library_lifetime_tests.rs"]
mod native_library_lifetime;

use anyhow::{Context, Result};
use libloading::{Library, Symbol};
use std::ffi::{CStr, CString};
use std::mem::ManuallyDrop;
use std::os::raw::{c_char, c_int, c_void};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, Ordering};

/// CUDA graph captures begun through this crate since the process started
/// (serving should reach zero new captures once warm; see AGENTS.md).
static GRAPH_CAPTURES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// CUDA graph captures begun so far (all libraries, all streams).
pub fn graph_captures() -> u64 {
    GRAPH_CAPTURES.load(std::sync::atomic::Ordering::Relaxed)
}

/// Captures begun per call site (`file:line` of the begin-capture call).
static GRAPH_CAPTURE_SITES: Mutex<Vec<(&'static std::panic::Location<'static>, u64)>> = Mutex::new(Vec::new());

/// Captures begun so far by call site, most first.
pub fn graph_capture_sites() -> Vec<(String, u64)> {
    let sites = GRAPH_CAPTURE_SITES.lock().unwrap_or_else(|p| p.into_inner());
    let mut sites: Vec<_> = sites.iter().map(|(l, n)| (format!("{}:{}", l.file(), l.line()), *n)).collect();
    sites.sort_by(|a, b| b.1.cmp(&a.1));
    sites
}

pub type CuteafdStatus = c_int;

pub const CUTEAFD_STATUS_OK: CuteafdStatus = 0;
pub const CUTEAFD_STATUS_CUDA_UNAVAILABLE: CuteafdStatus = 3;
pub const CUTEAFD_STATUS_RDMA_UNAVAILABLE: CuteafdStatus = 7;
pub const CUTEAFD_STATUS_NCCL_UNAVAILABLE: CuteafdStatus = 8;
pub const CUTEAFD_DEVICE_BUFFER_FLAG_HOST_FALLBACK: u64 = 1;
pub const CUTEAFD_DEVICE_BUFFER_FLAG_MANAGED: u64 = 2;
pub const CUTEAFD_DEVICE_BUFFER_FLAG_MAPPED_HOST: u64 = 4;
pub const CUTEAFD_HOST_BUFFER_FLAG_NONE: u64 = 0;
pub const CUTEAFD_HOST_BUFFER_FLAG_PINNED: u64 = 1;
pub const CUTEAFD_HOST_BUFFER_FLAG_HOST_FALLBACK: u64 = 2;
pub const CUTEAFD_HOST_BUFFER_FLAG_MAPPED: u64 = 4;
pub const CUTEAFD_ROUTE_SHARD_WIRE_BF16: u32 = 1;
pub const CUTEAFD_ROUTE_SHARD_WIRE_FP8_E4M3_ROW_SCALED: u32 = 2;
pub const CUTEAFD_ROUTE_SHARD_WIRE_NVFP4_E2M1_FP8_E4M3: u32 = 3;
pub const CUTEAFD_ROUTE_SHARD_LOCAL_F32: u32 = 1;
pub const CUTEAFD_ROUTE_SHARD_LOCAL_BF16: u32 = 2;
pub const CUTEAFD_CUDA_ROUTER_TOPK_MAX_K: usize = 64;
pub const CUTEAFD_CUDA_SAMPLE_TOPK_MAX_K: usize = 64;

/// Per-row status codes of the v4.1 GPU target-sampler
/// (`native/shared/cuda/sampling_gpu.h` §5.4). The integer values are the
/// device ABI, not an enum: they are what `out_status` carries.
pub const CUTEAFD_V41_SAMPLER_STATUS_OK: u32 = 0;
pub const CUTEAFD_V41_SAMPLER_STATUS_EMPTY_CANDIDATES: u32 = 1;
pub const CUTEAFD_V41_SAMPLER_STATUS_NONFINITE_LOGIT: u32 = 2;
pub const CUTEAFD_V41_SAMPLER_STATUS_INVALID_TEMPERATURE: u32 = 3;
pub const CUTEAFD_V41_SAMPLER_STATUS_MASK_WIDTH: u32 = 4;
pub const CUTEAFD_V41_SAMPLER_STATUS_INTERNAL: u32 = 5;

/// `out_status_detail` sentinel for "no detail". Token 0 is a real token, so
/// the device cannot use 0 as the absent marker; the host normalizes this to 0
/// before it is observable (approved design deviation, see the header).
pub const CUTEAFD_V41_SAMPLER_NO_DETAIL: u32 = u32::MAX;

/// Per-row flags of `cuteafd_sampler_row_t`.
pub const CUTEAFD_V41_SAMPLER_FLAG_GREEDY: u32 = 0x1;
pub const CUTEAFD_V41_SAMPLER_FLAG_DIAGNOSE: u32 = 0x2;
/// An unconstrained row: the kernel treats every token `< vocab` as allowed and
/// does not read the mask arena (approved redefinition of design §5.1 bit2).
pub const CUTEAFD_V41_SAMPLER_FLAG_NO_MASK: u32 = 0x4;
pub const CUTEAFD_V41_SAMPLER_FLAG_ORACLE_CROSSCHECK: u32 = 0x8;
/// Strict whole-row finiteness for a row that would otherwise take the
/// permissive stochastic branch. Greedy rows are strict unconditionally; the
/// host sets this bit on greedy and constrained rows only, and the validator
/// rejects it on a row that is neither.
pub const CUTEAFD_V41_SAMPLER_FLAG_STRICT_FINITE: u32 = 0x10;

/// Every flag bit this chunk's ABI defines.
pub const CUTEAFD_V41_SAMPLER_FLAG_KNOWN_MASK: u32 = CUTEAFD_V41_SAMPLER_FLAG_GREEDY
    | CUTEAFD_V41_SAMPLER_FLAG_DIAGNOSE
    | CUTEAFD_V41_SAMPLER_FLAG_NO_MASK
    | CUTEAFD_V41_SAMPLER_FLAG_ORACLE_CROSSCHECK
    | CUTEAFD_V41_SAMPLER_FLAG_STRICT_FINITE;

/// Per-row K1 scratch stride in bytes (design §11.1).
pub const CUTEAFD_V41_SAMPLER_SCRATCH_BYTES: usize = 64;

/// Size of the per-row parameter block in bytes (design §5.1).
pub const CUTEAFD_V41_SAMPLER_PARAM_BYTES: usize = 64;

/// 64-byte per-row parameter block, exactly as declared in
/// `native/shared/cuda/sampling_gpu.h`. Field order, sizes and natural
/// alignment are pinned by tests; `#[repr(C)]` is the ABI.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CuteafdV41SamplerRow {
    pub seed: u64,
    pub position: u64,
    pub temperature: f32,
    pub top_p: f32,
    pub min_p: f32,
    pub top_k: u32,
    pub mask_row: u32,
    pub flags: u32,
    pub output_row: u32,
    /// Host-precomputed `min_p.ln()`; the device never calls `logf`.
    pub ln_min_p: f32,
    pub reserved0: u32,
    pub reserved1: u32,
    pub reserved2: u64,
}

impl Default for CuteafdV41SamplerRow {
    fn default() -> Self {
        Self {
            seed: 0,
            position: 0,
            temperature: 0.0,
            top_p: 1.0,
            min_p: 0.0,
            top_k: 0,
            mask_row: CUTEAFD_V41_SAMPLER_NO_MASK_ROW,
            flags: 0,
            output_row: 0,
            ln_min_p: f32::NEG_INFINITY,
            reserved0: 0,
            reserved1: 0,
            reserved2: 0,
        }
    }
}

/// `mask_row` sentinel for an unconstrained row.
pub const CUTEAFD_V41_SAMPLER_NO_MASK_ROW: u32 = u32::MAX;

/// Packed mask words for a vocabulary: `ceil(vocab / 32)` (design §5.2).
pub const fn cuteafd_sampler_mask_words(vocab: usize) -> usize {
    vocab.div_ceil(32)
}

/// Clear every mask bit `>= vocab` in the final word before upload
/// (design §5.3 rule 2). A no-op when the vocabulary is a multiple of 32.
pub fn cuteafd_sampler_clear_remainder(words: &mut [u32], vocab: usize) {
    let remainder = vocab % 32;
    if remainder == 0 || vocab == 0 {
        return;
    }
    if let Some(last) = words.get_mut((vocab - 1) / 32) {
        *last &= (1u32 << remainder) - 1;
    }
}

pub const CUTEAFD_CUDA_MLA_FP8_DS_NOPE_VALUES: usize = 512;
pub const CUTEAFD_CUDA_MLA_FP8_DS_ROPE_VALUES: usize = 64;
pub const CUTEAFD_CUDA_MLA_FP8_DS_PROJECTED_VALUES: usize =
    CUTEAFD_CUDA_MLA_FP8_DS_NOPE_VALUES + CUTEAFD_CUDA_MLA_FP8_DS_ROPE_VALUES;
pub const CUTEAFD_CUDA_MLA_FP8_DS_SCALE_BYTES: usize = 16;
pub const CUTEAFD_CUDA_MLA_FP8_DS_PACKED_BYTES: usize = CUTEAFD_CUDA_MLA_FP8_DS_NOPE_VALUES
    + CUTEAFD_CUDA_MLA_FP8_DS_SCALE_BYTES
    + CUTEAFD_CUDA_MLA_FP8_DS_ROPE_VALUES * std::mem::size_of::<u16>();
/// Generic transactional-KV page width; DeepSeek target attention separately
/// owns a 256-token physical-page ABI.
pub const CUTEAFD_CUDA_GENERIC_KV_PAGE_SIZE: usize = 64;
pub const CUTEAFD_CUDA_MLA_MXFP4_DS_NOPE_VALUES: usize = 512;
pub const CUTEAFD_CUDA_MLA_MXFP4_DS_ROPE_VALUES: usize = 64;
pub const CUTEAFD_CUDA_MLA_MXFP4_DS_PROJECTED_VALUES: usize =
    CUTEAFD_CUDA_MLA_MXFP4_DS_NOPE_VALUES + CUTEAFD_CUDA_MLA_MXFP4_DS_ROPE_VALUES;
pub const CUTEAFD_CUDA_MLA_MXFP4_DS_BLOCK_SIZE: usize = 16;
pub const CUTEAFD_CUDA_MLA_MXFP4_DS_CODE_BYTES: usize = CUTEAFD_CUDA_MLA_MXFP4_DS_NOPE_VALUES / 2;
pub const CUTEAFD_CUDA_MLA_MXFP4_DS_SCALE_BYTES: usize =
    CUTEAFD_CUDA_MLA_MXFP4_DS_NOPE_VALUES / CUTEAFD_CUDA_MLA_MXFP4_DS_BLOCK_SIZE;
pub const CUTEAFD_CUDA_MLA_MXFP4_DS_PADDING_BYTES: usize = 16;
pub const CUTEAFD_CUDA_MLA_MXFP4_DS_PACKED_BYTES: usize = CUTEAFD_CUDA_MLA_MXFP4_DS_CODE_BYTES
    + CUTEAFD_CUDA_MLA_MXFP4_DS_SCALE_BYTES
    + CUTEAFD_CUDA_MLA_MXFP4_DS_PADDING_BYTES
    + CUTEAFD_CUDA_MLA_MXFP4_DS_ROPE_VALUES * std::mem::size_of::<u16>();

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CuteafdCudaDeviceInfo {
    pub device_id: c_int,
    pub cuda_available: c_int,
    pub compute_capability_major: c_int,
    pub compute_capability_minor: c_int,
    pub integrated: c_int,
    pub can_map_host_memory: c_int,
    pub unified_addressing: c_int,
    pub total_memory_bytes: u64,
    pub name: [c_char; 128],
    pub driver_version: [c_char; 64],
    pub runtime_version: [c_char; 64],
}

impl Default for CuteafdCudaDeviceInfo {
    fn default() -> Self {
        Self {
            device_id: 0,
            cuda_available: 0,
            compute_capability_major: 0,
            compute_capability_minor: 0,
            integrated: 0,
            can_map_host_memory: 0,
            unified_addressing: 0,
            total_memory_bytes: 0,
            name: [0; 128],
            driver_version: [0; 64],
            runtime_version: [0; 64],
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CuteafdDeviceBuffer {
    pub ptr: *mut c_void,
    pub bytes: usize,
    pub device_id: c_int,
    pub flags: u64,
}

impl Default for CuteafdDeviceBuffer {
    fn default() -> Self {
        Self {
            ptr: std::ptr::null_mut(),
            bytes: 0,
            device_id: -1,
            flags: 0,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct CuteafdRouteShardReductionBuffers {
    pub local: CuteafdDeviceBuffer,
    pub peers: [CuteafdDeviceBuffer; 3],
    pub output_f32: CuteafdDeviceBuffer,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CuteafdHostBuffer {
    pub ptr: *mut c_void,
    pub bytes: usize,
    pub flags: u64,
}

impl Default for CuteafdHostBuffer {
    fn default() -> Self {
        Self {
            ptr: std::ptr::null_mut(),
            bytes: 0,
            flags: CUTEAFD_HOST_BUFFER_FLAG_NONE,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct CuteafdNvfp4RouteBatchedMetadata {
    pub gate_weight: usize,
    pub gate_scale: usize,
    pub up_weight: usize,
    pub up_scale: usize,
    pub down_weight: usize,
    pub down_scale: usize,
    pub intermediate: usize,
    pub down_weight_row_stride_bytes: usize,
    pub down_scale_row_stride_bytes: usize,
    pub gate_scale_2: f32,
    pub up_scale_2: f32,
    pub down_scale_2: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct CuteafdDs4FlashSparkExl3K2MoeBuffers {
    pub input: CuteafdDeviceBuffer,
    pub w13_trellis: CuteafdDeviceBuffer,
    pub w2_trellis: CuteafdDeviceBuffer,
    pub gate_suh: CuteafdDeviceBuffer,
    pub up_suh: CuteafdDeviceBuffer,
    pub intermediate_rotations: CuteafdDeviceBuffer,
    pub down_svh: CuteafdDeviceBuffer,
    pub expert_map: CuteafdDeviceBuffer,
    pub dummy_scale: CuteafdDeviceBuffer,
    pub trellis_lut: CuteafdDeviceBuffer,
    pub global_scale: CuteafdDeviceBuffer,
    pub topk_ids: CuteafdDeviceBuffer,
    pub topk_weights: CuteafdDeviceBuffer,
    pub rotation_gate: CuteafdDeviceBuffer,
    pub rotation_up: CuteafdDeviceBuffer,
    pub fc1_output: CuteafdDeviceBuffer,
    pub activated: CuteafdDeviceBuffer,
    pub routed_output: CuteafdDeviceBuffer,
    pub output_f32: CuteafdDeviceBuffer,
    pub output_bf16: CuteafdDeviceBuffer,
    pub packed_route_indices: CuteafdDeviceBuffer,
    pub block_expert_ids: CuteafdDeviceBuffer,
    pub packed_route_count: CuteafdDeviceBuffer,
    pub expert_counts: CuteafdDeviceBuffer,
    pub expert_offsets: CuteafdDeviceBuffer,
    pub fc1_scratch: CuteafdDeviceBuffer,
    pub fc2_scratch: CuteafdDeviceBuffer,
    pub workspace: CuteafdDeviceBuffer,
}

// Flash and Pro use distinct native symbols and validation contracts but the
// caller-owned EXL3 buffer table has the same C layout for both profiles.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct CuteafdDs4FlashSharedExpertFp8Buffers {
    pub input: CuteafdDeviceBuffer,
    pub w1_weight: CuteafdDeviceBuffer,
    pub w1_scale_mma: CuteafdDeviceBuffer,
    pub w3_weight: CuteafdDeviceBuffer,
    pub w3_scale_mma: CuteafdDeviceBuffer,
    pub w2_weight: CuteafdDeviceBuffer,
    pub w2_scale_mma: CuteafdDeviceBuffer,
    pub gate: CuteafdDeviceBuffer,
    pub up: CuteafdDeviceBuffer,
    pub activated: CuteafdDeviceBuffer,
    pub output: CuteafdDeviceBuffer,
    pub alpha: CuteafdDeviceBuffer,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct CuteafdCudaGraphCaptureInfo {
    pub graph: *mut c_void,
    pub graph_exec: *mut c_void,
    pub node_count: usize,
    pub kernel_node_count: usize,
    pub memcpy_node_count: usize,
    pub memset_node_count: usize,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct CuteafdBf16Summary {
    pub checksum: f64,
    pub values: u64,
    pub finite_values: u64,
    pub nonzero_values: u64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CuteafdRdmaDeviceInfo {
    pub rdma_enabled: c_int,
    pub device_count: c_int,
    pub first_device_openable: c_int,
    pub first_device_guid: u64,
    pub first_device_name: [c_char; 128],
    pub first_device_transport: [c_char; 64],
    pub status: [c_char; 128],
}

impl Default for CuteafdRdmaDeviceInfo {
    fn default() -> Self {
        Self {
            rdma_enabled: 0,
            device_count: 0,
            first_device_openable: 0,
            first_device_guid: 0,
            first_device_name: [0; 128],
            first_device_transport: [0; 64],
            status: [0; 128],
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CuteafdRdmaHostBufferPlan {
    pub original_addr: usize,
    pub original_bytes: usize,
    pub alignment: usize,
    pub registered_addr: usize,
    pub prefix_bytes: usize,
    pub registered_span_bytes: usize,
    pub span_aligned: c_int,
    pub rdma_enabled: c_int,
}

impl Default for CuteafdRdmaHostBufferPlan {
    fn default() -> Self {
        Self {
            original_addr: 0,
            original_bytes: 0,
            alignment: 0,
            registered_addr: 0,
            prefix_bytes: 0,
            registered_span_bytes: 0,
            span_aligned: 0,
            rdma_enabled: 0,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CuteafdRdmaRegisterProbe {
    pub bytes: usize,
    pub registered: c_int,
    pub lkey: u32,
    pub rkey: u32,
    pub device_name: [c_char; 128],
}

impl Default for CuteafdRdmaRegisterProbe {
    fn default() -> Self {
        Self {
            bytes: 0,
            registered: 0,
            lkey: 0,
            rkey: 0,
            device_name: [0; 128],
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CuteafdRdmaRcQpProbe {
    pub rdma_enabled: c_int,
    pub created: c_int,
    pub port_num: u32,
    pub qp_num: u32,
    pub lid: u32,
    pub active_mtu: u32,
    pub requested_send_wr: u32,
    pub requested_recv_wr: u32,
    pub requested_max_sge: u32,
    pub actual_max_send_wr: u32,
    pub actual_max_recv_wr: u32,
    pub actual_max_send_sge: u32,
    pub actual_max_recv_sge: u32,
    pub actual_max_inline_data: u32,
    pub device_name: [c_char; 128],
    pub status: [c_char; 128],
}

impl Default for CuteafdRdmaRcQpProbe {
    fn default() -> Self {
        Self {
            rdma_enabled: 0,
            created: 0,
            port_num: 0,
            qp_num: 0,
            lid: 0,
            active_mtu: 0,
            requested_send_wr: 0,
            requested_recv_wr: 0,
            requested_max_sge: 0,
            actual_max_send_wr: 0,
            actual_max_recv_wr: 0,
            actual_max_send_sge: 0,
            actual_max_recv_sge: 0,
            actual_max_inline_data: 0,
            device_name: [0; 128],
            status: [0; 128],
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CuteafdRdmaRcSendRecvProbe {
    pub rdma_enabled: c_int,
    pub completed: c_int,
    pub payload_matches: c_int,
    pub port_num: u32,
    pub bytes: usize,
    pub sender_qp_num: u32,
    pub receiver_qp_num: u32,
    pub send_completions: u32,
    pub recv_completions: u32,
    pub poll_iterations: u32,
    pub device_name: [c_char; 128],
    pub status: [c_char; 128],
}

impl Default for CuteafdRdmaRcSendRecvProbe {
    fn default() -> Self {
        Self {
            rdma_enabled: 0,
            completed: 0,
            payload_matches: 0,
            port_num: 0,
            bytes: 0,
            sender_qp_num: 0,
            receiver_qp_num: 0,
            send_completions: 0,
            recv_completions: 0,
            poll_iterations: 0,
            device_name: [0; 128],
            status: [0; 128],
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CuteafdRdmaRcProtocolV2LoopbackProbe {
    pub rdma_enabled: c_int,
    pub completed: c_int,
    pub request_payload_matches: c_int,
    pub response_payload_matches: c_int,
    pub port_num: u32,
    pub request_bytes: usize,
    pub response_bytes: usize,
    pub client_qp_num: u32,
    pub server_qp_num: u32,
    pub send_completions: u32,
    pub recv_completions: u32,
    pub poll_iterations: u32,
    pub device_name: [c_char; 128],
    pub status: [c_char; 128],
}

impl Default for CuteafdRdmaRcProtocolV2LoopbackProbe {
    fn default() -> Self {
        Self {
            rdma_enabled: 0,
            completed: 0,
            request_payload_matches: 0,
            response_payload_matches: 0,
            port_num: 0,
            request_bytes: 0,
            response_bytes: 0,
            client_qp_num: 0,
            server_qp_num: 0,
            send_completions: 0,
            recv_completions: 0,
            poll_iterations: 0,
            device_name: [0; 128],
            status: [0; 128],
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CuteafdRdmaRcEndpointInfo {
    pub rdma_enabled: c_int,
    pub handle: *mut c_void,
    pub port_num: u32,
    pub qp_num: u32,
    pub psn: u32,
    pub lid: u32,
    pub active_mtu: u32,
    pub send_frame_bytes: usize,
    pub recv_frame_bytes: usize,
    pub send_registered_span_bytes: usize,
    pub recv_registered_span_bytes: usize,
    pub max_send_wr: u32,
    pub max_recv_wr: u32,
    pub max_sge: u32,
    pub gid_hex: [c_char; 33],
    pub device_name: [c_char; 128],
    pub status: [c_char; 128],
}

impl Default for CuteafdRdmaRcEndpointInfo {
    fn default() -> Self {
        Self {
            rdma_enabled: 0,
            handle: std::ptr::null_mut(),
            port_num: 0,
            qp_num: 0,
            psn: 0,
            lid: 0,
            active_mtu: 0,
            send_frame_bytes: 0,
            recv_frame_bytes: 0,
            send_registered_span_bytes: 0,
            recv_registered_span_bytes: 0,
            max_send_wr: 0,
            max_recv_wr: 0,
            max_sge: 0,
            gid_hex: [0; 33],
            device_name: [0; 128],
            status: [0; 128],
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CuteafdRdmaRcEndpointBufferView {
    pub host_ptr: *mut c_void,
    pub device_ptr: *mut c_void,
    pub bytes: usize,
    pub device_id: c_int,
    pub host_flags: u64,
}

impl Default for CuteafdRdmaRcEndpointBufferView {
    fn default() -> Self {
        Self {
            host_ptr: std::ptr::null_mut(),
            device_ptr: std::ptr::null_mut(),
            bytes: 0,
            device_id: -1,
            host_flags: CUTEAFD_HOST_BUFFER_FLAG_NONE,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CuteafdRdmaRcCompletionStats {
    pub expected_send_completions: u32,
    pub expected_recv_completions: u32,
    pub send_completions: u32,
    pub recv_completions: u32,
    pub poll_iterations: u32,
    pub status: [c_char; 128],
}

/// `cuteafd_rdma_gpu_landing_probe_t`: dma-buf GPU landing support and a
/// loopback measurement of landing in device vs pinned host memory.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CuteafdRdmaGpuLandingProbe {
    pub cuda_device: i32,
    pub dma_buf_supported: i32,
    pub gpudirect_rdma_supported: i32,
    pub writes_ordering: i32,
    pub registered: i32,
    pub gpu_gbps: f64,
    pub host_gbps: f64,
    pub device_name: [c_char; 64],
    pub status: [c_char; 256],
}

impl Default for CuteafdRdmaGpuLandingProbe {
    fn default() -> Self {
        Self {
            cuda_device: -1,
            dma_buf_supported: 0,
            gpudirect_rdma_supported: 0,
            writes_ordering: -1,
            registered: 0,
            gpu_gbps: 0.0,
            host_gbps: 0.0,
            device_name: [0; 64],
            status: [0; 256],
        }
    }
}

impl Default for CuteafdRdmaRcCompletionStats {
    fn default() -> Self {
        Self {
            expected_send_completions: 0,
            expected_recv_completions: 0,
            send_completions: 0,
            recv_completions: 0,
            poll_iterations: 0,
            status: [0; 128],
        }
    }
}

type VersionFn = unsafe extern "C" fn(out: *mut c_char, out_len: usize) -> CuteafdStatus;
type AllocHostBufferFn =
    unsafe extern "C" fn(bytes: usize, out: *mut CuteafdHostBuffer) -> CuteafdStatus;
type CudaHostBufferDeviceAliasFn =
    unsafe extern "C" fn(host: CuteafdHostBuffer, out: *mut CuteafdDeviceBuffer) -> CuteafdStatus;
type FreeHostBufferFn = unsafe extern "C" fn(buf: *mut CuteafdHostBuffer) -> CuteafdStatus;
type AllocDeviceBufferFn =
    unsafe extern "C" fn(bytes: usize, out: *mut CuteafdDeviceBuffer) -> CuteafdStatus;
type AllocManagedDeviceBufferFn =
    unsafe extern "C" fn(bytes: usize, out: *mut CuteafdDeviceBuffer) -> CuteafdStatus;
type FreeDeviceBufferFn = unsafe extern "C" fn(buf: *mut CuteafdDeviceBuffer) -> CuteafdStatus;
type CudaStreamCreateFn = unsafe extern "C" fn(out: *mut *mut c_void) -> CuteafdStatus;
type CudaStreamDestroyFn = unsafe extern "C" fn(cuda_stream: *mut c_void) -> CuteafdStatus;
type CudaStreamSynchronizeFn = unsafe extern "C" fn(cuda_stream: *mut c_void) -> CuteafdStatus;
type CudaStreamQueryFn = unsafe extern "C" fn(cuda_stream: *mut c_void, ready: *mut i32) -> CuteafdStatus;
type CudaStreamWaitEventFn =
    unsafe extern "C" fn(cuda_stream: *mut c_void, cuda_event: *mut c_void) -> CuteafdStatus;
type CudaEventCreateFn = unsafe extern "C" fn(out: *mut *mut c_void) -> CuteafdStatus;
type CudaEventDestroyFn = unsafe extern "C" fn(cuda_event: *mut c_void) -> CuteafdStatus;
type CudaEventRecordFn =
    unsafe extern "C" fn(cuda_event: *mut c_void, cuda_stream: *mut c_void) -> CuteafdStatus;
type CudaEventSynchronizeFn = unsafe extern "C" fn(cuda_event: *mut c_void) -> CuteafdStatus;
type CudaEventElapsedMsFn = unsafe extern "C" fn(
    start_event: *mut c_void,
    end_event: *mut c_void,
    out_ms: *mut f32,
) -> CuteafdStatus;
type CudaGraphBeginCaptureFn = unsafe extern "C" fn(cuda_stream: *mut c_void) -> CuteafdStatus;
type CudaGraphEndCaptureFn = unsafe extern "C" fn(
    cuda_stream: *mut c_void,
    out_cuda_graph_exec: *mut *mut c_void,
) -> CuteafdStatus;
type CudaGraphLaunchFn =
    unsafe extern "C" fn(cuda_graph_exec: *mut c_void, cuda_stream: *mut c_void) -> CuteafdStatus;
type CudaGraphExecDestroyFn = unsafe extern "C" fn(cuda_graph_exec: *mut c_void) -> CuteafdStatus;
type CopyH2DFn =
    unsafe extern "C" fn(dst: CuteafdDeviceBuffer, src: *const c_void, bytes: usize) -> CuteafdStatus;
type CopyD2HFn =
    unsafe extern "C" fn(dst: *mut c_void, src: CuteafdDeviceBuffer, bytes: usize) -> CuteafdStatus;
type CopyD2DFn = unsafe extern "C" fn(
    dst: CuteafdDeviceBuffer,
    src: CuteafdDeviceBuffer,
    bytes: usize,
) -> CuteafdStatus;
type CopyH2DAsyncFn = unsafe extern "C" fn(
    dst: CuteafdDeviceBuffer,
    src: *const c_void,
    bytes: usize,
    cuda_stream: *mut c_void,
) -> CuteafdStatus;
type CopyH2DBatchAsyncFn = unsafe extern "C" fn(
    dsts: *const CuteafdDeviceBuffer,
    srcs: *const *const c_void,
    bytes: *const usize,
    count: usize,
    cuda_stream: *mut c_void,
) -> CuteafdStatus;
type CopyH2D2DAsyncFn = unsafe extern "C" fn(
    dst: CuteafdDeviceBuffer,
    dst_pitch_bytes: usize,
    src: *const c_void,
    src_pitch_bytes: usize,
    width_bytes: usize,
    rows: usize,
    cuda_stream: *mut c_void,
) -> CuteafdStatus;
type CopyD2HAsyncFn = unsafe extern "C" fn(
    dst: *mut c_void,
    src: CuteafdDeviceBuffer,
    bytes: usize,
    cuda_stream: *mut c_void,
) -> CuteafdStatus;
type CopyD2DAsyncFn = unsafe extern "C" fn(
    dst: CuteafdDeviceBuffer,
    src: CuteafdDeviceBuffer,
    bytes: usize,
    cuda_stream: *mut c_void,
) -> CuteafdStatus;
type LastErrorFn = unsafe extern "C" fn(out: *mut c_char, out_len: usize) -> CuteafdStatus;
type NcclCommDestroyFn = unsafe extern "C" fn(handle: *mut c_void) -> CuteafdStatus;
type CudaRmsNormBf16AsyncFn = unsafe extern "C" fn(
    x: *const u16,
    weight: *const u16,
    out: *mut u16,
    rows: c_int,
    hidden: c_int,
    eps: f32,
    cuda_stream: *mut c_void,
) -> CuteafdStatus;
type CudaDs4RmsNormBf16RneAsyncFn = CudaRmsNormBf16AsyncFn;

type CudaNvfp4SwizzleScaleAsyncFn = unsafe extern "C" fn(
    CuteafdDeviceBuffer,
    CuteafdDeviceBuffer,
    usize,
    usize,
    *mut c_void,
) -> CuteafdStatus;
type CudaNvfp4PadExpertAsyncFn = unsafe extern "C" fn(
    *const CuteafdDeviceBuffer,
    *const CuteafdDeviceBuffer,
    usize,
    usize,
    *mut c_void,
) -> CuteafdStatus;
type CudaZeroBytesFn = unsafe extern "C" fn(dst: *mut c_void, bytes: usize) -> CuteafdStatus;
type CudaZeroBytesAsyncFn =
    unsafe extern "C" fn(dst: *mut c_void, bytes: usize, cuda_stream: *mut c_void) -> CuteafdStatus;
type CudaF32ToBf16Fn =
    unsafe extern "C" fn(src: *const f32, dst: *mut u16, count: usize) -> CuteafdStatus;
type CudaF32ToBf16AsyncFn = unsafe extern "C" fn(
    src: *const f32,
    dst: *mut u16,
    count: usize,
    cuda_stream: *mut c_void,
) -> CuteafdStatus;
type CudaGatherRowsF32ToFp8E4m3RowScaledAsyncFn = unsafe extern "C" fn(
    src: *const f32,
    row_indices: *const u32,
    dst: *mut u8,
    rows: usize,
    row_width: usize,
    dst_row_stride_bytes: usize,
    cuda_stream: *mut c_void,
) -> CuteafdStatus;
type CudaGatherRowsF32ToNvfp4E2m1Fp8E4m3AsyncFn = unsafe extern "C" fn(
    src: *const f32,
    row_indices: *const u32,
    dst: *mut u8,
    rows: usize,
    row_width: usize,
    dst_row_stride_bytes: usize,
    cuda_stream: *mut c_void,
) -> CuteafdStatus;
type CudaReduceRouteShardsToF32AsyncFn = unsafe extern "C" fn(
    buffers: *const CuteafdRouteShardReductionBuffers,
    rows: usize,
    row_width: usize,
    peer_row_stride_bytes: usize,
    local_dtype: u32,
    peer_dtype: u32,
    peer_count: u32,
    cuda_stream: *mut c_void,
) -> CuteafdStatus;
type CudaEmbeddingLookupBf16AsyncFn = unsafe extern "C" fn(
    embedding: *const u16,
    token_ids: *const u32,
    out: *mut u16,
    rows: usize,
    vocab: usize,
    hidden: usize,
    cuda_stream: *mut c_void,
) -> CuteafdStatus;
/// K1 entry points of the v4.1 GPU target-sampler
/// (`native/shared/cuda/sampling_gpu.h`).
type CudaV41TargetSampleFn = unsafe extern "C" fn(
    logits: *const f32,
    rows: usize,
    vocab: usize,
    logits_stride: usize,
    params: *const CuteafdV41SamplerRow,
    mask_words: *const u32,
    mask_words_per_row: usize,
    out_indices: *mut u32,
    out_status: *mut u32,
    out_status_detail: *mut u32,
    out_scores: *mut f32,
    out_total: *mut f32,
    out_nucleus_count: *mut u32,
    scratch: *mut c_void,
) -> CuteafdStatus;
type CudaV41TargetSampleAsyncFn = unsafe extern "C" fn(
    logits: *const f32,
    rows: usize,
    vocab: usize,
    logits_stride: usize,
    params: *const CuteafdV41SamplerRow,
    mask_words: *const u32,
    mask_words_per_row: usize,
    out_indices: *mut u32,
    out_status: *mut u32,
    out_status_detail: *mut u32,
    out_scores: *mut f32,
    out_total: *mut f32,
    out_nucleus_count: *mut u32,
    scratch: *mut c_void,
    cuda_stream: *mut c_void,
) -> CuteafdStatus;
/// Chunk-3a K3/K4 entry points of the v4.1 GPU target-sampler
/// (`native/shared/cuda/sampling_gpu.h`). They read K1's `scratch` and
/// materialize the retained set in CPU rank order into the rank-order arena.
type CudaV41TopkSelectFn = unsafe extern "C" fn(
    logits: *const f32,
    rows: usize,
    vocab: usize,
    logits_stride: usize,
    params: *const CuteafdV41SamplerRow,
    mask_words: *const u32,
    mask_words_per_row: usize,
    rank_order_ids: *mut u32,
    rank_order_scratch: *mut u64,
    rank_order_capacity: usize,
    out_retained_count: *mut u32,
    out_pivot_passes: *mut u32,
    scratch: *mut c_void,
) -> CuteafdStatus;
type CudaV41TopkSelectAsyncFn = unsafe extern "C" fn(
    logits: *const f32,
    rows: usize,
    vocab: usize,
    logits_stride: usize,
    params: *const CuteafdV41SamplerRow,
    mask_words: *const u32,
    mask_words_per_row: usize,
    rank_order_ids: *mut u32,
    rank_order_scratch: *mut u64,
    rank_order_capacity: usize,
    out_retained_count: *mut u32,
    out_pivot_passes: *mut u32,
    scratch: *mut c_void,
    cuda_stream: *mut c_void,
) -> CuteafdStatus;
/// Chunk-3b K5 nucleus/draw entry point (`cuteafd_cuda_v41_nucleus`).
type CudaV41NucleusFn = unsafe extern "C" fn(
    logits: *const f32,
    rows: usize,
    vocab: usize,
    logits_stride: usize,
    params: *const CuteafdV41SamplerRow,
    mask_words: *const u32,
    mask_words_per_row: usize,
    rank_order_ids: *const u32,
    rank_order_capacity: usize,
    rank_retained_count: *const u32,
    out_indices: *mut u32,
    out_status: *mut u32,
    out_total: *mut f32,
    out_nucleus_count: *mut u32,
    scratch: *mut c_void,
) -> CuteafdStatus;
type CudaV41NucleusAsyncFn = unsafe extern "C" fn(
    logits: *const f32,
    rows: usize,
    vocab: usize,
    logits_stride: usize,
    params: *const CuteafdV41SamplerRow,
    mask_words: *const u32,
    mask_words_per_row: usize,
    rank_order_ids: *const u32,
    rank_order_capacity: usize,
    rank_retained_count: *const u32,
    out_indices: *mut u32,
    out_status: *mut u32,
    out_total: *mut f32,
    out_nucleus_count: *mut u32,
    scratch: *mut c_void,
    cuda_stream: *mut c_void,
) -> CuteafdStatus;
type CudaLogitsArgmaxF32AsyncFn = unsafe extern "C" fn(
    logits: *const f32,
    out_indices: *mut u32,
    out_scores: *mut f32,
    rows: usize,
    vocab: usize,
    cuda_stream: *mut c_void,
) -> CuteafdStatus;
type RdmaDeviceInfoFn = unsafe extern "C" fn(out: *mut CuteafdRdmaDeviceInfo) -> CuteafdStatus;
type RdmaPlanHostBufferRegistrationFn = unsafe extern "C" fn(
    ptr: *const c_void,
    bytes: usize,
    alignment: usize,
    out: *mut CuteafdRdmaHostBufferPlan,
) -> CuteafdStatus;
type RdmaRegisterHostBufferProbeFn = unsafe extern "C" fn(
    ptr: *mut c_void,
    bytes: usize,
    out: *mut CuteafdRdmaRegisterProbe,
) -> CuteafdStatus;
type RdmaCreateRcQpProbeFn = unsafe extern "C" fn(
    port_num: u32,
    send_wr: u32,
    recv_wr: u32,
    max_sge: u32,
    out: *mut CuteafdRdmaRcQpProbe,
) -> CuteafdStatus;
type RdmaRcSendRecvLoopbackProbeFn = unsafe extern "C" fn(
    port_num: u32,
    bytes: usize,
    out: *mut CuteafdRdmaRcSendRecvProbe,
) -> CuteafdStatus;
type RdmaRcProtocolV2LoopbackProbeFn = unsafe extern "C" fn(
    port_num: u32,
    request_frame: *const c_void,
    request_bytes: usize,
    response_frame: *const c_void,
    response_bytes: usize,
    out: *mut CuteafdRdmaRcProtocolV2LoopbackProbe,
) -> CuteafdStatus;
type RdmaRcEndpointCreateFn = unsafe extern "C" fn(
    port_num: u32,
    local_psn: u32,
    send_frame_bytes: usize,
    recv_frame_bytes: usize,
    send_registered_span_bytes: usize,
    recv_registered_span_bytes: usize,
    max_send_wr: u32,
    max_recv_wr: u32,
    max_sge: u32,
    out: *mut CuteafdRdmaRcEndpointInfo,
) -> CuteafdStatus;
type RdmaRcEndpointCreateWithBufferFlagsFn = unsafe extern "C" fn(
    port_num: u32,
    local_psn: u32,
    send_frame_bytes: usize,
    recv_frame_bytes: usize,
    send_registered_span_bytes: usize,
    recv_registered_span_bytes: usize,
    max_send_wr: u32,
    max_recv_wr: u32,
    max_sge: u32,
    host_buffer_flags: u64,
    out: *mut CuteafdRdmaRcEndpointInfo,
) -> CuteafdStatus;
type RdmaRcEndpointCreateOnDeviceWithBufferFlagsFn = unsafe extern "C" fn(
    device_name: *const c_char,
    port_num: u32,
    local_psn: u32,
    send_frame_bytes: usize,
    recv_frame_bytes: usize,
    send_registered_span_bytes: usize,
    recv_registered_span_bytes: usize,
    max_send_wr: u32,
    max_recv_wr: u32,
    max_sge: u32,
    host_buffer_flags: u64,
    out: *mut CuteafdRdmaRcEndpointInfo,
) -> CuteafdStatus;
type RdmaRcEndpointCreateOnGidWithBufferFlagsFn = unsafe extern "C" fn(
    device_name: *const c_char,
    port_num: u32,
    gid_index: u32,
    local_psn: u32,
    send_frame_bytes: usize,
    recv_frame_bytes: usize,
    send_registered_span_bytes: usize,
    recv_registered_span_bytes: usize,
    max_send_wr: u32,
    max_recv_wr: u32,
    max_sge: u32,
    host_buffer_flags: u64,
    out: *mut CuteafdRdmaRcEndpointInfo,
) -> CuteafdStatus;
type RdmaRcEndpointBufferViewFn = unsafe extern "C" fn(
    handle: *mut c_void,
    receive_buffer: c_int,
    out: *mut CuteafdRdmaRcEndpointBufferView,
) -> CuteafdStatus;
type RdmaRcEndpointConnectFn = unsafe extern "C" fn(
    handle: *mut c_void,
    remote_qp_num: u32,
    remote_psn: u32,
    remote_lid: u32,
    remote_gid_hex: *const c_char,
) -> CuteafdStatus;
type RdmaRcEndpointPostRecvFn =
    unsafe extern "C" fn(handle: *mut c_void, bytes: usize, wr_id: u64) -> CuteafdStatus;
type RdmaRcEndpointPostRecvAtFn = unsafe extern "C" fn(
    handle: *mut c_void,
    offset_bytes: usize,
    bytes: usize,
    wr_id: u64,
) -> CuteafdStatus;
type RdmaRcEndpointSetRecvLandingFn = unsafe extern "C" fn(
    handle: *mut c_void,
    device_ptr: *mut c_void,
    bytes: usize,
    header_bytes: usize,
) -> CuteafdStatus;
type RdmaGpuLandingProbeFn = unsafe extern "C" fn(
    device_name: *const c_char,
    port_num: u32,
    bytes: usize,
    iterations: u32,
    out: *mut CuteafdRdmaGpuLandingProbe,
) -> CuteafdStatus;
type RdmaRcEndpointPostSendAtFn = unsafe extern "C" fn(
    handle: *mut c_void,
    offset_bytes: usize,
    bytes: usize,
    wr_id: u64,
) -> CuteafdStatus;
type RdmaRcEndpointSendFn = unsafe extern "C" fn(
    handle: *mut c_void,
    frame: *const c_void,
    bytes: usize,
    wr_id: u64,
) -> CuteafdStatus;
type RdmaRcEndpointSendAtFn = unsafe extern "C" fn(
    handle: *mut c_void,
    frame: *const c_void,
    offset_bytes: usize,
    bytes: usize,
    wr_id: u64,
) -> CuteafdStatus;
type RdmaRcEndpointSendPartsAtFn = unsafe extern "C" fn(
    handle: *mut c_void,
    prefix: *const c_void,
    prefix_bytes: usize,
    payload: *const c_void,
    payload_bytes: usize,
    offset_bytes: usize,
    wr_id: u64,
) -> CuteafdStatus;
type RdmaRcEndpointPollFn = unsafe extern "C" fn(
    handle: *mut c_void,
    expected_send_completions: u32,
    expected_recv_completions: u32,
    max_poll_iterations: u32,
    out: *mut CuteafdRdmaRcCompletionStats,
) -> CuteafdStatus;
type RdmaRcEndpointPollWithTimeoutFn = unsafe extern "C" fn(
    handle: *mut c_void,
    expected_send_completions: u32,
    expected_recv_completions: u32,
    max_poll_iterations: u32,
    active_event_poll_timeout_ms: u32,
    out: *mut CuteafdRdmaRcCompletionStats,
) -> CuteafdStatus;
type RdmaRcEndpointTryPollFn = unsafe extern "C" fn(
    handle: *mut c_void,
    max_send_completions: u32,
    max_recv_completions: u32,
    out: *mut CuteafdRdmaRcCompletionStats,
) -> CuteafdStatus;
type RdmaRcEndpointCopyRecvFn = unsafe extern "C" fn(
    handle: *mut c_void,
    out: *mut c_void,
    out_bytes: usize,
    bytes: usize,
) -> CuteafdStatus;
type RdmaRcEndpointCopyRecvAtFn = unsafe extern "C" fn(
    handle: *mut c_void,
    out: *mut c_void,
    out_bytes: usize,
    offset_bytes: usize,
    bytes: usize,
) -> CuteafdStatus;
type RdmaRcEndpointDestroyFn = unsafe extern "C" fn(handle: *mut c_void) -> CuteafdStatus;

type XGrammarCompilerCreateFn = unsafe extern "C" fn(
    tokenizer_json_path: *const c_char,
    vocab_size: usize,
    stop_token_ids: *const i32,
    stop_token_count: usize,
    out_compiler: *mut *mut c_void,
    error: *mut c_char,
    error_bytes: usize,
) -> CuteafdStatus;
type XGrammarCompilerDestroyFn = unsafe extern "C" fn(compiler: *mut c_void) -> CuteafdStatus;
type XGrammarCompileFn = unsafe extern "C" fn(
    compiler: *mut c_void,
    kind: c_int,
    grammar_json: *const c_char,
    strict: c_int,
    out_grammar: *mut *mut c_void,
    error: *mut c_char,
    error_bytes: usize,
) -> CuteafdStatus;
type XGrammarGrammarDestroyFn = unsafe extern "C" fn(grammar: *mut c_void) -> CuteafdStatus;
type XGrammarMatcherCreateFn = unsafe extern "C" fn(
    grammar: *const c_void,
    out_matcher: *mut *mut c_void,
    error: *mut c_char,
    error_bytes: usize,
) -> CuteafdStatus;
type XGrammarMatcherForkFn = unsafe extern "C" fn(
    matcher: *const c_void,
    out_matcher: *mut *mut c_void,
    error: *mut c_char,
    error_bytes: usize,
) -> CuteafdStatus;
type XGrammarMatcherDestroyFn = unsafe extern "C" fn(matcher: *mut c_void) -> CuteafdStatus;
type XGrammarMatcherFillBitmaskFn = unsafe extern "C" fn(
    matcher: *mut c_void,
    bitmask: *mut u32,
    bitmask_words: usize,
    out_needs_mask: *mut c_int,
    error: *mut c_char,
    error_bytes: usize,
) -> CuteafdStatus;
type XGrammarMatcherAcceptTokenFn = unsafe extern "C" fn(
    matcher: *mut c_void,
    token_id: u32,
    out_accepted: *mut c_int,
    error: *mut c_char,
    error_bytes: usize,
) -> CuteafdStatus;
type XGrammarMatcherIsCompletedFn = unsafe extern "C" fn(
    matcher: *const c_void,
    out_completed: *mut c_int,
    error: *mut c_char,
    error_bytes: usize,
) -> CuteafdStatus;

pub struct NativeLibrary {
    lib: ManuallyDrop<Library>,
    quarantine_after_failed_drain: AtomicBool,
    sync_h2d_staging: Mutex<SyncH2DStagingBuffer>,
    rdma_rc_endpoint_try_poll_fn: RdmaRcEndpointTryPollFn,
}

pub const CUTEAFD_XGRAMMAR_JSON_OBJECT: c_int = 1;
pub const CUTEAFD_XGRAMMAR_JSON_SCHEMA: c_int = 2;
pub const CUTEAFD_XGRAMMAR_STRUCTURAL_TAG: c_int = 3;

pub struct CuteafdXGrammarCompiler<'a> {
    handle: *mut c_void,
    library: &'a NativeLibrary,
}

pub struct CuteafdXGrammarGrammar<'a> {
    handle: *mut c_void,
    library: &'a NativeLibrary,
}

pub struct CuteafdXGrammarMatcher<'a> {
    handle: *mut c_void,
    library: &'a NativeLibrary,
}

unsafe impl Send for CuteafdXGrammarCompiler<'_> {}
unsafe impl Sync for CuteafdXGrammarCompiler<'_> {}
unsafe impl Send for CuteafdXGrammarGrammar<'_> {}
unsafe impl Sync for CuteafdXGrammarGrammar<'_> {}
unsafe impl Send for CuteafdXGrammarMatcher<'_> {}

const XGRAMMAR_ERROR_BYTES: usize = 2_048;

fn xgrammar_status(
    context: &str,
    status: CuteafdStatus,
    error: &[c_char; XGRAMMAR_ERROR_BYTES],
) -> Result<()> {
    if status == CUTEAFD_STATUS_OK {
        return Ok(());
    }
    let detail = unsafe { CStr::from_ptr(error.as_ptr()) }
        .to_string_lossy()
        .into_owned();
    if detail.is_empty() {
        anyhow::bail!("{context} returned status {status}");
    }
    anyhow::bail!("{context} returned status {status}: {detail}")
}

impl NativeLibrary {
    pub fn xgrammar_compiler<'a>(
        &'a self,
        tokenizer_json_path: &Path,
        vocab_size: usize,
        stop_token_ids: &[i32],
    ) -> Result<CuteafdXGrammarCompiler<'a>> {
        let path = CString::new(tokenizer_json_path.to_string_lossy().as_bytes())
            .context("XGrammar tokenizer path contains a NUL byte")?;
        let create: Symbol<XGrammarCompilerCreateFn> =
            unsafe { self.lib.get(b"cuteafd_xgrammar_compiler_create")? };
        let mut handle = std::ptr::null_mut();
        let mut error = [0 as c_char; XGRAMMAR_ERROR_BYTES];
        let status = unsafe {
            create(
                path.as_ptr(),
                vocab_size,
                if stop_token_ids.is_empty() {
                    std::ptr::null()
                } else {
                    stop_token_ids.as_ptr()
                },
                stop_token_ids.len(),
                &mut handle,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        xgrammar_status("cuteafd_xgrammar_compiler_create", status, &error)?;
        anyhow::ensure!(!handle.is_null(), "native XGrammar compiler handle is null");
        Ok(CuteafdXGrammarCompiler {
            handle,
            library: self,
        })
    }
}

impl<'a> CuteafdXGrammarCompiler<'a> {
    pub fn compile(
        &self,
        kind: c_int,
        grammar_json: Option<&str>,
        strict: bool,
    ) -> Result<CuteafdXGrammarGrammar<'a>> {
        let grammar_json = grammar_json
            .map(|value| {
                CString::new(value).context("XGrammar source contains an embedded NUL byte")
            })
            .transpose()?;
        let compile: Symbol<XGrammarCompileFn> =
            unsafe { self.library.lib.get(b"cuteafd_xgrammar_compile")? };
        let mut handle = std::ptr::null_mut();
        let mut error = [0 as c_char; XGRAMMAR_ERROR_BYTES];
        let status = unsafe {
            compile(
                self.handle,
                kind,
                grammar_json
                    .as_ref()
                    .map_or(std::ptr::null(), |value| value.as_ptr()),
                c_int::from(strict),
                &mut handle,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        xgrammar_status("cuteafd_xgrammar_compile", status, &error)?;
        anyhow::ensure!(!handle.is_null(), "native XGrammar grammar handle is null");
        Ok(CuteafdXGrammarGrammar {
            handle,
            library: self.library,
        })
    }
}

impl<'a> CuteafdXGrammarGrammar<'a> {
    pub fn matcher(&self) -> Result<CuteafdXGrammarMatcher<'a>> {
        let create: Symbol<XGrammarMatcherCreateFn> =
            unsafe { self.library.lib.get(b"cuteafd_xgrammar_matcher_create")? };
        let mut handle = std::ptr::null_mut();
        let mut error = [0 as c_char; XGRAMMAR_ERROR_BYTES];
        let status = unsafe { create(self.handle, &mut handle, error.as_mut_ptr(), error.len()) };
        xgrammar_status("cuteafd_xgrammar_matcher_create", status, &error)?;
        anyhow::ensure!(!handle.is_null(), "native XGrammar matcher handle is null");
        Ok(CuteafdXGrammarMatcher {
            handle,
            library: self.library,
        })
    }
}

impl<'a> CuteafdXGrammarMatcher<'a> {
    pub fn fork(&self) -> Result<Self> {
        let fork: Symbol<XGrammarMatcherForkFn> =
            unsafe { self.library.lib.get(b"cuteafd_xgrammar_matcher_fork")? };
        let mut handle = std::ptr::null_mut();
        let mut error = [0 as c_char; XGRAMMAR_ERROR_BYTES];
        let status = unsafe { fork(self.handle, &mut handle, error.as_mut_ptr(), error.len()) };
        xgrammar_status("cuteafd_xgrammar_matcher_fork", status, &error)?;
        anyhow::ensure!(!handle.is_null(), "forked native XGrammar matcher is null");
        Ok(Self {
            handle,
            library: self.library,
        })
    }

    pub fn fill_bitmask(&mut self, bitmask: &mut [u32]) -> Result<bool> {
        anyhow::ensure!(!bitmask.is_empty(), "XGrammar bitmask is empty");
        let fill: Symbol<XGrammarMatcherFillBitmaskFn> = unsafe {
            self.library
                .lib
                .get(b"cuteafd_xgrammar_matcher_fill_bitmask")?
        };
        let mut needs_mask = 0;
        let mut error = [0 as c_char; XGRAMMAR_ERROR_BYTES];
        let status = unsafe {
            fill(
                self.handle,
                bitmask.as_mut_ptr(),
                bitmask.len(),
                &mut needs_mask,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        xgrammar_status("cuteafd_xgrammar_matcher_fill_bitmask", status, &error)?;
        Ok(needs_mask != 0)
    }

    pub fn accept_token(&mut self, token_id: u32) -> Result<bool> {
        let accept: Symbol<XGrammarMatcherAcceptTokenFn> = unsafe {
            self.library
                .lib
                .get(b"cuteafd_xgrammar_matcher_accept_token")?
        };
        let mut accepted = 0;
        let mut error = [0 as c_char; XGRAMMAR_ERROR_BYTES];
        let status = unsafe {
            accept(
                self.handle,
                token_id,
                &mut accepted,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        xgrammar_status("cuteafd_xgrammar_matcher_accept_token", status, &error)?;
        Ok(accepted != 0)
    }

    pub fn is_completed(&self) -> Result<bool> {
        let completed: Symbol<XGrammarMatcherIsCompletedFn> = unsafe {
            self.library
                .lib
                .get(b"cuteafd_xgrammar_matcher_is_completed")?
        };
        let mut is_completed = 0;
        let mut error = [0 as c_char; XGRAMMAR_ERROR_BYTES];
        let status = unsafe {
            completed(
                self.handle,
                &mut is_completed,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        xgrammar_status("cuteafd_xgrammar_matcher_is_completed", status, &error)?;
        Ok(is_completed != 0)
    }
}

impl Drop for CuteafdXGrammarCompiler<'_> {
    fn drop(&mut self) {
        if let Ok(destroy) = unsafe {
            self.library
                .lib
                .get::<XGrammarCompilerDestroyFn>(b"cuteafd_xgrammar_compiler_destroy")
        } {
            let _ = unsafe { destroy(self.handle) };
        }
        self.handle = std::ptr::null_mut();
    }
}

impl Drop for CuteafdXGrammarGrammar<'_> {
    fn drop(&mut self) {
        if let Ok(destroy) = unsafe {
            self.library
                .lib
                .get::<XGrammarGrammarDestroyFn>(b"cuteafd_xgrammar_grammar_destroy")
        } {
            let _ = unsafe { destroy(self.handle) };
        }
        self.handle = std::ptr::null_mut();
    }
}

impl Drop for CuteafdXGrammarMatcher<'_> {
    fn drop(&mut self) {
        if let Ok(destroy) = unsafe {
            self.library
                .lib
                .get::<XGrammarMatcherDestroyFn>(b"cuteafd_xgrammar_matcher_destroy")
        } {
            let _ = unsafe { destroy(self.handle) };
        }
        self.handle = std::ptr::null_mut();
    }
}

pub struct CuteafdNcclComm {
    handle: *mut c_void,
    library: Arc<NativeLibrary>,
    world_size: usize,
    rank: usize,
}

unsafe impl Send for CuteafdNcclComm {}


impl CuteafdNcclComm {
    pub fn world_size(&self) -> usize {
        self.world_size
    }

    pub fn rank(&self) -> usize {
        self.rank
    }




}

impl Drop for CuteafdNcclComm {
    fn drop(&mut self) {
        if self.handle.is_null() {
            return;
        }
        if let Ok(destroy_fn) = unsafe {
            self.library
                .lib
                .get::<NcclCommDestroyFn>(b"cuteafd_nccl_comm_destroy")
        } {
            let _ = unsafe { destroy_fn(self.handle) };
        }
        self.handle = std::ptr::null_mut();
    }
}

/// RC endpoint send/recv spans are allocated natively (pinned or malloc'd and
/// registered); the ledger keys them by endpoint handle.
fn record_rdma_rings(handle: *mut c_void, bytes: usize) {
    let _scope = memory_ledger::scope("transport/rdma-rings");
    memory_ledger::record_alloc(memory_ledger::Space::Pinned, -1, handle as usize, bytes);
}

struct SyncH2DStagingBuffer {
    // Released explicitly by NativeLibrary while its module is loaded. Do not
    // add an automatic release here: failed-drain quarantine retains this raw
    // pinned allocation even after the Rust wrapper is dropped.
    buffer: CuteafdHostBuffer,
}

impl Default for SyncH2DStagingBuffer {
    fn default() -> Self {
        Self {
            buffer: CuteafdHostBuffer::default(),
        }
    }
}

unsafe impl Send for SyncH2DStagingBuffer {}

impl SyncH2DStagingBuffer {
    fn ensure(&mut self, library: &NativeLibrary, bytes: usize) -> Result<CuteafdHostBuffer> {
        if self.buffer.ptr.is_null() || self.buffer.bytes < bytes {
            if !self.buffer.ptr.is_null() {
                library
                    .free_host_buffer(&mut self.buffer)
                    .context("freeing undersized synchronous H2D pinned staging buffer")?;
                self.buffer = CuteafdHostBuffer::default();
            }
            let _scope = memory_ledger::scope("staging/sync-h2d");
            self.buffer = library
                .alloc_host_buffer(bytes)
                .context("allocating reusable synchronous H2D pinned staging buffer")?;
            if self.buffer.ptr.is_null() {
                anyhow::bail!("reusable synchronous H2D pinned staging buffer is null");
            }
            if self.buffer.bytes < bytes {
                let allocated_bytes = self.buffer.bytes;
                library
                    .free_host_buffer(&mut self.buffer)
                    .context("freeing undersized reusable synchronous H2D pinned staging buffer")?;
                self.buffer = CuteafdHostBuffer::default();
                anyhow::bail!(
                    "reusable synchronous H2D pinned staging buffer bytes {} is smaller than source bytes {bytes}",
                    allocated_bytes
                );
            }
        }
        Ok(self.buffer)
    }

    fn release_with_library(&mut self, lib: &Library) {
        if self.buffer.ptr.is_null() {
            return;
        }
        if let Ok(free_fn) = unsafe { lib.get::<FreeHostBufferFn>(b"cuteafd_free_host_buffer") } {
            memory_ledger::record_free(self.buffer.ptr as usize);
            let _ = unsafe { free_fn(&mut self.buffer) };
        }
        self.buffer = CuteafdHostBuffer::default();
    }
}

impl Drop for NativeLibrary {
    fn drop(&mut self) {
        if self.quarantine_after_failed_drain.load(Ordering::Relaxed) {
            // A queued kernel may still execute code from this module. The
            // staging wrapper has no Drop: retaining its raw allocation also
            // avoids releasing pinned memory while completion is unknown.
            return;
        }
        if let Ok(staging) = self.sync_h2d_staging.get_mut() {
            staging.release_with_library(&self.lib);
        }
        // SAFETY: this is the sole normal-path owner of the module. Staging
        // was released while its free function was still loaded; quarantine
        // deliberately bypasses both releases.
        unsafe { ManuallyDrop::drop(&mut self.lib) };
    }
}

/// Hands the process expert hidden size to the native pack and route-reduce
/// helpers, which instantiate their indexing per supported size.
fn sync_expert_hidden(lib: &Library) -> Result<()> {
    type SetExpertHiddenFn = unsafe extern "C" fn(u32) -> i32;
    let hidden = cuteafd_core::expert_geometry().hidden;
    match unsafe { lib.get::<SetExpertHiddenFn>(b"cuteafd_set_expert_hidden") } {
        Ok(set) => {
            let status = unsafe { set(hidden) };
            anyhow::ensure!(
                status == 0,
                "native expert helpers have no instantiation for hidden size {hidden}; add it to native/shared/cuda/expert_hidden.cuh"
            );
        }
        Err(_) => anyhow::ensure!(
            hidden == 5120,
            "native library predates geometry-aware expert helpers; rebuild it to serve hidden size {hidden}"
        ),
    }
    Ok(())
}

impl NativeLibrary {
    pub unsafe fn load(path: impl AsRef<Path>) -> Result<Self> {
        let lib = unsafe { Library::new(path.as_ref()) }
            .with_context(|| format!("loading native library {}", path.as_ref().display()))?;
        let rdma_rc_endpoint_try_poll_fn =
            unsafe { *lib.get::<RdmaRcEndpointTryPollFn>(b"cuteafd_rdma_rc_endpoint_try_poll")? };
        sync_expert_hidden(&lib)?;
        Ok(Self {
            lib: ManuallyDrop::new(lib),
            quarantine_after_failed_drain: AtomicBool::new(false),
            sync_h2d_staging: Mutex::new(SyncH2DStagingBuffer::default()),
            rdma_rc_endpoint_try_poll_fn,
        })
    }

    /// Irreversibly retain this module and its reusable pinned H2D staging
    /// until process teardown after a stream drain failed to prove completion.
    /// Idempotent; normal loads and launches do not take an extra owner.
    ///
    /// This protects module code and library-owned staging. The daemon's shared
    /// device/pinned allocation owners also honor the irreversible marker. The
    /// caller must separately retain other source, destination, scratch owners,
    /// and other native owner that queued work may still use, and abandon the
    /// failed operation. It does not repair arbitrary pre-engine load errors
    /// or make subsequent work safe. Other NativeLibrary instances are not
    /// quarantined.
    pub fn quarantine_module_after_failed_drain(&self) {
        self.quarantine_after_failed_drain.store(true, Ordering::Relaxed);
    }

    /// Whether native completion was irreversibly left unproved. Allocation
    /// owners must retain their storage on this path: even an unrelated
    /// cudaFree can synchronize with another stream's pending peer work.
    pub fn is_quarantined_after_failed_drain(&self) -> bool {
        self.quarantine_after_failed_drain.load(Ordering::Relaxed)
    }

    pub fn version(&self) -> Result<String> {
        let version_fn: Symbol<VersionFn> = unsafe { self.lib.get(b"cuteafd_native_version")? };
        let mut buf = vec![0 as c_char; 128];
        let status = unsafe { version_fn(buf.as_mut_ptr(), buf.len()) };
        self.status_to_result("cuteafd_native_version", status)?;
        let cstr = unsafe { CStr::from_ptr(buf.as_ptr()) };
        Ok(cstr.to_string_lossy().into_owned())
    }





    pub fn alloc_host_buffer(&self, bytes: usize) -> Result<CuteafdHostBuffer> {
        let alloc_fn: Symbol<AllocHostBufferFn> =
            unsafe { self.lib.get(b"cuteafd_alloc_host_buffer")? };
        let mut buffer = CuteafdHostBuffer::default();
        let status = unsafe { alloc_fn(bytes, &mut buffer) };
        self.status_to_result("cuteafd_alloc_host_buffer", status)?;
        memory_ledger::record_alloc(memory_ledger::Space::Pinned, -1, buffer.ptr as usize, buffer.bytes);
        Ok(buffer)
    }

    pub fn cuda_host_buffer_device_alias(
        &self,
        host: CuteafdHostBuffer,
    ) -> Result<CuteafdDeviceBuffer> {
        if host.ptr.is_null() || host.bytes == 0 {
            anyhow::bail!("mapped host buffer is empty");
        }
        let alias_fn: Symbol<CudaHostBufferDeviceAliasFn> =
            unsafe { self.lib.get(b"cuteafd_cuda_host_buffer_device_alias")? };
        let mut alias = CuteafdDeviceBuffer::default();
        let status = unsafe { alias_fn(host, &mut alias) };
        self.status_to_result("cuteafd_cuda_host_buffer_device_alias", status)?;
        Ok(alias)
    }

    pub fn free_host_buffer(&self, buffer: &mut CuteafdHostBuffer) -> Result<()> {
        let free_fn: Symbol<FreeHostBufferFn> = unsafe { self.lib.get(b"cuteafd_free_host_buffer")? };
        memory_ledger::record_free(buffer.ptr as usize);
        let status = unsafe { free_fn(buffer) };
        self.status_to_result("cuteafd_free_host_buffer", status)
    }

    /// Physical free and total bytes, including other processes and CUDA's
    /// untracked graph/module allocations. Audits can distinguish this from
    /// the logical admission sample returned by `cuda_memory_info`.
    pub fn cuda_physical_memory_info(&self) -> Result<(usize, usize)> {
        let info: Symbol<unsafe extern "C" fn(*mut usize, *mut usize) -> CuteafdStatus> =
            unsafe { self.lib.get(b"cuteafd_cuda_memory_info")? };
        let (mut free, mut total) = (0, 0);
        self.status_to_result("cuteafd_cuda_memory_info", unsafe { info(&mut free, &mut total) })?;
        anyhow::ensure!(free <= total && total > 0, "invalid CUDA memory information");
        Ok((free, total))
    }

    /// Effective free/total bytes under the coordinator's logical ceiling.
    /// Subtract physical usage, not the min of physical free and the budget:
    /// otherwise every successive owner would spend the same budget again.
    pub fn cuda_memory_info(&self) -> Result<(usize, usize)> {
        let (free, total) = self.cuda_physical_memory_info()?;
        let Some(budget) = coordinator_gpu_budget() else { return Ok((free, total)) };
        let device = self.cuda_get_device()?;
        let sample = cuteafd_core::serving_capacity::DeviceMemory {
            device: u32::try_from(device)?, total_bytes: total as u64, baseline_free_bytes: free as u64,
        };
        // Managed pages may remain on the host and escape cudaMemGetInfo.
        // Reserve their full ownership conservatively, even when CUDA already
        // charges resident pages; current families use device allocations.
        let managed = memory_ledger::current_bytes(memory_ledger::Space::Managed, device) as u64;
        budget.admit(sample, managed)?;
        let mut effective = budget.apply(sample)?;
        effective.baseline_free_bytes -= managed;
        static LOGGED: std::sync::OnceLock<Mutex<std::collections::HashSet<i32>>> = std::sync::OnceLock::new();
        if LOGGED.get_or_init(Default::default).lock().unwrap_or_else(|e| e.into_inner()).insert(device) {
            tracing::info!(device, physical_total_bytes = total, physical_free_bytes = free,
                requested_budget_bytes = budget.0, effective_total_bytes = effective.total_bytes,
                effective_free_bytes = effective.baseline_free_bytes, managed_reserve_bytes = managed,
                "coordinator GPU memory budget (logical ceiling; no guard allocation)");
        }
        Ok((usize::try_from(effective.baseline_free_bytes)?, usize::try_from(effective.total_bytes)?))
    }

    fn admit_gpu_allocation(&self, bytes: usize) -> Result<Option<std::sync::MutexGuard<'static, ()>>> {
        let Some(budget) = coordinator_gpu_budget() else { return Ok(None) };
        // Serialize sample + allocation across loading threads so concurrent
        // owners cannot each admit against the same remaining logical bytes.
        let guard = GPU_BUDGET_ALLOCATION.lock().unwrap_or_else(|e| e.into_inner());
        let (free, total) = self.cuda_memory_info()?;
        budget.admit(cuteafd_core::serving_capacity::DeviceMemory {
            device: u32::try_from(self.cuda_get_device()?)?, total_bytes: total as u64,
            baseline_free_bytes: free as u64,
        }, bytes as u64)?;
        Ok(Some(guard))
    }

    pub fn cuda_device_info(&self, device: i32) -> Result<CuteafdCudaDeviceInfo> {
        // SAFETY: the symbol follows the bundled C ABI; initialized info is
        // exclusively writable and the library remains live across the call.
        unsafe {
            let call: Symbol<unsafe extern "C" fn(i32, *mut CuteafdCudaDeviceInfo) -> CuteafdStatus> =
                self.lib.get(b"cuteafd_cuda_device_info")?;
            let mut info = CuteafdCudaDeviceInfo::default();
            self.status_to_result("cuteafd_cuda_device_info", call(device, &mut info))?;
            Ok(info)
        }
    }

    pub fn cuda_get_device(&self) -> Result<i32> {
        let call: Symbol<unsafe extern "C" fn(*mut i32) -> CuteafdStatus> =
            unsafe { self.lib.get(b"cuteafd_cuda_get_device")? };
        let mut device = -1;
        self.status_to_result("cuteafd_cuda_get_device", unsafe { call(&mut device) })?;
        Ok(device)
    }

    /// Host-thread-local selection. Restore the previous device before yielding.
    pub fn cuda_set_device(&self, device: i32) -> Result<()> {
        let call: Symbol<unsafe extern "C" fn(i32) -> CuteafdStatus> =
            unsafe { self.lib.get(b"cuteafd_cuda_set_device")? };
        self.status_to_result("cuteafd_cuda_set_device", unsafe { call(device) })
    }

    pub fn cuda_enable_peer(&self, peer: i32) -> Result<()> {
        let call: Symbol<unsafe extern "C" fn(i32) -> CuteafdStatus> =
            unsafe { self.lib.get(b"cuteafd_cuda_enable_peer")? };
        self.status_to_result("cuteafd_cuda_enable_peer", unsafe { call(peer) })
    }

    /// Enqueue a cross-device transfer without joining other streams.
    ///
    /// # Safety
    /// `stream` must be live on the current, destination device. Source writes
    /// must precede the copy (use a stream event dependency). Both allocations
    /// must remain live, with no conflicting access, until the copy completes.
    pub unsafe fn copy_peer_async(
        &self,
        dst: CuteafdDeviceBuffer,
        src: CuteafdDeviceBuffer,
        bytes: usize,
        stream: *mut c_void,
    ) -> Result<()> {
        let call: Symbol<unsafe extern "C" fn(
            CuteafdDeviceBuffer, CuteafdDeviceBuffer, usize, *mut c_void,
        ) -> CuteafdStatus> = unsafe { self.lib.get(b"cuteafd_copy_peer_async")? };
        self.status_to_result("cuteafd_copy_peer_async", unsafe { call(dst, src, bytes, stream) })
    }

    /// Copy pitched byte rows locally or between peer GPUs without synchronization.
    ///
    /// # Safety
    /// The stream belongs to the current destination device. Source writes are
    /// complete or ordered before this copy. Buffers remain live and disjoint,
    /// without conflicting access, until stream completion.
    pub unsafe fn copy_device_rows_async(
        &self, dst: CuteafdDeviceBuffer, src: CuteafdDeviceBuffer,
        width: usize, rows: usize, dst_pitch: usize, src_pitch: usize,
        stream: *mut c_void,
    ) -> Result<()> {
        let call: Symbol<unsafe extern "C" fn(
            CuteafdDeviceBuffer, CuteafdDeviceBuffer, usize, usize, usize, usize, *mut c_void,
        ) -> CuteafdStatus> = unsafe { self.lib.get(b"cuteafd_copy_device_rows_async")? };
        self.status_to_result("cuteafd_copy_device_rows_async", unsafe {
            call(dst, src, width, rows, dst_pitch, src_pitch, stream)
        })
    }

    pub fn alloc_device_buffer(&self, bytes: usize) -> Result<CuteafdDeviceBuffer> {
        let _budget_guard = self.admit_gpu_allocation(bytes)?;
        let alloc_fn: Symbol<AllocDeviceBufferFn> =
            unsafe { self.lib.get(b"cuteafd_alloc_device_buffer")? };
        let mut buffer = CuteafdDeviceBuffer::default();
        let status = unsafe { alloc_fn(bytes, &mut buffer) };
        self.status_to_result("cuteafd_alloc_device_buffer", status)?;
        memory_ledger::record_alloc(memory_ledger::Space::Device, buffer.device_id, buffer.ptr as usize, buffer.bytes);
        Ok(buffer)
    }

    pub fn alloc_managed_device_buffer(&self, bytes: usize) -> Result<CuteafdDeviceBuffer> {
        let _budget_guard = self.admit_gpu_allocation(bytes)?;
        let alloc_fn: Symbol<AllocManagedDeviceBufferFn> =
            unsafe { self.lib.get(b"cuteafd_alloc_managed_device_buffer")? };
        let mut buffer = CuteafdDeviceBuffer::default();
        let status = unsafe { alloc_fn(bytes, &mut buffer) };
        self.status_to_result("cuteafd_alloc_managed_device_buffer", status)?;
        memory_ledger::record_alloc(memory_ledger::Space::Managed, buffer.device_id, buffer.ptr as usize, buffer.bytes);
        Ok(buffer)
    }

    pub fn free_device_buffer(&self, buffer: &mut CuteafdDeviceBuffer) -> Result<()> {
        let free_fn: Symbol<FreeDeviceBufferFn> =
            unsafe { self.lib.get(b"cuteafd_free_device_buffer")? };
        memory_ledger::record_free(buffer.ptr as usize);
        let status = unsafe { free_fn(buffer) };
        self.status_to_result("cuteafd_free_device_buffer", status)
    }

    pub fn cuda_stream_create(&self) -> Result<*mut c_void> {
        let create_fn: Symbol<CudaStreamCreateFn> =
            unsafe { self.lib.get(b"cuteafd_cuda_stream_create")? };
        let mut cuda_stream = std::ptr::null_mut();
        let status = unsafe { create_fn(&mut cuda_stream) };
        self.status_to_result("cuteafd_cuda_stream_create", status)?;
        Ok(cuda_stream)
    }

    pub unsafe fn cuda_stream_destroy(&self, cuda_stream: *mut c_void) -> Result<()> {
        let destroy_fn: Symbol<CudaStreamDestroyFn> =
            unsafe { self.lib.get(b"cuteafd_cuda_stream_destroy")? };
        let status = unsafe { destroy_fn(cuda_stream) };
        self.status_to_result("cuteafd_cuda_stream_destroy", status)
    }

    pub unsafe fn cuda_stream_synchronize(&self, cuda_stream: *mut c_void) -> Result<()> {
        let synchronize_fn: Symbol<CudaStreamSynchronizeFn> =
            unsafe { self.lib.get(b"cuteafd_cuda_stream_synchronize")? };
        let status = unsafe { synchronize_fn(cuda_stream) };
        self.status_to_result("cuteafd_cuda_stream_synchronize", status)
    }

    /// Poll completion without blocking the CUDA owner thread.
    /// The stream must remain live on the current CUDA device.
    pub unsafe fn cuda_stream_query(&self, cuda_stream: *mut c_void) -> Result<bool> {
        let query_fn: Symbol<CudaStreamQueryFn> =
            unsafe { self.lib.get(b"cuteafd_cuda_stream_query")? };
        let mut ready = 0;
        let status = unsafe { query_fn(cuda_stream, &mut ready) };
        self.status_to_result("cuteafd_cuda_stream_query", status)?;
        anyhow::ensure!(ready == 0 || ready == 1, "invalid CUDA stream query result");
        Ok(ready == 1)
    }

    pub unsafe fn cuda_stream_wait_event(
        &self,
        cuda_stream: *mut c_void,
        cuda_event: *mut c_void,
    ) -> Result<()> {
        let wait_fn: Symbol<CudaStreamWaitEventFn> =
            unsafe { self.lib.get(b"cuteafd_cuda_stream_wait_event")? };
        let status = unsafe { wait_fn(cuda_stream, cuda_event) };
        self.status_to_result("cuteafd_cuda_stream_wait_event", status)
    }

    pub fn cuda_event_create(&self) -> Result<*mut c_void> {
        let create_fn: Symbol<CudaEventCreateFn> =
            unsafe { self.lib.get(b"cuteafd_cuda_event_create")? };
        let mut cuda_event = std::ptr::null_mut();
        let status = unsafe { create_fn(&mut cuda_event) };
        self.status_to_result("cuteafd_cuda_event_create", status)?;
        Ok(cuda_event)
    }

    /// Timing-disabled event for stream ordering only.
    pub fn cuda_event_create_ordering(&self) -> Result<*mut c_void> {
        let create_fn: Symbol<CudaEventCreateFn> =
            unsafe { self.lib.get(b"cuteafd_cuda_event_create_ordering")? };
        let mut cuda_event = std::ptr::null_mut();
        let status = unsafe { create_fn(&mut cuda_event) };
        self.status_to_result("cuteafd_cuda_event_create_ordering", status)?;
        Ok(cuda_event)
    }

    pub unsafe fn cuda_event_destroy(&self, cuda_event: *mut c_void) -> Result<()> {
        let destroy_fn: Symbol<CudaEventDestroyFn> =
            unsafe { self.lib.get(b"cuteafd_cuda_event_destroy")? };
        let status = unsafe { destroy_fn(cuda_event) };
        self.status_to_result("cuteafd_cuda_event_destroy", status)
    }

    pub unsafe fn cuda_event_record(
        &self,
        cuda_event: *mut c_void,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        let record_fn: Symbol<CudaEventRecordFn> =
            unsafe { self.lib.get(b"cuteafd_cuda_event_record")? };
        let status = unsafe { record_fn(cuda_event, cuda_stream) };
        self.status_to_result("cuteafd_cuda_event_record", status)
    }

    pub unsafe fn cuda_event_synchronize(&self, cuda_event: *mut c_void) -> Result<()> {
        let synchronize_fn: Symbol<CudaEventSynchronizeFn> =
            unsafe { self.lib.get(b"cuteafd_cuda_event_synchronize")? };
        let status = unsafe { synchronize_fn(cuda_event) };
        self.status_to_result("cuteafd_cuda_event_synchronize", status)
    }

    pub unsafe fn cuda_event_elapsed_ms(
        &self,
        start_event: *mut c_void,
        end_event: *mut c_void,
    ) -> Result<f32> {
        let elapsed_fn: Symbol<CudaEventElapsedMsFn> =
            unsafe { self.lib.get(b"cuteafd_cuda_event_elapsed_ms")? };
        let mut out_ms = 0.0_f32;
        let status = unsafe { elapsed_fn(start_event, end_event, &mut out_ms) };
        self.status_to_result("cuteafd_cuda_event_elapsed_ms", status)?;
        Ok(out_ms)
    }

    #[track_caller]
    pub unsafe fn cuda_graph_begin_capture(&self, cuda_stream: *mut c_void) -> Result<()> {
        let site = std::panic::Location::caller();
        {
            let mut sites = GRAPH_CAPTURE_SITES.lock().unwrap_or_else(|p| p.into_inner());
            match sites.iter_mut().find(|(l, _)| std::ptr::eq(*l, site)) {
                Some((_, n)) => *n += 1,
                None => sites.push((site, 1)),
            }
        }
        let begin_capture_fn: Symbol<CudaGraphBeginCaptureFn> =
            unsafe { self.lib.get(b"cuteafd_cuda_graph_begin_capture")? };
        let status = unsafe { begin_capture_fn(cuda_stream) };
        GRAPH_CAPTURES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.status_to_result("cuteafd_cuda_graph_begin_capture", status)
    }

    pub unsafe fn cuda_graph_end_capture(&self, cuda_stream: *mut c_void) -> Result<*mut c_void> {
        let _budget_guard = coordinator_gpu_budget().map(|_| GPU_BUDGET_ALLOCATION.lock()
            .unwrap_or_else(|e| e.into_inner()));
        let end_capture_fn: Symbol<CudaGraphEndCaptureFn> =
            unsafe { self.lib.get(b"cuteafd_cuda_graph_end_capture")? };
        let mut cuda_graph_exec = std::ptr::null_mut();
        let status = unsafe { end_capture_fn(cuda_stream, &mut cuda_graph_exec) };
        self.status_to_result("cuteafd_cuda_graph_end_capture", status)?;
        if coordinator_gpu_budget().is_some() {
            if let Err(error) = self.cuda_memory_info() {
                // SAFETY: capture ended and this executable has never launched;
                // destroying it cannot invalidate queued graph work.
                unsafe { self.cuda_graph_exec_destroy(cuda_graph_exec)? };
                return Err(error.context("graph executable exceeds coordinator GPU budget"));
            }
        }
        Ok(cuda_graph_exec)
    }


    pub unsafe fn cuda_graph_launch(
        &self,
        cuda_graph_exec: *mut c_void,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        let launch_fn: Symbol<CudaGraphLaunchFn> =
            unsafe { self.lib.get(b"cuteafd_cuda_graph_launch")? };
        let status = unsafe { launch_fn(cuda_graph_exec, cuda_stream) };
        self.status_to_result("cuteafd_cuda_graph_launch", status)
    }



    pub unsafe fn cuda_graph_exec_destroy(&self, cuda_graph_exec: *mut c_void) -> Result<()> {
        let destroy_fn: Symbol<CudaGraphExecDestroyFn> =
            unsafe { self.lib.get(b"cuteafd_cuda_graph_exec_destroy")? };
        let status = unsafe { destroy_fn(cuda_graph_exec) };
        self.status_to_result("cuteafd_cuda_graph_exec_destroy", status)
    }






















    pub fn copy_h2d(&self, dst: CuteafdDeviceBuffer, src: &[u8]) -> Result<()> {
        if src.is_empty() {
            return Ok(());
        }
        if src.len() > dst.bytes {
            anyhow::bail!(
                "cuteafd_copy_h2d staged source byte count {} exceeds destination device buffer bytes {}",
                src.len(),
                dst.bytes
            );
        }
        let mut staging = self
            .sync_h2d_staging
            .lock()
            .map_err(|_| anyhow::anyhow!("synchronous H2D pinned staging lock is poisoned"))?;
        let staging = staging.ensure(self, src.len())?;
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), staging.ptr.cast::<u8>(), src.len());
        }
        unsafe { self.copy_h2d_raw_ptr(dst, staging.ptr as *const c_void, src.len()) }
    }

    pub fn copy_host_buffer_h2d(
        &self,
        dst: CuteafdDeviceBuffer,
        src: CuteafdHostBuffer,
        bytes: usize,
    ) -> Result<()> {
        if bytes == 0 {
            return Ok(());
        }
        if src.ptr.is_null() {
            anyhow::bail!("cuteafd_copy_h2d pinned source host buffer is null");
        }
        if dst.ptr.is_null() {
            anyhow::bail!("cuteafd_copy_h2d destination device buffer is null");
        }
        if bytes > src.bytes {
            anyhow::bail!(
                "cuteafd_copy_h2d pinned source byte count {bytes} exceeds host buffer bytes {}",
                src.bytes
            );
        }
        if bytes > dst.bytes {
            anyhow::bail!(
                "cuteafd_copy_h2d pinned source byte count {bytes} exceeds destination device buffer bytes {}",
                dst.bytes
            );
        }
        unsafe { self.copy_h2d_raw_ptr(dst, src.ptr as *const c_void, bytes) }
    }

    unsafe fn copy_h2d_raw_ptr(
        &self,
        dst: CuteafdDeviceBuffer,
        src: *const c_void,
        bytes: usize,
    ) -> Result<()> {
        let copy_fn: Symbol<CopyH2DFn> = unsafe { self.lib.get(b"cuteafd_copy_h2d")? };
        let status = unsafe { copy_fn(dst, src, bytes) };
        self.status_to_result("cuteafd_copy_h2d", status)
    }

    pub unsafe fn copy_host_buffer_h2d_async(
        &self,
        dst: CuteafdDeviceBuffer,
        src: CuteafdHostBuffer,
        bytes: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        if src.ptr.is_null() {
            anyhow::bail!("cuteafd_copy_h2d_async pinned source host buffer is null");
        }
        if bytes > src.bytes {
            anyhow::bail!(
                "cuteafd_copy_h2d_async pinned source byte count {bytes} exceeds host buffer bytes {}",
                src.bytes
            );
        }
        let copy_fn: Symbol<CopyH2DAsyncFn> = unsafe { self.lib.get(b"cuteafd_copy_h2d_async")? };
        let status = unsafe { copy_fn(dst, src.ptr as *const c_void, bytes, cuda_stream) };
        self.status_to_result("cuteafd_copy_h2d_async", status)
    }

    pub unsafe fn copy_host_buffers_h2d_batch_async(
        &self,
        dsts: &[CuteafdDeviceBuffer],
        srcs: &[CuteafdHostBuffer],
        bytes: &[usize],
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        let context = "cuteafd_copy_h2d_batch_async";
        if dsts.is_empty() {
            return Ok(());
        }
        if dsts.len() != srcs.len() || dsts.len() != bytes.len() {
            anyhow::bail!(
                "{context} array lengths differ: dsts={} srcs={} bytes={}",
                dsts.len(),
                srcs.len(),
                bytes.len()
            );
        }
        if cuda_stream.is_null() {
            anyhow::bail!("{context} requires a non-default CUDA stream");
        }
        for (index, ((dst, src), &entry_bytes)) in dsts.iter().zip(srcs).zip(bytes).enumerate() {
            if entry_bytes == 0 {
                anyhow::bail!("{context} entry {index} is empty");
            }
            if src.ptr.is_null() {
                anyhow::bail!("{context} entry {index} source host buffer is null");
            }
            if entry_bytes > src.bytes {
                anyhow::bail!(
                    "{context} entry {index} byte count {entry_bytes} exceeds host buffer bytes {}",
                    src.bytes
                );
            }
            validate_device_buffer_bytes(
                &format!("{context} entry {index} destination"),
                *dst,
                entry_bytes,
            )?;
        }
        let source_ptrs = srcs
            .iter()
            .map(|source| source.ptr.cast_const())
            .collect::<Vec<_>>();
        let copy_fn: Symbol<CopyH2DBatchAsyncFn> =
            unsafe { self.lib.get(b"cuteafd_copy_h2d_batch_async")? };
        let status = unsafe {
            copy_fn(
                dsts.as_ptr(),
                source_ptrs.as_ptr(),
                bytes.as_ptr(),
                dsts.len(),
                cuda_stream,
            )
        };
        self.status_to_result(context, status)
    }

    pub unsafe fn copy_host_buffer_h2d_2d_async(
        &self,
        dst: CuteafdDeviceBuffer,
        dst_pitch_bytes: usize,
        src: CuteafdHostBuffer,
        src_pitch_bytes: usize,
        width_bytes: usize,
        rows: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        let context = "cuteafd_copy_h2d_2d_async";
        if width_bytes == 0 || rows == 0 {
            return Ok(());
        }
        if src.ptr.is_null() {
            anyhow::bail!("{context} pinned source host buffer is null");
        }
        if dst_pitch_bytes < width_bytes || src_pitch_bytes < width_bytes {
            anyhow::bail!("{context} pitch must be at least the row width");
        }
        let dst_required = (rows - 1)
            .checked_mul(dst_pitch_bytes)
            .and_then(|bytes| bytes.checked_add(width_bytes))
            .context("2D H2D destination byte span overflow")?;
        let src_required = (rows - 1)
            .checked_mul(src_pitch_bytes)
            .and_then(|bytes| bytes.checked_add(width_bytes))
            .context("2D H2D source byte span overflow")?;
        validate_device_buffer_bytes(&format!("{context} destination"), dst, dst_required)?;
        if src_required > src.bytes {
            anyhow::bail!(
                "{context} source byte span {src_required} exceeds host buffer bytes {}",
                src.bytes
            );
        }
        let copy_fn: Symbol<CopyH2D2DAsyncFn> =
            unsafe { self.lib.get(b"cuteafd_copy_h2d_2d_async")? };
        let status = unsafe {
            copy_fn(
                dst,
                dst_pitch_bytes,
                src.ptr as *const c_void,
                src_pitch_bytes,
                width_bytes,
                rows,
                cuda_stream,
            )
        };
        self.status_to_result(context, status)
    }

    pub fn copy_d2h(&self, dst: &mut [u8], src: CuteafdDeviceBuffer) -> Result<()> {
        let copy_fn: Symbol<CopyD2HFn> = unsafe { self.lib.get(b"cuteafd_copy_d2h")? };
        let status = unsafe { copy_fn(dst.as_mut_ptr().cast(), src, dst.len()) };
        self.status_to_result("cuteafd_copy_d2h", status)
    }

    /// Frees the pinned staging buffer [`Self::copy_h2d`] grows to its largest
    /// upload. Loaders call this once their weights are resident: the buffer
    /// otherwise stays pinned for the life of the process (on GB10 it is the
    /// same unified memory the experts live in). Later copies reallocate it.
    pub fn release_sync_h2d_staging(&self) -> Result<usize> {
        let mut staging = self
            .sync_h2d_staging
            .lock()
            .map_err(|_| anyhow::anyhow!("synchronous H2D pinned staging lock is poisoned"))?;
        if staging.buffer.ptr.is_null() {
            return Ok(0);
        }
        let bytes = staging.buffer.bytes;
        self.free_host_buffer(&mut staging.buffer)
            .context("freeing synchronous H2D pinned staging buffer")?;
        staging.buffer = CuteafdHostBuffer::default();
        Ok(bytes)
    }

    #[cfg(test)]
    fn sync_h2d_staging_snapshot(&self) -> Option<(usize, usize)> {
        let staging = self.sync_h2d_staging.lock().ok()?;
        (!staging.buffer.ptr.is_null())
            .then_some((staging.buffer.ptr as usize, staging.buffer.bytes))
    }

    /// Copy initialized device storage and wait for the copy to complete.
    /// Producers must have completed before this call; subsequent consumers
    /// may use any stream. Use `copy_d2d_async` for explicitly ordered work.
    pub fn copy_d2d(
        &self,
        dst: CuteafdDeviceBuffer,
        src: CuteafdDeviceBuffer,
        bytes: usize,
    ) -> Result<()> {
        if bytes == 0 {
            return Ok(());
        }
        if dst.ptr.is_null() {
            anyhow::bail!("cuteafd_copy_d2d destination device buffer is null");
        }
        if src.ptr.is_null() {
            anyhow::bail!("cuteafd_copy_d2d source device buffer is null");
        }
        if bytes > dst.bytes {
            anyhow::bail!(
                "cuteafd_copy_d2d byte count {bytes} exceeds destination device buffer bytes {}",
                dst.bytes
            );
        }
        if bytes > src.bytes {
            anyhow::bail!(
                "cuteafd_copy_d2d byte count {bytes} exceeds source device buffer bytes {}",
                src.bytes
            );
        }
        let copy_fn: Symbol<CopyD2DFn> = unsafe { self.lib.get(b"cuteafd_copy_d2d")? };
        let status = unsafe { copy_fn(dst, src, bytes) };
        self.status_to_result("cuteafd_copy_d2d", status)
    }

    pub unsafe fn copy_h2d_async(
        &self,
        dst: CuteafdDeviceBuffer,
        src: &[u8],
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        let copy_fn: Symbol<CopyH2DAsyncFn> = unsafe { self.lib.get(b"cuteafd_copy_h2d_async")? };
        let status = unsafe { copy_fn(dst, src.as_ptr().cast(), src.len(), cuda_stream) };
        self.status_to_result("cuteafd_copy_h2d_async", status)
    }

    pub unsafe fn copy_d2h_async(
        &self,
        dst: &mut [u8],
        src: CuteafdDeviceBuffer,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        let copy_fn: Symbol<CopyD2HAsyncFn> = unsafe { self.lib.get(b"cuteafd_copy_d2h_async")? };
        let status = unsafe { copy_fn(dst.as_mut_ptr().cast(), src, dst.len(), cuda_stream) };
        self.status_to_result("cuteafd_copy_d2h_async", status)
    }

    pub unsafe fn copy_d2h_host_buffer_async(
        &self,
        dst: CuteafdHostBuffer,
        src: CuteafdDeviceBuffer,
        bytes: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        if dst.ptr.is_null() {
            anyhow::bail!("cuteafd_copy_d2h_async pinned destination host buffer is null");
        }
        if bytes > dst.bytes {
            anyhow::bail!(
                "cuteafd_copy_d2h_async pinned destination byte count {bytes} exceeds host buffer bytes {}",
                dst.bytes
            );
        }
        let copy_fn: Symbol<CopyD2HAsyncFn> = unsafe { self.lib.get(b"cuteafd_copy_d2h_async")? };
        let status = unsafe { copy_fn(dst.ptr, src, bytes, cuda_stream) };
        self.status_to_result("cuteafd_copy_d2h_async", status)
    }

    /// Copies `rows` rows of `width_bytes` between pitched device buffers
    /// (`cudaMemcpy2DAsync`; with peer access the two may live on different GPUs).
    ///
    /// # Safety
    /// Both buffers are live, cover their pitched spans (checked natively),
    /// do not overlap, and stay untouched by other work until `cuda_stream`
    /// reaches the copy; the stream belongs to the current device.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn copy_d2d_2d_async(&self, dst: CuteafdDeviceBuffer, dst_pitch_bytes: usize, src: CuteafdDeviceBuffer,
        src_pitch_bytes: usize, width_bytes: usize, rows: usize, cuda_stream: *mut c_void) -> Result<()> {
        type F = unsafe extern "C" fn(CuteafdDeviceBuffer, usize, CuteafdDeviceBuffer, usize, usize, usize, *mut c_void)
            -> CuteafdStatus;
        let copy_fn: Symbol<F> = unsafe { self.lib.get(b"cuteafd_copy_d2d_2d_async")? };
        let status = unsafe { copy_fn(dst, dst_pitch_bytes, src, src_pitch_bytes, width_bytes, rows, cuda_stream) };
        self.status_to_result("cuteafd_copy_d2d_2d_async", status)
    }

    pub unsafe fn copy_d2d_async(
        &self,
        dst: CuteafdDeviceBuffer,
        src: CuteafdDeviceBuffer,
        bytes: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        if bytes == 0 {
            return Ok(());
        }
        if dst.ptr.is_null() {
            anyhow::bail!("cuteafd_copy_d2d_async destination device buffer is null");
        }
        if src.ptr.is_null() {
            anyhow::bail!("cuteafd_copy_d2d_async source device buffer is null");
        }
        if bytes > dst.bytes {
            anyhow::bail!(
                "cuteafd_copy_d2d_async byte count {bytes} exceeds destination device buffer bytes {}",
                dst.bytes
            );
        }
        if bytes > src.bytes {
            anyhow::bail!(
                "cuteafd_copy_d2d_async byte count {bytes} exceeds source device buffer bytes {}",
                src.bytes
            );
        }
        let copy_fn: Symbol<CopyD2DAsyncFn> = unsafe { self.lib.get(b"cuteafd_copy_d2d_async")? };
        let status = unsafe { copy_fn(dst, src, bytes, cuda_stream) };
        self.status_to_result("cuteafd_copy_d2d_async", status)
    }





    /// Dequantize gathered engram FP8 rows and UE8M0 scales into a preallocated BF16 buffer.
    ///
    /// # Safety
    /// Buffers must be nonoverlapping, device-resident on the stream's device,
    /// and remain valid until the stream completes.
    pub unsafe fn cuda_engram_dequant_bf16_async(
        &self, weights: CuteafdDeviceBuffer, scales: CuteafdDeviceBuffer,
        out: CuteafdDeviceBuffer, hash_rows: i32, cuda_stream: *mut c_void,
    ) -> Result<()> {
        const NAME: &str = "cuteafd_cuda_engram_dequant_bf16_async";
        validate_f32_rows(NAME, hash_rows, 256)?;
        let values = checked_row_values(NAME, hash_rows as usize, 256)?;
        validate_device_buffer_bytes(NAME, weights, values)?;
        validate_device_buffer_bytes(NAME, scales, values / 32)?;
        validate_u16_buffer_values(NAME, out, values)?;
        type Kernel = unsafe extern "C" fn(*const u8, *const u8, *mut u16, i32, *mut c_void) -> CuteafdStatus;
        let kernel: Symbol<Kernel> = unsafe { self.lib.get(b"cuteafd_cuda_engram_dequant_bf16_async")? };
        let status = unsafe { kernel(weights.ptr.cast(), scales.ptr.cast(), out.ptr.cast(), hash_rows, cuda_stream) };
        self.status_to_result(NAME, status)
    }

    /// Decode packed NVFP4 PLE rows directly to BF16 without another quantization.
    /// # Safety
    /// Input/output regions must not overlap and must remain live on the stream's device.
    pub unsafe fn cuda_engram_nvfp4_dequant_bf16_async(
        &self, weights: CuteafdDeviceBuffer, scales: CuteafdDeviceBuffer, global_scale: f32,
        out: CuteafdDeviceBuffer, hash_rows: i32, cuda_stream: *mut c_void,
    ) -> Result<()> {
        const NAME: &str = "cuteafd_cuda_engram_nvfp4_dequant_bf16_async";
        anyhow::ensure!(global_scale.is_finite() && global_scale > 0.0, "invalid NVFP4 PLE global scale");
        validate_f32_rows(NAME, hash_rows, 256)?;
        let values = checked_row_values(NAME, hash_rows as usize, 256)?;
        validate_device_buffer_bytes(NAME, weights, values / 2)?;
        validate_device_buffer_bytes(NAME, scales, values / 16)?;
        validate_u16_buffer_values(NAME, out, values)?;
        type Kernel = unsafe extern "C" fn(*const u8, *const u8, f32, *mut u16, i32, *mut c_void) -> CuteafdStatus;
        let kernel: Symbol<Kernel> = unsafe { self.lib.get(b"cuteafd_cuda_engram_nvfp4_dequant_bf16_async")? };
        let status = unsafe { kernel(weights.ptr.cast(), scales.ptr.cast(), global_scale,
            out.ptr.cast(), hash_rows, cuda_stream) };
        self.status_to_result(NAME, status)
    }

    /// Fused official V4.1 engram gate; buffers must remain live on the stream.
    ///
    /// # Safety
    /// All buffers must be device-resident on the stream's device; output may
    /// equal x but must not overlap projected_kv, weights, or the text mask.
    pub unsafe fn cuda_engram_gate_bf16_async(
        &self,
        x: CuteafdDeviceBuffer,
        projected_kv: CuteafdDeviceBuffer,
        q_weight: CuteafdDeviceBuffer,
        k_weight: CuteafdDeviceBuffer,
        text_mask: Option<CuteafdDeviceBuffer>,
        out: CuteafdDeviceBuffer,
        rows: i32,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        const NAME: &str = "cuteafd_cuda_engram_gate_bf16_async";
        validate_f32_rows(NAME, rows, 5120)?;
        let residual_values = checked_row_values(NAME, rows as usize, 4 * 5120)?;
        let kv_values = checked_row_values(NAME, rows as usize, 5 * 5120)?;
        validate_u16_buffer_values(NAME, x, residual_values)?;
        validate_u16_buffer_values(NAME, out, residual_values)?;
        validate_u16_buffer_values(NAME, projected_kv, kv_values)?;
        validate_u16_buffer_values(NAME, q_weight, 4 * 5120)?;
        validate_u16_buffer_values(NAME, k_weight, 4 * 5120)?;
        if let Some(mask) = text_mask { validate_device_buffer_bytes(NAME, mask, rows as usize)?; }
        type Kernel = unsafe extern "C" fn(*const u16, *const u16, *const u16, *const u16,
            *const u8, *mut u16, i32, *mut c_void) -> CuteafdStatus;
        let kernel: Symbol<Kernel> = unsafe { self.lib.get(b"cuteafd_cuda_engram_gate_bf16_async")? };
        let status = unsafe { kernel(x.ptr.cast(), projected_kv.ptr.cast(), q_weight.ptr.cast(),
            k_weight.ptr.cast(), text_mask.map_or(std::ptr::null(), |mask| mask.ptr.cast()),
            out.ptr.cast(), rows, cuda_stream) };
        self.status_to_result(NAME, status)
    }



    pub unsafe fn cuda_ds4_rmsnorm_bf16_rne_async(
        &self,
        x: CuteafdDeviceBuffer,
        weight: CuteafdDeviceBuffer,
        out: CuteafdDeviceBuffer,
        rows: i32,
        hidden: i32,
        eps: f32,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        validate_f32_rows("cuteafd_cuda_ds4_rmsnorm_bf16_rne_async", rows, hidden)?;
        let row_values = checked_row_values(
            "cuteafd_cuda_ds4_rmsnorm_bf16_rne_async x",
            rows as usize,
            hidden as usize,
        )?;
        validate_u16_buffer_values("cuteafd_cuda_ds4_rmsnorm_bf16_rne_async x", x, row_values)?;
        validate_u16_buffer_values(
            "cuteafd_cuda_ds4_rmsnorm_bf16_rne_async weight",
            weight,
            hidden as usize,
        )?;
        validate_u16_buffer_values("cuteafd_cuda_ds4_rmsnorm_bf16_rne_async out", out, row_values)?;

        let kernel_fn: Symbol<CudaDs4RmsNormBf16RneAsyncFn> =
            unsafe { self.lib.get(b"cuteafd_cuda_ds4_rmsnorm_bf16_rne_async")? };
        let status = unsafe {
            kernel_fn(
                x.ptr.cast::<u16>() as *const u16,
                weight.ptr.cast::<u16>() as *const u16,
                out.ptr.cast::<u16>(),
                rows,
                hidden,
                eps,
                cuda_stream,
            )
        };
        self.status_to_result("cuteafd_cuda_ds4_rmsnorm_bf16_rne_async", status)
    }








































    /// Zero-pad one expert's four planes and swizzle scales at load time.
    pub unsafe fn cuda_nvfp4_pad_expert_async(
        &self,
        sources: [CuteafdDeviceBuffer; 4],
        destinations: [CuteafdDeviceBuffer; 4],
        source_intermediate: usize,
        kernel_intermediate: usize,
        stream: *mut c_void,
    ) -> Result<()> {
        let kernel_fn: Symbol<CudaNvfp4PadExpertAsyncFn> =
            unsafe { self.lib.get(b"cuteafd_cuda_nvfp4_pad_expert_async")? };
        let status = unsafe {
            kernel_fn(sources.as_ptr(), destinations.as_ptr(), source_intermediate,
                      kernel_intermediate, stream)
        };
        self.status_to_result("cuteafd_cuda_nvfp4_pad_expert_async", status)
    }

    /// Re-swizzle a plain E4M3 plane into 128x4 scale-factor atoms.
    pub unsafe fn cuda_nvfp4_swizzle_scale_async(
        &self,
        source: CuteafdDeviceBuffer,
        destination: CuteafdDeviceBuffer,
        rows: usize,
        cols: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        let kernel_fn: Symbol<CudaNvfp4SwizzleScaleAsyncFn> =
            unsafe { self.lib.get(b"cuteafd_cuda_nvfp4_swizzle_scale_async")? };
        let status = unsafe { kernel_fn(source, destination, rows, cols, cuda_stream) };
        self.status_to_result("cuteafd_cuda_nvfp4_swizzle_scale_async", status)
    }















































    pub fn cuda_zero_bytes(&self, dst: CuteafdDeviceBuffer, bytes: usize) -> Result<()> {
        validate_device_buffer_bytes("cuteafd_cuda_zero_bytes dst", dst, bytes)?;

        let kernel_fn: Symbol<CudaZeroBytesFn> = unsafe { self.lib.get(b"cuteafd_cuda_zero_bytes")? };
        let status = unsafe { kernel_fn(dst.ptr, bytes) };
        self.status_to_result("cuteafd_cuda_zero_bytes", status)
    }

    pub unsafe fn cuda_zero_bytes_async(
        &self,
        dst: CuteafdDeviceBuffer,
        bytes: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        validate_device_buffer_bytes("cuteafd_cuda_zero_bytes_async dst", dst, bytes)?;

        let kernel_fn: Symbol<CudaZeroBytesAsyncFn> =
            unsafe { self.lib.get(b"cuteafd_cuda_zero_bytes_async")? };
        let status = unsafe { kernel_fn(dst.ptr, bytes, cuda_stream) };
        self.status_to_result("cuteafd_cuda_zero_bytes_async", status)
    }

    pub fn cuda_f32_to_bf16(
        &self,
        src: CuteafdDeviceBuffer,
        dst: CuteafdDeviceBuffer,
        count: usize,
    ) -> Result<()> {
        validate_f32_to_bf16_buffers("cuteafd_cuda_f32_to_bf16", src, dst, count)?;

        let kernel_fn: Symbol<CudaF32ToBf16Fn> =
            unsafe { self.lib.get(b"cuteafd_cuda_f32_to_bf16")? };
        let status = unsafe {
            kernel_fn(
                src.ptr.cast::<f32>() as *const f32,
                dst.ptr.cast::<u16>(),
                count,
            )
        };
        self.status_to_result("cuteafd_cuda_f32_to_bf16", status)
    }

    pub unsafe fn cuda_f32_to_bf16_async(
        &self,
        src: CuteafdDeviceBuffer,
        dst: CuteafdDeviceBuffer,
        count: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        validate_f32_to_bf16_buffers("cuteafd_cuda_f32_to_bf16_async", src, dst, count)?;

        let kernel_fn: Symbol<CudaF32ToBf16AsyncFn> =
            unsafe { self.lib.get(b"cuteafd_cuda_f32_to_bf16_async")? };
        let status = unsafe {
            kernel_fn(
                src.ptr.cast::<f32>() as *const f32,
                dst.ptr.cast::<u16>(),
                count,
                cuda_stream,
            )
        };
        self.status_to_result("cuteafd_cuda_f32_to_bf16_async", status)
    }




    #[allow(clippy::too_many_arguments)]
    pub unsafe fn cuda_gather_rows_f32_to_fp8_e4m3_row_scaled_async(
        &self,
        src: CuteafdDeviceBuffer,
        src_rows: usize,
        row_indices: CuteafdDeviceBuffer,
        dst: CuteafdDeviceBuffer,
        rows: usize,
        row_width: usize,
        dst_row_stride_bytes: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        validate_row_gather_f32_to_fp8_e4m3_row_scaled_buffers(
            "cuteafd_cuda_gather_rows_f32_to_fp8_e4m3_row_scaled_async",
            src,
            src_rows,
            row_indices,
            dst,
            rows,
            row_width,
            dst_row_stride_bytes,
        )?;
        let kernel_fn: Symbol<CudaGatherRowsF32ToFp8E4m3RowScaledAsyncFn> = unsafe {
            self.lib
                .get(b"cuteafd_cuda_gather_rows_f32_to_fp8_e4m3_row_scaled_async")?
        };
        let status = unsafe {
            kernel_fn(
                src.ptr.cast::<f32>() as *const f32,
                row_indices.ptr.cast::<u32>() as *const u32,
                dst.ptr.cast::<u8>(),
                rows,
                row_width,
                dst_row_stride_bytes,
                cuda_stream,
            )
        };
        self.status_to_result(
            "cuteafd_cuda_gather_rows_f32_to_fp8_e4m3_row_scaled_async",
            status,
        )
    }





    #[allow(clippy::too_many_arguments)]
    pub unsafe fn cuda_gather_rows_f32_to_nvfp4_e2m1_fp8_e4m3_async(
        &self,
        src: CuteafdDeviceBuffer,
        src_rows: usize,
        row_indices: CuteafdDeviceBuffer,
        dst: CuteafdDeviceBuffer,
        rows: usize,
        row_width: usize,
        dst_row_stride_bytes: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        validate_row_gather_f32_to_nvfp4_e2m1_fp8_e4m3_buffers(
            "cuteafd_cuda_gather_rows_f32_to_nvfp4_e2m1_fp8_e4m3_async",
            src,
            src_rows,
            row_indices,
            dst,
            rows,
            row_width,
            dst_row_stride_bytes,
        )?;
        let kernel_fn: Symbol<CudaGatherRowsF32ToNvfp4E2m1Fp8E4m3AsyncFn> = unsafe {
            self.lib
                .get(b"cuteafd_cuda_gather_rows_f32_to_nvfp4_e2m1_fp8_e4m3_async")?
        };
        let status = unsafe {
            kernel_fn(
                src.ptr.cast::<f32>() as *const f32,
                row_indices.ptr.cast::<u32>() as *const u32,
                dst.ptr.cast::<u8>(),
                rows,
                row_width,
                dst_row_stride_bytes,
                cuda_stream,
            )
        };
        self.status_to_result(
            "cuteafd_cuda_gather_rows_f32_to_nvfp4_e2m1_fp8_e4m3_async",
            status,
        )
    }














    #[allow(clippy::too_many_arguments)]
    pub unsafe fn cuda_reduce_route_shards_to_f32_async(
        &self,
        buffers: &CuteafdRouteShardReductionBuffers,
        rows: usize,
        row_width: usize,
        peer_row_stride_bytes: usize,
        local_dtype: u32,
        peer_dtype: u32,
        peer_count: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        validate_route_shard_reduction_buffers(
            "cuteafd_cuda_reduce_route_shards_to_f32_async",
            buffers,
            rows,
            row_width,
            peer_row_stride_bytes,
            local_dtype,
            peer_dtype,
            peer_count,
        )?;
        let peer_count = u32::try_from(peer_count).context("route shard peer count exceeds u32")?;
        let kernel_fn: Symbol<CudaReduceRouteShardsToF32AsyncFn> = unsafe {
            self.lib
                .get(b"cuteafd_cuda_reduce_route_shards_to_f32_async")?
        };
        let status = unsafe {
            kernel_fn(
                buffers,
                rows,
                row_width,
                peer_row_stride_bytes,
                local_dtype,
                peer_dtype,
                peer_count,
                cuda_stream,
            )
        };
        self.status_to_result("cuteafd_cuda_reduce_route_shards_to_f32_async", status)
    }




























































































    pub unsafe fn cuda_embedding_lookup_bf16_async(
        &self,
        embedding: CuteafdDeviceBuffer,
        token_ids: CuteafdDeviceBuffer,
        out: CuteafdDeviceBuffer,
        rows: usize,
        vocab: usize,
        hidden: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        validate_embedding_lookup_bf16_buffers(
            "cuteafd_cuda_embedding_lookup_bf16_async",
            embedding,
            token_ids,
            out,
            rows,
            vocab,
            hidden,
        )?;

        let kernel_fn: Symbol<CudaEmbeddingLookupBf16AsyncFn> =
            unsafe { self.lib.get(b"cuteafd_cuda_embedding_lookup_bf16_async")? };
        let status = unsafe {
            kernel_fn(
                embedding.ptr.cast::<u16>() as *const u16,
                token_ids.ptr.cast::<u32>() as *const u32,
                out.ptr.cast::<u16>(),
                rows,
                vocab,
                hidden,
                cuda_stream,
            )
        };
        self.status_to_result("cuteafd_cuda_embedding_lookup_bf16_async", status)
    }










    /// Launch the v4.1 GPU target-sampler's K1 on `cuda_stream` and return
    /// immediately. Buffers must follow `sampling_gpu.h`; every buffer is
    /// validated first, mirroring `validate_logits_argmax_buffers`.
    ///
    /// `params` is the host-side parameter block used for validation;
    /// `params_device` is the device allocation K1 actually reads, one 64-byte
    /// block per row. Passing a host pointer as `params_device` is not
    /// supported: a pageable host address is not device-addressable, even where
    /// a given driver happens to expose it.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn cuda_v41_target_sample_async(
        &self,
        logits: CuteafdDeviceBuffer,
        rows: usize,
        vocab: usize,
        logits_stride: usize,
        params: &[CuteafdV41SamplerRow],
        params_device: CuteafdDeviceBuffer,
        mask_words: Option<CuteafdDeviceBuffer>,
        mask_words_per_row: usize,
        out_indices: CuteafdDeviceBuffer,
        out_status: CuteafdDeviceBuffer,
        out_status_detail: CuteafdDeviceBuffer,
        out_scores: CuteafdDeviceBuffer,
        out_total: Option<CuteafdDeviceBuffer>,
        out_nucleus_count: Option<CuteafdDeviceBuffer>,
        scratch: CuteafdDeviceBuffer,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        const NAME: &str = "cuteafd_cuda_v41_target_sample_async";
        validate_v41_sampling_buffers(
            NAME,
            logits,
            rows,
            vocab,
            logits_stride,
            params,
            mask_words,
            mask_words_per_row,
            out_indices,
            out_status,
            out_status_detail,
            out_scores,
            out_total,
            out_nucleus_count,
            scratch,
        )?;
        validate_v41_params_device(NAME, params_device, rows)?;
        let kernel_fn: Symbol<CudaV41TargetSampleAsyncFn> =
            unsafe { self.lib.get(b"cuteafd_cuda_v41_target_sample_async")? };
        let status = unsafe {
            kernel_fn(
                logits.ptr.cast::<f32>() as *const f32,
                rows,
                vocab,
                logits_stride,
                params_device.ptr.cast::<CuteafdV41SamplerRow>() as *const CuteafdV41SamplerRow,
                mask_words
                    .map(|buffer| buffer.ptr.cast::<u32>() as *const u32)
                    .unwrap_or(std::ptr::null()),
                mask_words_per_row,
                out_indices.ptr.cast::<u32>(),
                out_status.ptr.cast::<u32>(),
                out_status_detail.ptr.cast::<u32>(),
                out_scores.ptr.cast::<f32>(),
                out_total
                    .map(|buffer| buffer.ptr.cast::<f32>())
                    .unwrap_or(std::ptr::null_mut()),
                out_nucleus_count
                    .map(|buffer| buffer.ptr.cast::<u32>())
                    .unwrap_or(std::ptr::null_mut()),
                scratch.ptr,
                cuda_stream,
            )
        };
        self.status_to_result(NAME, status)
    }

    /// Blocking form of [`Self::cuda_v41_target_sample_async`], matching the
    /// synchronizing convention of `sampling.cu`'s `*_f32` entry points.
    #[allow(clippy::too_many_arguments)]
    pub fn cuda_v41_target_sample(
        &self,
        logits: CuteafdDeviceBuffer,
        rows: usize,
        vocab: usize,
        logits_stride: usize,
        params: &[CuteafdV41SamplerRow],
        params_device: CuteafdDeviceBuffer,
        mask_words: Option<CuteafdDeviceBuffer>,
        mask_words_per_row: usize,
        out_indices: CuteafdDeviceBuffer,
        out_status: CuteafdDeviceBuffer,
        out_status_detail: CuteafdDeviceBuffer,
        out_scores: CuteafdDeviceBuffer,
        out_total: Option<CuteafdDeviceBuffer>,
        out_nucleus_count: Option<CuteafdDeviceBuffer>,
        scratch: CuteafdDeviceBuffer,
    ) -> Result<()> {
        const NAME: &str = "cuteafd_cuda_v41_target_sample";
        validate_v41_sampling_buffers(
            NAME,
            logits,
            rows,
            vocab,
            logits_stride,
            params,
            mask_words,
            mask_words_per_row,
            out_indices,
            out_status,
            out_status_detail,
            out_scores,
            out_total,
            out_nucleus_count,
            scratch,
        )?;
        validate_v41_params_device(NAME, params_device, rows)?;
        let kernel_fn: Symbol<CudaV41TargetSampleFn> =
            unsafe { self.lib.get(b"cuteafd_cuda_v41_target_sample")? };
        let status = unsafe {
            kernel_fn(
                logits.ptr.cast::<f32>() as *const f32,
                rows,
                vocab,
                logits_stride,
                params_device.ptr.cast::<CuteafdV41SamplerRow>() as *const CuteafdV41SamplerRow,
                mask_words
                    .map(|buffer| buffer.ptr.cast::<u32>() as *const u32)
                    .unwrap_or(std::ptr::null()),
                mask_words_per_row,
                out_indices.ptr.cast::<u32>(),
                out_status.ptr.cast::<u32>(),
                out_status_detail.ptr.cast::<u32>(),
                out_scores.ptr.cast::<f32>(),
                out_total
                    .map(|buffer| buffer.ptr.cast::<f32>())
                    .unwrap_or(std::ptr::null_mut()),
                out_nucleus_count
                    .map(|buffer| buffer.ptr.cast::<u32>())
                    .unwrap_or(std::ptr::null_mut()),
                scratch.ptr,
            )
        };
        self.status_to_result(NAME, status)
    }

    /// Launch the chunk-3a K3/K4 stages on `cuda_stream` and return immediately.
    ///
    /// These read the `scratch` block a preceding K1 launch filled (the chunk-2
    /// convention) and materialize the retained set in the CPU's exact rank
    /// order. `rank_order_ids`/`rank_order_scratch` may both be `None` for a
    /// selection-only call (`rank_order_capacity == 0`); when present they must
    /// be one `rows * rank_order_capacity` arena each and
    /// `rank_order_capacity` must cover every row's `top_k`.
    ///
    /// `params` is the host slice the validator reads; `params_device` is the
    /// device allocation the kernel dereferences (`params[blockIdx.x]`), one
    /// 64-byte block per row, validated by `validate_v41_params_device` exactly
    /// as K1's. A pageable host address is not device-addressable under CUDA's
    /// documented model even where a driver happens to expose it.
    ///
    /// See the rank-order contract in `sampling_gpu.h`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn cuda_v41_topk_select_async(
        &self,
        logits: CuteafdDeviceBuffer,
        rows: usize,
        vocab: usize,
        logits_stride: usize,
        params: &[CuteafdV41SamplerRow],
        params_device: CuteafdDeviceBuffer,
        mask_words: Option<CuteafdDeviceBuffer>,
        mask_words_per_row: usize,
        rank_order_ids: Option<CuteafdDeviceBuffer>,
        rank_order_scratch: Option<CuteafdDeviceBuffer>,
        rank_order_capacity: usize,
        out_retained_count: CuteafdDeviceBuffer,
        out_pivot_passes: CuteafdDeviceBuffer,
        scratch: CuteafdDeviceBuffer,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        const NAME: &str = "cuteafd_cuda_v41_topk_select_async";
        validate_v41_topk_select_buffers(
            NAME,
            logits,
            rows,
            vocab,
            logits_stride,
            params,
            mask_words,
            mask_words_per_row,
            rank_order_ids,
            rank_order_scratch,
            rank_order_capacity,
            out_retained_count,
            out_pivot_passes,
            scratch,
        )?;
        validate_v41_params_device(NAME, params_device, rows)?;
        let kernel_fn: Symbol<CudaV41TopkSelectAsyncFn> =
            unsafe { self.lib.get(b"cuteafd_cuda_v41_topk_select_async")? };
        let status = unsafe {
            kernel_fn(
                logits.ptr.cast::<f32>() as *const f32,
                rows,
                vocab,
                logits_stride,
                params_device.ptr.cast::<CuteafdV41SamplerRow>() as *const CuteafdV41SamplerRow,
                mask_words
                    .map(|buffer| buffer.ptr.cast::<u32>() as *const u32)
                    .unwrap_or(std::ptr::null()),
                mask_words_per_row,
                rank_order_ids
                    .map(|buffer| buffer.ptr.cast::<u32>())
                    .unwrap_or(std::ptr::null_mut()),
                rank_order_scratch
                    .map(|buffer| buffer.ptr.cast::<u64>())
                    .unwrap_or(std::ptr::null_mut()),
                rank_order_capacity,
                out_retained_count.ptr.cast::<u32>(),
                out_pivot_passes.ptr.cast::<u32>(),
                scratch.ptr,
                cuda_stream,
            )
        };
        self.status_to_result(NAME, status)
    }

    /// Blocking form of [`Self::cuda_v41_topk_select_async`].
    #[allow(clippy::too_many_arguments)]
    pub fn cuda_v41_topk_select(
        &self,
        logits: CuteafdDeviceBuffer,
        rows: usize,
        vocab: usize,
        logits_stride: usize,
        params: &[CuteafdV41SamplerRow],
        params_device: CuteafdDeviceBuffer,
        mask_words: Option<CuteafdDeviceBuffer>,
        mask_words_per_row: usize,
        rank_order_ids: Option<CuteafdDeviceBuffer>,
        rank_order_scratch: Option<CuteafdDeviceBuffer>,
        rank_order_capacity: usize,
        out_retained_count: CuteafdDeviceBuffer,
        out_pivot_passes: CuteafdDeviceBuffer,
        scratch: CuteafdDeviceBuffer,
    ) -> Result<()> {
        const NAME: &str = "cuteafd_cuda_v41_topk_select";
        validate_v41_topk_select_buffers(
            NAME,
            logits,
            rows,
            vocab,
            logits_stride,
            params,
            mask_words,
            mask_words_per_row,
            rank_order_ids,
            rank_order_scratch,
            rank_order_capacity,
            out_retained_count,
            out_pivot_passes,
            scratch,
        )?;
        validate_v41_params_device(NAME, params_device, rows)?;
        let kernel_fn: Symbol<CudaV41TopkSelectFn> =
            unsafe { self.lib.get(b"cuteafd_cuda_v41_topk_select")? };
        let status = unsafe {
            kernel_fn(
                logits.ptr.cast::<f32>() as *const f32,
                rows,
                vocab,
                logits_stride,
                params_device.ptr.cast::<CuteafdV41SamplerRow>() as *const CuteafdV41SamplerRow,
                mask_words
                    .map(|buffer| buffer.ptr.cast::<u32>() as *const u32)
                    .unwrap_or(std::ptr::null()),
                mask_words_per_row,
                rank_order_ids
                    .map(|buffer| buffer.ptr.cast::<u32>())
                    .unwrap_or(std::ptr::null_mut()),
                rank_order_scratch
                    .map(|buffer| buffer.ptr.cast::<u64>())
                    .unwrap_or(std::ptr::null_mut()),
                rank_order_capacity,
                out_retained_count.ptr.cast::<u32>(),
                out_pivot_passes.ptr.cast::<u32>(),
                scratch.ptr,
            )
        };
        self.status_to_result(NAME, status)
    }

    /// Chunk-3b K5: the inclusive-prefix top-p nucleus and the rank-order draw
    /// on the same stream after K1 and K3/K4. Reads K1's `scratch` and the
    /// rank-order retained ids K3/K4 wrote (indexed by block row); writes
    /// `out_indices` (and, when supplied, the diagnostic `out_total` and
    /// `out_nucleus_count`) indexed by `output_row`. `out_total` and
    /// `out_nucleus_count` are only written for rows whose `DIAGNOSE` flag is
    /// set. See the K5 contract in `sampling_gpu.h`.
    ///
    /// `params` is the host slice the validator reads; `params_device` is the
    /// device allocation the kernel dereferences, validated by
    /// `validate_v41_params_device` (the same contract K1 uses).
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn cuda_v41_nucleus_async(
        &self,
        logits: CuteafdDeviceBuffer,
        rows: usize,
        vocab: usize,
        logits_stride: usize,
        params: &[CuteafdV41SamplerRow],
        params_device: CuteafdDeviceBuffer,
        mask_words: Option<CuteafdDeviceBuffer>,
        mask_words_per_row: usize,
        rank_order_ids: Option<CuteafdDeviceBuffer>,
        rank_order_capacity: usize,
        rank_retained_count: CuteafdDeviceBuffer,
        out_indices: CuteafdDeviceBuffer,
        out_status: CuteafdDeviceBuffer,
        out_total: Option<CuteafdDeviceBuffer>,
        out_nucleus_count: Option<CuteafdDeviceBuffer>,
        scratch: CuteafdDeviceBuffer,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        const NAME: &str = "cuteafd_cuda_v41_nucleus_async";
        validate_v41_nucleus_buffers(
            NAME,
            logits,
            rows,
            vocab,
            logits_stride,
            params,
            mask_words,
            mask_words_per_row,
            rank_order_ids,
            rank_order_capacity,
            rank_retained_count,
            out_indices,
            out_status,
            out_total,
            out_nucleus_count,
            scratch,
        )?;
        validate_v41_params_device(NAME, params_device, rows)?;
        let kernel_fn: Symbol<CudaV41NucleusAsyncFn> =
            unsafe { self.lib.get(b"cuteafd_cuda_v41_nucleus_async")? };
        let status = unsafe {
            kernel_fn(
                logits.ptr.cast::<f32>() as *const f32,
                rows,
                vocab,
                logits_stride,
                params_device.ptr.cast::<CuteafdV41SamplerRow>() as *const CuteafdV41SamplerRow,
                mask_words
                    .map(|buffer| buffer.ptr.cast::<u32>() as *const u32)
                    .unwrap_or(std::ptr::null()),
                mask_words_per_row,
                rank_order_ids
                    .map(|buffer| buffer.ptr.cast::<u32>() as *const u32)
                    .unwrap_or(std::ptr::null()),
                rank_order_capacity,
                rank_retained_count.ptr.cast::<u32>() as *const u32,
                out_indices.ptr.cast::<u32>(),
                out_status.ptr.cast::<u32>(),
                out_total
                    .map(|buffer| buffer.ptr.cast::<f32>())
                    .unwrap_or(std::ptr::null_mut()),
                out_nucleus_count
                    .map(|buffer| buffer.ptr.cast::<u32>())
                    .unwrap_or(std::ptr::null_mut()),
                scratch.ptr,
                cuda_stream,
            )
        };
        self.status_to_result(NAME, status)
    }

    /// Blocking form of [`Self::cuda_v41_nucleus_async`].
    #[allow(clippy::too_many_arguments)]
    pub fn cuda_v41_nucleus(
        &self,
        logits: CuteafdDeviceBuffer,
        rows: usize,
        vocab: usize,
        logits_stride: usize,
        params: &[CuteafdV41SamplerRow],
        params_device: CuteafdDeviceBuffer,
        mask_words: Option<CuteafdDeviceBuffer>,
        mask_words_per_row: usize,
        rank_order_ids: Option<CuteafdDeviceBuffer>,
        rank_order_capacity: usize,
        rank_retained_count: CuteafdDeviceBuffer,
        out_indices: CuteafdDeviceBuffer,
        out_status: CuteafdDeviceBuffer,
        out_total: Option<CuteafdDeviceBuffer>,
        out_nucleus_count: Option<CuteafdDeviceBuffer>,
        scratch: CuteafdDeviceBuffer,
    ) -> Result<()> {
        const NAME: &str = "cuteafd_cuda_v41_nucleus";
        validate_v41_nucleus_buffers(
            NAME,
            logits,
            rows,
            vocab,
            logits_stride,
            params,
            mask_words,
            mask_words_per_row,
            rank_order_ids,
            rank_order_capacity,
            rank_retained_count,
            out_indices,
            out_status,
            out_total,
            out_nucleus_count,
            scratch,
        )?;
        validate_v41_params_device(NAME, params_device, rows)?;
        let kernel_fn: Symbol<CudaV41NucleusFn> =
            unsafe { self.lib.get(b"cuteafd_cuda_v41_nucleus")? };
        let status = unsafe {
            kernel_fn(
                logits.ptr.cast::<f32>() as *const f32,
                rows,
                vocab,
                logits_stride,
                params_device.ptr.cast::<CuteafdV41SamplerRow>() as *const CuteafdV41SamplerRow,
                mask_words
                    .map(|buffer| buffer.ptr.cast::<u32>() as *const u32)
                    .unwrap_or(std::ptr::null()),
                mask_words_per_row,
                rank_order_ids
                    .map(|buffer| buffer.ptr.cast::<u32>() as *const u32)
                    .unwrap_or(std::ptr::null()),
                rank_order_capacity,
                rank_retained_count.ptr.cast::<u32>() as *const u32,
                out_indices.ptr.cast::<u32>(),
                out_status.ptr.cast::<u32>(),
                out_total
                    .map(|buffer| buffer.ptr.cast::<f32>())
                    .unwrap_or(std::ptr::null_mut()),
                out_nucleus_count
                    .map(|buffer| buffer.ptr.cast::<u32>())
                    .unwrap_or(std::ptr::null_mut()),
                scratch.ptr,
            )
        };
        self.status_to_result(NAME, status)
    }



    #[allow(clippy::too_many_arguments)]
    pub unsafe fn cuda_logits_argmax_checked_f32_async(
        &self,
        logits: CuteafdDeviceBuffer,
        out_indices: CuteafdDeviceBuffer,
        out_scores: CuteafdDeviceBuffer,
        rows: usize,
        vocab: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        validate_logits_argmax_buffers(
            "cuteafd_cuda_logits_argmax_checked_f32_async",
            logits,
            out_indices,
            out_scores,
            rows,
            vocab,
        )?;

        let kernel_fn: Symbol<CudaLogitsArgmaxF32AsyncFn> =
            unsafe { self.lib.get(b"cuteafd_cuda_logits_argmax_checked_f32_async")? };
        let status = unsafe {
            kernel_fn(
                logits.ptr.cast::<f32>() as *const f32,
                out_indices.ptr.cast::<u32>(),
                out_scores.ptr.cast::<f32>(),
                rows,
                vocab,
                cuda_stream,
            )
        };
        self.status_to_result("cuteafd_cuda_logits_argmax_checked_f32_async", status)
    }







    pub fn rdma_device_info(&self) -> Result<CuteafdRdmaDeviceInfo> {
        let info_fn: Symbol<RdmaDeviceInfoFn> = unsafe { self.lib.get(b"cuteafd_rdma_device_info")? };
        let mut info = CuteafdRdmaDeviceInfo::default();
        let status = unsafe { info_fn(&mut info) };
        self.status_to_result("cuteafd_rdma_device_info", status)?;
        Ok(info)
    }

    pub fn rdma_plan_host_buffer_registration(
        &self,
        ptr: *const c_void,
        bytes: usize,
        alignment: usize,
    ) -> Result<CuteafdRdmaHostBufferPlan> {
        let plan_fn: Symbol<RdmaPlanHostBufferRegistrationFn> =
            unsafe { self.lib.get(b"cuteafd_rdma_plan_host_buffer_registration")? };
        let mut plan = CuteafdRdmaHostBufferPlan::default();
        let status = unsafe { plan_fn(ptr, bytes, alignment, &mut plan) };
        self.status_to_result("cuteafd_rdma_plan_host_buffer_registration", status)?;
        Ok(plan)
    }

    pub fn rdma_register_host_buffer_probe(
        &self,
        buffer: &mut [u8],
    ) -> Result<CuteafdRdmaRegisterProbe> {
        let register_fn: Symbol<RdmaRegisterHostBufferProbeFn> =
            unsafe { self.lib.get(b"cuteafd_rdma_register_host_buffer_probe")? };
        let mut probe = CuteafdRdmaRegisterProbe::default();
        let status = unsafe { register_fn(buffer.as_mut_ptr().cast(), buffer.len(), &mut probe) };
        self.status_to_result("cuteafd_rdma_register_host_buffer_probe", status)?;
        Ok(probe)
    }

    pub fn rdma_create_rc_qp_probe(
        &self,
        port_num: u32,
        send_wr: u32,
        recv_wr: u32,
        max_sge: u32,
    ) -> Result<CuteafdRdmaRcQpProbe> {
        let qp_fn: Symbol<RdmaCreateRcQpProbeFn> =
            unsafe { self.lib.get(b"cuteafd_rdma_create_rc_qp_probe")? };
        let mut probe = CuteafdRdmaRcQpProbe::default();
        let status = unsafe { qp_fn(port_num, send_wr, recv_wr, max_sge, &mut probe) };
        self.status_to_result("cuteafd_rdma_create_rc_qp_probe", status)?;
        Ok(probe)
    }

    pub fn rdma_rc_send_recv_loopback_probe(
        &self,
        port_num: u32,
        bytes: usize,
    ) -> Result<CuteafdRdmaRcSendRecvProbe> {
        let probe_fn: Symbol<RdmaRcSendRecvLoopbackProbeFn> =
            unsafe { self.lib.get(b"cuteafd_rdma_rc_send_recv_loopback_probe")? };
        let mut probe = CuteafdRdmaRcSendRecvProbe::default();
        let status = unsafe { probe_fn(port_num, bytes, &mut probe) };
        self.status_to_result("cuteafd_rdma_rc_send_recv_loopback_probe", status)?;
        Ok(probe)
    }

    pub fn rdma_rc_protocol_v2_loopback_probe(
        &self,
        port_num: u32,
        request_frame: &[u8],
        response_frame: &[u8],
    ) -> Result<CuteafdRdmaRcProtocolV2LoopbackProbe> {
        let probe_fn: Symbol<RdmaRcProtocolV2LoopbackProbeFn> =
            unsafe { self.lib.get(b"cuteafd_rdma_rc_protocol_v2_loopback_probe")? };
        let mut probe = CuteafdRdmaRcProtocolV2LoopbackProbe::default();
        let status = unsafe {
            probe_fn(
                port_num,
                request_frame.as_ptr().cast(),
                request_frame.len(),
                response_frame.as_ptr().cast(),
                response_frame.len(),
                &mut probe,
            )
        };
        self.status_to_result("cuteafd_rdma_rc_protocol_v2_loopback_probe", status)?;
        Ok(probe)
    }

    pub fn rdma_rc_endpoint_create(
        &self,
        port_num: u32,
        local_psn: u32,
        send_frame_bytes: usize,
        recv_frame_bytes: usize,
        send_registered_span_bytes: usize,
        recv_registered_span_bytes: usize,
        max_send_wr: u32,
        max_recv_wr: u32,
        max_sge: u32,
    ) -> Result<CuteafdRdmaRcEndpointInfo> {
        let create_fn: Symbol<RdmaRcEndpointCreateFn> =
            unsafe { self.lib.get(b"cuteafd_rdma_rc_endpoint_create")? };
        let mut info = CuteafdRdmaRcEndpointInfo::default();
        let status = unsafe {
            create_fn(
                port_num,
                local_psn,
                send_frame_bytes,
                recv_frame_bytes,
                send_registered_span_bytes,
                recv_registered_span_bytes,
                max_send_wr,
                max_recv_wr,
                max_sge,
                &mut info,
            )
        };
        self.status_to_result("cuteafd_rdma_rc_endpoint_create", status)?;
        record_rdma_rings(info.handle, send_registered_span_bytes + recv_registered_span_bytes);
        Ok(info)
    }

    pub fn rdma_rc_endpoint_create_with_buffer_flags(
        &self,
        port_num: u32,
        local_psn: u32,
        send_frame_bytes: usize,
        recv_frame_bytes: usize,
        send_registered_span_bytes: usize,
        recv_registered_span_bytes: usize,
        max_send_wr: u32,
        max_recv_wr: u32,
        max_sge: u32,
        host_buffer_flags: u64,
    ) -> Result<CuteafdRdmaRcEndpointInfo> {
        let create_fn: Symbol<RdmaRcEndpointCreateWithBufferFlagsFn> = unsafe {
            self.lib
                .get(b"cuteafd_rdma_rc_endpoint_create_with_buffer_flags")?
        };
        let mut info = CuteafdRdmaRcEndpointInfo::default();
        let status = unsafe {
            create_fn(
                port_num,
                local_psn,
                send_frame_bytes,
                recv_frame_bytes,
                send_registered_span_bytes,
                recv_registered_span_bytes,
                max_send_wr,
                max_recv_wr,
                max_sge,
                host_buffer_flags,
                &mut info,
            )
        };
        self.status_to_result("cuteafd_rdma_rc_endpoint_create_with_buffer_flags", status)?;
        record_rdma_rings(info.handle, send_registered_span_bytes + recv_registered_span_bytes);
        Ok(info)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn rdma_rc_endpoint_create_on_device_with_buffer_flags(
        &self,
        device_name: &str,
        port_num: u32,
        local_psn: u32,
        send_frame_bytes: usize,
        recv_frame_bytes: usize,
        send_registered_span_bytes: usize,
        recv_registered_span_bytes: usize,
        max_send_wr: u32,
        max_recv_wr: u32,
        max_sge: u32,
        host_buffer_flags: u64,
    ) -> Result<CuteafdRdmaRcEndpointInfo> {
        let create_fn: Symbol<RdmaRcEndpointCreateOnDeviceWithBufferFlagsFn> = unsafe {
            self.lib
                .get(b"cuteafd_rdma_rc_endpoint_create_on_device_with_buffer_flags")?
        };
        let device_name = CString::new(device_name).context("RDMA device name contains NUL")?;
        let mut info = CuteafdRdmaRcEndpointInfo::default();
        let status = unsafe {
            create_fn(
                device_name.as_ptr(),
                port_num,
                local_psn,
                send_frame_bytes,
                recv_frame_bytes,
                send_registered_span_bytes,
                recv_registered_span_bytes,
                max_send_wr,
                max_recv_wr,
                max_sge,
                host_buffer_flags,
                &mut info,
            )
        };
        self.status_to_result(
            "cuteafd_rdma_rc_endpoint_create_on_device_with_buffer_flags",
            status,
        )?;
        record_rdma_rings(info.handle, send_registered_span_bytes + recv_registered_span_bytes);
        Ok(info)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn rdma_rc_endpoint_create_on_gid_with_buffer_flags(
        &self,
        device_name: &str,
        port_num: u32,
        gid_index: u32,
        local_psn: u32,
        send_frame_bytes: usize,
        recv_frame_bytes: usize,
        send_registered_span_bytes: usize,
        recv_registered_span_bytes: usize,
        max_send_wr: u32,
        max_recv_wr: u32,
        max_sge: u32,
        host_buffer_flags: u64,
    ) -> Result<CuteafdRdmaRcEndpointInfo> {
        // SAFETY: the symbol has the declared C ABI; all argument buffers live through the call.
        let create_fn: Symbol<RdmaRcEndpointCreateOnGidWithBufferFlagsFn> = unsafe {
            self.lib
                .get(b"cuteafd_rdma_rc_endpoint_create_on_gid_with_buffer_flags")?
        };
        let device_name = CString::new(device_name).context("RDMA device name contains NUL")?;
        let mut info = CuteafdRdmaRcEndpointInfo::default();
        // SAFETY: device_name and info remain valid for the synchronous native call.
        let status = unsafe {
            create_fn(
                device_name.as_ptr(),
                port_num,
                gid_index,
                local_psn,
                send_frame_bytes,
                recv_frame_bytes,
                send_registered_span_bytes,
                recv_registered_span_bytes,
                max_send_wr,
                max_recv_wr,
                max_sge,
                host_buffer_flags,
                &mut info,
            )
        };
        self.status_to_result(
            "cuteafd_rdma_rc_endpoint_create_on_gid_with_buffer_flags",
            status,
        )?;
        record_rdma_rings(info.handle, send_registered_span_bytes + recv_registered_span_bytes);
        Ok(info)
    }

    pub fn rdma_rc_endpoint_buffer_view(
        &self,
        handle: *mut c_void,
        receive_buffer: bool,
    ) -> Result<CuteafdRdmaRcEndpointBufferView> {
        let view_fn: Symbol<RdmaRcEndpointBufferViewFn> =
            unsafe { self.lib.get(b"cuteafd_rdma_rc_endpoint_buffer_view")? };
        let mut view = CuteafdRdmaRcEndpointBufferView::default();
        let status = unsafe { view_fn(handle, c_int::from(receive_buffer), &mut view) };
        self.status_to_result("cuteafd_rdma_rc_endpoint_buffer_view", status)?;
        Ok(view)
    }

    pub fn rdma_rc_endpoint_connect(
        &self,
        handle: *mut c_void,
        remote_qp_num: u32,
        remote_psn: u32,
        remote_lid: u32,
        remote_gid_hex: &str,
    ) -> Result<()> {
        let connect_fn: Symbol<RdmaRcEndpointConnectFn> =
            unsafe { self.lib.get(b"cuteafd_rdma_rc_endpoint_connect")? };
        let remote_gid_hex =
            CString::new(remote_gid_hex).context("RDMA remote GID contains nul byte")?;
        let status = unsafe {
            connect_fn(
                handle,
                remote_qp_num,
                remote_psn,
                remote_lid,
                remote_gid_hex.as_ptr(),
            )
        };
        self.status_to_result("cuteafd_rdma_rc_endpoint_connect", status)
    }

    /// [`Self::rdma_rc_endpoint_connect`] with an explicit RoCE v2 flow label
    /// (20 bits), from which the NIC derives the QP's UDP source port; 0 keeps
    /// the kernel's label derived from both QP numbers. Libraries without
    /// `cuteafd_rdma_rc_endpoint_connect_flow_label` return an error.
    pub fn rdma_rc_endpoint_connect_flow_label(
        &self,
        handle: *mut c_void,
        remote_qp_num: u32,
        remote_psn: u32,
        remote_lid: u32,
        remote_gid_hex: &str,
        flow_label: u32,
    ) -> Result<()> {
        type ConnectFlowLabelFn =
            unsafe extern "C" fn(*mut c_void, u32, u32, u32, *const c_char, u32) -> CuteafdStatus;
        // SAFETY: resolving the symbol does not call it.
        let connect: Symbol<ConnectFlowLabelFn> =
            unsafe { self.lib.get(b"cuteafd_rdma_rc_endpoint_connect_flow_label")? };
        let remote_gid_hex =
            CString::new(remote_gid_hex).context("RDMA remote GID contains nul byte")?;
        // SAFETY: `handle` is a live endpoint owned by the caller and the GID
        // string outlives the call.
        let status = unsafe {
            connect(handle, remote_qp_num, remote_psn, remote_lid, remote_gid_hex.as_ptr(), flow_label)
        };
        self.status_to_result("cuteafd_rdma_rc_endpoint_connect_flow_label", status)
    }

    /// Registers the first `bytes` of the endpoint's send buffer for remote
    /// reads and returns its (address, rkey).
    pub fn rdma_rc_endpoint_expose_send_read(&self, handle: *mut c_void, bytes: usize) -> Result<(u64, u32)> {
        type ExposeReadFn = unsafe extern "C" fn(*mut c_void, usize, *mut u64, *mut u32) -> CuteafdStatus;
        // SAFETY: resolving the symbol does not call it.
        let expose: Symbol<ExposeReadFn> =
            unsafe { self.lib.get(b"cuteafd_rdma_rc_endpoint_expose_send_read")? };
        let (mut addr, mut rkey) = (0_u64, 0_u32);
        // SAFETY: `handle` is a live endpoint; the registration covers its own
        // send buffer and is released with it.
        let status = unsafe { expose(handle, bytes, &mut addr, &mut rkey) };
        self.status_to_result("cuteafd_rdma_rc_endpoint_expose_send_read", status)?;
        Ok((addr, rkey))
    }

    /// RDMA-reads `bytes` from the peer's `remote_addr`/`rkey` into the
    /// endpoint's receive buffer at `offset` and waits up to `timeout_ms`.
    pub fn rdma_rc_endpoint_read_wait(
        &self,
        handle: *mut c_void,
        offset: usize,
        bytes: usize,
        remote_addr: u64,
        rkey: u32,
        timeout_ms: u32,
    ) -> Result<()> {
        type ReadWaitFn = unsafe extern "C" fn(*mut c_void, usize, usize, u64, u32, u32) -> CuteafdStatus;
        // SAFETY: resolving the symbol does not call it.
        let read: Symbol<ReadWaitFn> = unsafe { self.lib.get(b"cuteafd_rdma_rc_endpoint_read_wait")? };
        // SAFETY: `handle` is a live connected endpoint; the read lands in its
        // own registered receive buffer, which the native side bounds-checks.
        let status = unsafe { read(handle, offset, bytes, remote_addr, rkey, timeout_ms) };
        self.status_to_result("cuteafd_rdma_rc_endpoint_read_wait", status)
    }

    pub fn rdma_rc_endpoint_post_recv(
        &self,
        handle: *mut c_void,
        bytes: usize,
        wr_id: u64,
    ) -> Result<()> {
        let recv_fn: Symbol<RdmaRcEndpointPostRecvFn> =
            unsafe { self.lib.get(b"cuteafd_rdma_rc_endpoint_post_recv")? };
        let status = unsafe { recv_fn(handle, bytes, wr_id) };
        self.status_to_result("cuteafd_rdma_rc_endpoint_post_recv", status)
    }

    pub fn rdma_rc_endpoint_post_recv_at(
        &self,
        handle: *mut c_void,
        offset_bytes: usize,
        bytes: usize,
        wr_id: u64,
    ) -> Result<()> {
        let recv_fn: Symbol<RdmaRcEndpointPostRecvAtFn> =
            unsafe { self.lib.get(b"cuteafd_rdma_rc_endpoint_post_recv_at")? };
        let status = unsafe { recv_fn(handle, offset_bytes, bytes, wr_id) };
        self.status_to_result("cuteafd_rdma_rc_endpoint_post_recv_at", status)
    }

    /// Scatter each later receive: its first `header_bytes` into the host
    /// slot, the rest into `device` (registered over dma-buf). `None` restores
    /// host-only receives.
    ///
    /// # Safety
    /// `handle` is a live endpoint, and `device` must stay allocated, and not
    /// be read while a receive may still write it, until the endpoint is
    /// destroyed or its landing is cleared.
    pub unsafe fn rdma_rc_endpoint_set_recv_landing(
        &self,
        handle: *mut c_void,
        device: Option<CuteafdDeviceBuffer>,
        header_bytes: usize,
    ) -> Result<()> {
        let set_fn: Symbol<RdmaRcEndpointSetRecvLandingFn> =
            unsafe { self.lib.get(b"cuteafd_rdma_rc_endpoint_set_recv_landing")? };
        let (ptr, bytes) = device.map_or((std::ptr::null_mut(), 0), |d| (d.ptr, d.bytes));
        // SAFETY: the caller guarantees the handle and range contract above.
        let status = unsafe { set_fn(handle, ptr, bytes, header_bytes) };
        self.status_to_result("cuteafd_rdma_rc_endpoint_set_recv_landing", status)
    }

    /// Probe dma-buf GPU landing on the current CUDA device and time loopback
    /// SENDs of `bytes` landing in device vs pinned host memory. The probe
    /// struct is returned even when landing is unavailable (see `status`).
    pub fn rdma_gpu_landing_probe(&self, device_name: Option<&str>, port_num: u32, bytes: usize,
        iterations: u32) -> Result<(CuteafdRdmaGpuLandingProbe, Option<String>)> {
        let probe_fn: Symbol<RdmaGpuLandingProbeFn> =
            unsafe { self.lib.get(b"cuteafd_rdma_gpu_landing_probe")? };
        let name = device_name.map(std::ffi::CString::new).transpose()?;
        let mut out = CuteafdRdmaGpuLandingProbe::default();
        // SAFETY: the name is NUL-terminated or null, and `out` is a live struct.
        let status = unsafe {
            probe_fn(name.as_ref().map_or(std::ptr::null(), |n| n.as_ptr()), port_num, bytes, iterations, &mut out)
        };
        let failure = self.status_to_result("cuteafd_rdma_gpu_landing_probe", status).err().map(|e| e.to_string());
        Ok((out, failure))
    }

    pub fn rdma_rc_endpoint_post_send_at(
        &self,
        handle: *mut c_void,
        offset_bytes: usize,
        bytes: usize,
        wr_id: u64,
    ) -> Result<()> {
        let send_fn: Symbol<RdmaRcEndpointPostSendAtFn> =
            unsafe { self.lib.get(b"cuteafd_rdma_rc_endpoint_post_send_at")? };
        let status = unsafe { send_fn(handle, offset_bytes, bytes, wr_id) };
        self.status_to_result("cuteafd_rdma_rc_endpoint_post_send_at", status)
    }

    pub fn rdma_rc_endpoint_send(
        &self,
        handle: *mut c_void,
        frame: &[u8],
        wr_id: u64,
    ) -> Result<()> {
        let send_fn: Symbol<RdmaRcEndpointSendFn> =
            unsafe { self.lib.get(b"cuteafd_rdma_rc_endpoint_send")? };
        let status = unsafe { send_fn(handle, frame.as_ptr().cast(), frame.len(), wr_id) };
        self.status_to_result("cuteafd_rdma_rc_endpoint_send", status)
    }

    pub fn rdma_rc_endpoint_send_at(
        &self,
        handle: *mut c_void,
        frame: &[u8],
        offset_bytes: usize,
        wr_id: u64,
    ) -> Result<()> {
        let send_fn: Symbol<RdmaRcEndpointSendAtFn> =
            unsafe { self.lib.get(b"cuteafd_rdma_rc_endpoint_send_at")? };
        let status = unsafe {
            send_fn(
                handle,
                frame.as_ptr().cast(),
                offset_bytes,
                frame.len(),
                wr_id,
            )
        };
        self.status_to_result("cuteafd_rdma_rc_endpoint_send_at", status)
    }

    pub fn rdma_rc_endpoint_send_parts_at(
        &self,
        handle: *mut c_void,
        prefix: &[u8],
        payload: &[u8],
        offset_bytes: usize,
        wr_id: u64,
    ) -> Result<()> {
        let send_fn: Symbol<RdmaRcEndpointSendPartsAtFn> =
            unsafe { self.lib.get(b"cuteafd_rdma_rc_endpoint_send_parts_at")? };
        let status = unsafe {
            send_fn(
                handle,
                prefix.as_ptr().cast(),
                prefix.len(),
                payload.as_ptr().cast(),
                payload.len(),
                offset_bytes,
                wr_id,
            )
        };
        self.status_to_result("cuteafd_rdma_rc_endpoint_send_parts_at", status)
    }

    pub fn rdma_rc_endpoint_poll(
        &self,
        handle: *mut c_void,
        expected_send_completions: u32,
        expected_recv_completions: u32,
        max_poll_iterations: u32,
        active_event_poll_timeout_ms: u32,
    ) -> Result<CuteafdRdmaRcCompletionStats> {
        let mut stats = CuteafdRdmaRcCompletionStats::default();
        if let Ok(poll_fn) = unsafe {
            self.lib
                .get::<RdmaRcEndpointPollWithTimeoutFn>(b"cuteafd_rdma_rc_endpoint_poll_with_timeout")
        } {
            let status = unsafe {
                poll_fn(
                    handle,
                    expected_send_completions,
                    expected_recv_completions,
                    max_poll_iterations,
                    active_event_poll_timeout_ms,
                    &mut stats,
                )
            };
            self.status_to_result("cuteafd_rdma_rc_endpoint_poll_with_timeout", status)?;
        } else {
            let poll_fn: Symbol<RdmaRcEndpointPollFn> =
                unsafe { self.lib.get(b"cuteafd_rdma_rc_endpoint_poll")? };
            let status = unsafe {
                poll_fn(
                    handle,
                    expected_send_completions,
                    expected_recv_completions,
                    max_poll_iterations,
                    &mut stats,
                )
            };
            self.status_to_result("cuteafd_rdma_rc_endpoint_poll", status)?;
        }
        Ok(stats)
    }

    pub fn rdma_rc_endpoint_try_poll(
        &self,
        handle: *mut c_void,
        max_send_completions: u32,
        max_recv_completions: u32,
    ) -> Result<CuteafdRdmaRcCompletionStats> {
        let mut stats = CuteafdRdmaRcCompletionStats::default();
        let status = unsafe {
            (self.rdma_rc_endpoint_try_poll_fn)(
                handle,
                max_send_completions,
                max_recv_completions,
                &mut stats,
            )
        };
        self.status_to_result("cuteafd_rdma_rc_endpoint_try_poll", status)?;
        Ok(stats)
    }

    pub fn rdma_rc_endpoint_copy_recv(
        &self,
        handle: *mut c_void,
        out: &mut [u8],
        bytes: usize,
    ) -> Result<()> {
        let copy_fn: Symbol<RdmaRcEndpointCopyRecvFn> =
            unsafe { self.lib.get(b"cuteafd_rdma_rc_endpoint_copy_recv")? };
        let status = unsafe { copy_fn(handle, out.as_mut_ptr().cast(), out.len(), bytes) };
        self.status_to_result("cuteafd_rdma_rc_endpoint_copy_recv", status)
    }

    pub fn rdma_rc_endpoint_copy_recv_at(
        &self,
        handle: *mut c_void,
        out: &mut [u8],
        offset_bytes: usize,
        bytes: usize,
    ) -> Result<()> {
        let copy_fn: Symbol<RdmaRcEndpointCopyRecvAtFn> =
            unsafe { self.lib.get(b"cuteafd_rdma_rc_endpoint_copy_recv_at")? };
        let status = unsafe {
            copy_fn(
                handle,
                out.as_mut_ptr().cast(),
                out.len(),
                offset_bytes,
                bytes,
            )
        };
        self.status_to_result("cuteafd_rdma_rc_endpoint_copy_recv_at", status)
    }

    /// Registers `[ptr, ptr + bytes)` on the endpoint's protection domain for
    /// [`Self::rdma_rc_endpoint_post_send_slot_region`]; returns the region index.
    ///
    /// # Safety
    /// The range must be host memory that stays allocated until the endpoint is
    /// destroyed.
    /// Registers a device range for remote RDMA writes on the endpoint and
    /// returns its rkey (`None` removes it).
    ///
    /// # Safety
    /// `handle` is a live endpoint; the range stays allocated until removed or
    /// the endpoint is destroyed.
    pub unsafe fn rdma_rc_endpoint_expose_device(&self, handle: *mut c_void, device: Option<CuteafdDeviceBuffer>)
        -> Result<u32> {
        type ExposeFn = unsafe extern "C" fn(*mut c_void, *mut c_void, usize, *mut u32) -> CuteafdStatus;
        let expose: Symbol<ExposeFn> = unsafe { self.lib.get(b"cuteafd_rdma_rc_endpoint_expose_device")? };
        let (ptr, bytes) = device.map_or((std::ptr::null_mut(), 0), |d| (d.ptr, d.bytes));
        let mut rkey = 0;
        let status = unsafe { expose(handle, ptr, bytes, &mut rkey) };
        self.status_to_result("cuteafd_rdma_rc_endpoint_expose_device", status)?;
        Ok(rkey)
    }

    /// RDMA-writes `bytes` at `offset` of the send buffer to `remote`, then
    /// `flag_value` to `flag_remote` (signaled with `wr_id`).
    ///
    /// # Safety
    /// `handle` is a live connected endpoint; the remote ranges were exposed
    /// by the peer with these rkeys and hold the written extents.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn rdma_rc_endpoint_post_write_flagged(&self, handle: *mut c_void, offset: usize, bytes: usize,
        remote: u64, rkey: u32, flag_value: u64, flag_remote: u64, flag_rkey: u32, wr_id: u64) -> Result<()> {
        type WriteFn = unsafe extern "C" fn(*mut c_void, usize, usize, u64, u32, u64, u64, u32, u64) -> CuteafdStatus;
        let write: Symbol<WriteFn> = unsafe { self.lib.get(b"cuteafd_rdma_rc_endpoint_post_write_flagged")? };
        let status = unsafe { write(handle, offset, bytes, remote, rkey, flag_value, flag_remote, flag_rkey, wr_id) };
        self.status_to_result("cuteafd_rdma_rc_endpoint_post_write_flagged", status)
    }

    pub unsafe fn rdma_rc_endpoint_register_region(
        &self,
        handle: *mut c_void,
        ptr: *mut c_void,
        bytes: usize,
    ) -> Result<u32> {
        type RegisterFn = unsafe extern "C" fn(*mut c_void, *mut c_void, usize, *mut u32) -> CuteafdStatus;
        let register: Symbol<RegisterFn> =
            unsafe { self.lib.get(b"cuteafd_rdma_rc_endpoint_register_region")? };
        let mut region = 0;
        let status = unsafe { register(handle, ptr, bytes, &mut region) };
        self.status_to_result("cuteafd_rdma_rc_endpoint_register_region", status)?;
        Ok(region)
    }

    /// Posts a signaled SEND gathering `slot_bytes` at `slot_offset` of the send
    /// ring and `region_bytes` at `region_offset` of a registered region.
    #[allow(clippy::too_many_arguments)]
    pub fn rdma_rc_endpoint_post_send_slot_region(
        &self,
        handle: *mut c_void,
        slot_offset: usize,
        slot_bytes: usize,
        region: u32,
        region_offset: usize,
        region_bytes: usize,
        wr_id: u64,
    ) -> Result<()> {
        type PostFn = unsafe extern "C" fn(*mut c_void, usize, usize, u32, usize, usize, u64) -> CuteafdStatus;
        let post: Symbol<PostFn> = unsafe { self.lib.get(b"cuteafd_rdma_rc_endpoint_post_send_slot_region")? };
        let status = unsafe { post(handle, slot_offset, slot_bytes, region, region_offset, region_bytes, wr_id) };
        self.status_to_result("cuteafd_rdma_rc_endpoint_post_send_slot_region", status)
    }

    pub fn rdma_rc_endpoint_destroy(&self, handle: *mut c_void) -> Result<()> {
        let destroy_fn: Symbol<RdmaRcEndpointDestroyFn> =
            unsafe { self.lib.get(b"cuteafd_rdma_rc_endpoint_destroy")? };
        memory_ledger::record_free(handle as usize);
        let status = unsafe { destroy_fn(handle) };
        self.status_to_result("cuteafd_rdma_rc_endpoint_destroy", status)
    }

    /// Metadata-only symbol check; terminal ownership is unavailable on older
    /// libraries and must be rejected before external landing allocation.
    pub fn rdma_rc_endpoint_quiesce_available(&self) -> Result<()> {
        // SAFETY: resolving a function does not invoke it or create CUDA state.
        let _: Symbol<RdmaRcEndpointDestroyFn> = unsafe { self.lib.get(b"cuteafd_rdma_rc_endpoint_quiesce")? };
        Ok(())
    }

    pub fn rdma_rc_endpoint_quiesce(&self, handle: *mut c_void) -> Result<()> {
        // SAFETY: the transport owns the live endpoint; this optional ABI only
        // destroys its QP and retains every registration and storage owner.
        let quiesce: Symbol<RdmaRcEndpointDestroyFn> = unsafe { self.lib.get(b"cuteafd_rdma_rc_endpoint_quiesce")? };
        let status = unsafe { quiesce(handle) };
        self.status_to_result("cuteafd_rdma_rc_endpoint_quiesce", status)
    }

    pub fn last_error(&self) -> Result<String> {
        let last_error_fn: Symbol<LastErrorFn> = unsafe { self.lib.get(b"cuteafd_last_error")? };
        let mut buf = vec![0 as c_char; 512];
        let status = unsafe { last_error_fn(buf.as_mut_ptr(), buf.len()) };
        if status != CUTEAFD_STATUS_OK {
            anyhow::bail!("cuteafd_last_error returned status {status}");
        }
        let cstr = unsafe { CStr::from_ptr(buf.as_ptr()) };
        Ok(cstr.to_string_lossy().into_owned())
    }

    fn status_to_result(&self, context: &str, status: CuteafdStatus) -> Result<()> {
        if status == CUTEAFD_STATUS_OK {
            return Ok(());
        }
        let last_error = self
            .last_error()
            .unwrap_or_else(|err| format!("last error unavailable: {err}"));
        anyhow::bail!("{context} returned status {status}: {last_error}");
    }
}

pub fn c_char_array_to_string(value: &[c_char]) -> String {
    let nul = value.iter().position(|ch| *ch == 0).unwrap_or(value.len());
    let bytes = value[..nul].iter().map(|ch| *ch as u8).collect::<Vec<_>>();
    String::from_utf8_lossy(&bytes).into_owned()
}

fn validate_positive_dim(context: &str, dim: i32) -> Result<()> {
    if dim <= 0 {
        anyhow::bail!("{context} must be positive, got {dim}");
    }
    Ok(())
}

fn validate_f32_rows(context: &str, rows: i32, hidden: i32) -> Result<()> {
    validate_positive_dim(&format!("{context} rows"), rows)?;
    validate_positive_dim(&format!("{context} hidden"), hidden)?;
    Ok(())
}

fn validate_f32_buffer_values(
    context: &str,
    buffer: CuteafdDeviceBuffer,
    value_count: usize,
) -> Result<()> {
    validate_device_buffer_bytes(context, buffer, checked_f32_bytes(context, value_count)?)
}

fn validate_u32_buffer_values(
    context: &str,
    buffer: CuteafdDeviceBuffer,
    value_count: usize,
) -> Result<()> {
    validate_device_buffer_bytes(
        context,
        buffer,
        value_count
            .checked_mul(std::mem::size_of::<u32>())
            .with_context(|| format!("{context} byte count overflows usize"))?,
    )
}


fn validate_u64_buffer_values(
    context: &str,
    buffer: CuteafdDeviceBuffer,
    value_count: usize,
) -> Result<()> {
    validate_device_buffer_bytes(
        context,
        buffer,
        value_count
            .checked_mul(std::mem::size_of::<u64>())
            .with_context(|| format!("{context} byte count overflows usize"))?,
    )
}

fn validate_u16_buffer_values(
    context: &str,
    buffer: CuteafdDeviceBuffer,
    value_count: usize,
) -> Result<()> {
    validate_device_buffer_bytes(
        context,
        buffer,
        value_count
            .checked_mul(std::mem::size_of::<u16>())
            .with_context(|| format!("{context} byte count overflows usize"))?,
    )
}

fn checked_row_values(context: &str, rows: usize, row_width: usize) -> Result<usize> {
    rows.checked_mul(row_width)
        .with_context(|| format!("{context} row value count overflows usize"))
}

fn checked_f32_bytes(context: &str, value_count: usize) -> Result<usize> {
    value_count
        .checked_mul(std::mem::size_of::<f32>())
        .with_context(|| format!("{context} byte count overflows usize"))
}



#[allow(clippy::too_many_arguments)]
fn validate_row_gather_f32_to_fp8_e4m3_row_scaled_buffers(
    context: &str,
    src: CuteafdDeviceBuffer,
    src_rows: usize,
    row_indices: CuteafdDeviceBuffer,
    dst: CuteafdDeviceBuffer,
    rows: usize,
    row_width: usize,
    dst_row_stride_bytes: usize,
) -> Result<()> {
    if rows == 0 || row_width == 0 || src_rows == 0 {
        anyhow::bail!("{context} rows, row_width, and src_rows must be positive");
    }
    let minimum_stride = row_width
        .checked_add(std::mem::size_of::<f32>())
        .with_context(|| format!("{context} minimum row stride overflows usize"))?;
    if dst_row_stride_bytes < minimum_stride
        || dst_row_stride_bytes % std::mem::align_of::<f32>() != 0
    {
        anyhow::bail!(
            "{context} destination row stride {dst_row_stride_bytes} must be FP32-aligned and at least {minimum_stride} bytes"
        );
    }
    validate_f32_buffer_values(
        &format!("{context} src"),
        src,
        checked_row_values(&format!("{context} src"), src_rows, row_width)?,
    )?;
    validate_u32_buffer_values(&format!("{context} row_indices"), row_indices, rows)?;
    validate_device_buffer_bytes(
        &format!("{context} dst"),
        dst,
        rows.checked_mul(dst_row_stride_bytes)
            .with_context(|| format!("{context} destination byte count overflows usize"))?,
    )
}




#[allow(clippy::too_many_arguments)]
fn validate_row_gather_f32_to_nvfp4_e2m1_fp8_e4m3_buffers(
    context: &str,
    src: CuteafdDeviceBuffer,
    src_rows: usize,
    row_indices: CuteafdDeviceBuffer,
    dst: CuteafdDeviceBuffer,
    rows: usize,
    row_width: usize,
    dst_row_stride_bytes: usize,
) -> Result<()> {
    if rows == 0 || src_rows == 0 {
        anyhow::bail!("{context} rows and src_rows must be positive");
    }
    let minimum_stride = checked_nvfp4_e2m1_fp8_e4m3_row_bytes(context, row_width)?;
    if dst_row_stride_bytes < minimum_stride {
        anyhow::bail!(
            "{context} destination row stride {dst_row_stride_bytes} must be at least {minimum_stride} bytes"
        );
    }
    validate_f32_buffer_values(
        &format!("{context} src"),
        src,
        checked_row_values(&format!("{context} src"), src_rows, row_width)?,
    )?;
    validate_u32_buffer_values(&format!("{context} row_indices"), row_indices, rows)?;
    validate_device_buffer_bytes(
        &format!("{context} dst"),
        dst,
        rows.checked_mul(dst_row_stride_bytes)
            .with_context(|| format!("{context} destination byte count overflows usize"))?,
    )
}


fn validate_f32_to_bf16_buffers(
    context: &str,
    src: CuteafdDeviceBuffer,
    dst: CuteafdDeviceBuffer,
    count: usize,
) -> Result<()> {
    if count == 0 {
        return Ok(());
    }
    validate_f32_buffer_values(&format!("{context} src"), src, count)?;
    validate_u16_buffer_values(&format!("{context} dst"), dst, count)
}






fn checked_nvfp4_e2m1_fp8_e4m3_row_bytes(context: &str, row_width: usize) -> Result<usize> {
    if row_width == 0 || row_width % 16 != 0 {
        anyhow::bail!("{context} row_width must be a positive multiple of 16, got {row_width}");
    }
    row_width
        .checked_div(2)
        .and_then(|packed| packed.checked_add(row_width / 16))
        .with_context(|| format!("{context} NVFP4 row byte count overflows usize"))
}

#[allow(clippy::too_many_arguments)]
fn validate_route_shard_reduction_buffers(
    context: &str,
    buffers: &CuteafdRouteShardReductionBuffers,
    rows: usize,
    row_width: usize,
    peer_row_stride_bytes: usize,
    local_dtype: u32,
    peer_dtype: u32,
    peer_count: usize,
) -> Result<()> {
    if rows == 0 || row_width == 0 {
        anyhow::bail!("{context} rows and row_width must be positive");
    }
    if !(1..=buffers.peers.len()).contains(&peer_count) {
        anyhow::bail!(
            "{context} peer count {peer_count} must be in 1..={}",
            buffers.peers.len()
        );
    }
    let values = checked_row_values(context, rows, row_width)?;
    match local_dtype {
        CUTEAFD_ROUTE_SHARD_LOCAL_F32 => {
            validate_f32_buffer_values(&format!("{context} local_f32"), buffers.local, values)?;
        }
        CUTEAFD_ROUTE_SHARD_LOCAL_BF16 => {
            validate_u16_buffer_values(&format!("{context} local_bf16"), buffers.local, values)?;
        }
        other => anyhow::bail!("{context} unsupported local dtype {other}"),
    }
    validate_f32_buffer_values(&format!("{context} output_f32"), buffers.output_f32, values)?;
    let minimum_peer_stride = match peer_dtype {
        CUTEAFD_ROUTE_SHARD_WIRE_BF16 => row_width
            .checked_mul(std::mem::size_of::<u16>())
            .with_context(|| format!("{context} BF16 row byte count overflows usize"))?,
        CUTEAFD_ROUTE_SHARD_WIRE_FP8_E4M3_ROW_SCALED => {
            if peer_row_stride_bytes % std::mem::align_of::<f32>() != 0 {
                anyhow::bail!("{context} FP8 peer row stride must be FP32-aligned");
            }
            row_width
                .checked_add(std::mem::size_of::<f32>())
                .with_context(|| format!("{context} FP8 row byte count overflows usize"))?
        }
        CUTEAFD_ROUTE_SHARD_WIRE_NVFP4_E2M1_FP8_E4M3 => {
            checked_nvfp4_e2m1_fp8_e4m3_row_bytes(context, row_width)?
        }
        other => anyhow::bail!("{context} unsupported peer dtype {other}"),
    };
    if peer_row_stride_bytes < minimum_peer_stride {
        anyhow::bail!(
            "{context} peer row stride {peer_row_stride_bytes} is below {minimum_peer_stride}"
        );
    }
    let peer_bytes = rows
        .checked_mul(peer_row_stride_bytes)
        .with_context(|| format!("{context} peer byte count overflows usize"))?;
    for (index, peer) in buffers.peers[..peer_count].iter().copied().enumerate() {
        validate_device_buffer_bytes(&format!("{context} peer {index}"), peer, peer_bytes)?;
    }
    Ok(())
}













































fn validate_embedding_lookup_bf16_buffers(
    context: &str,
    embedding: CuteafdDeviceBuffer,
    token_ids: CuteafdDeviceBuffer,
    out: CuteafdDeviceBuffer,
    rows: usize,
    vocab: usize,
    hidden: usize,
) -> Result<()> {
    if rows == 0 {
        anyhow::bail!("{context} rows must be positive");
    }
    if vocab == 0 {
        anyhow::bail!("{context} vocab must be positive");
    }
    if hidden == 0 {
        anyhow::bail!("{context} hidden must be positive");
    }
    let embedding_values = checked_row_values(&format!("{context} embedding"), vocab, hidden)?;
    let out_values = checked_row_values(&format!("{context} out"), rows, hidden)?;
    validate_u16_buffer_values(&format!("{context} embedding"), embedding, embedding_values)?;
    validate_u32_buffer_values(&format!("{context} token_ids"), token_ids, rows)?;
    validate_u16_buffer_values(&format!("{context} out"), out, out_values)
}






/// Validate the buffers and per-row parameter block for the v4.1 GPU
/// target-sampler (design §5.4) before a launch.
///
/// Mirrors `TargetSamplingParams::new` (`target_sampling.rs:108-134`) per row
/// and the buffer-extent checks of `validate_logits_argmax_buffers`. Device
/// pointers are passed as `CuteafdDeviceBuffer` values whose `ptr` field is the
/// raw address, so the same code validates real device allocations and the
/// synthetic addresses used by the FFI tests.
#[allow(clippy::too_many_arguments)]
/// The kernel dereferences the parameter block on device, so the launched
/// pointer must be a device allocation of one 64-byte block per row. The host
/// slice the wrapper also takes is the validator's input and is never launched.
fn validate_v41_params_device(context: &str, params_device: CuteafdDeviceBuffer,
    rows: usize,
) -> Result<()> {
    if params_device.ptr.is_null() {
        anyhow::bail!("{context} params must point at device memory");
    }
    let needed = rows
        .checked_mul(CUTEAFD_V41_SAMPLER_PARAM_BYTES)
        .context("v4.1 sampler parameter extent overflow")?;
    if params_device.bytes < needed {
        anyhow::bail!(
            "{context} params device buffer holds {} bytes, need {needed}",
            params_device.bytes
        );
    }
    Ok(())
}

/// Shared validation for the chunk-3b K5 entry points: the K1 buffer contract,
/// the optional rank-order arenas (read-only here), and the required
/// `rank_retained_count`/`out_indices`/`out_status` buffers. `out_total` and
/// `out_nucleus_count` are optional (design D3) and are only checked when
/// supplied.
///
/// K5 consumes K3/K4's `out_retained_count` and the rank-order arena **by block
/// row** while `out_*` are keyed by `output_row`; a non-identity `output_row`
/// mapping would pair a row with another row's retained count. The nucleus entry
/// point therefore requires `params[r].output_row == r` for every row. K1/K2/K3/
/// K4 keep supporting scattered `output_row`; only K5's consumption is affected.
#[allow(clippy::too_many_arguments)]
fn validate_v41_nucleus_buffers(
    context: &str,
    logits: CuteafdDeviceBuffer,
    rows: usize,
    vocab: usize,
    logits_stride: usize,
    params: &[CuteafdV41SamplerRow],
    mask_words: Option<CuteafdDeviceBuffer>,
    mask_words_per_row: usize,
    rank_order_ids: Option<CuteafdDeviceBuffer>,
    rank_order_capacity: usize,
    rank_retained_count: CuteafdDeviceBuffer,
    out_indices: CuteafdDeviceBuffer,
    out_status: CuteafdDeviceBuffer,
    out_total: Option<CuteafdDeviceBuffer>,
    out_nucleus_count: Option<CuteafdDeviceBuffer>,
    scratch: CuteafdDeviceBuffer,
) -> Result<()> {
    validate_v41_sampling_buffers(
        context,
        logits,
        rows,
        vocab,
        logits_stride,
        params,
        mask_words,
        mask_words_per_row,
        scratch,
        scratch,
        scratch,
        scratch,
        None,
        None,
        scratch,
    )?;
    for (row, params) in params.iter().enumerate() {
        if params.output_row as usize != row {
            anyhow::bail!(
                "{context} row {row} has output_row {}, but the nucleus entry point requires \
                 output_row == block row: K5 consumes K3/K4's rank_retained_count and rank-order \
                 arena by block row while out_* are written by output_row",
                params.output_row
            );
        }
    }
    validate_u32_buffer_values(&format!("{context} rank_retained_count"), rank_retained_count, rows)?;
    validate_u32_buffer_values(&format!("{context} out_indices"), out_indices, rows)?;
    validate_u32_buffer_values(&format!("{context} out_status"), out_status, rows)?;
    if let Some(total) = out_total {
        validate_f32_buffer_values(&format!("{context} out_total"), total, rows)?;
    }
    if let Some(nucleus) = out_nucleus_count {
        validate_u32_buffer_values(&format!("{context} out_nucleus_count"), nucleus, rows)?;
    }
    match rank_order_ids {
        None => {
            if rank_order_capacity != 0 {
                anyhow::bail!(
                    "{context} rank_order_capacity is {rank_order_capacity} but no rank-order \
                     arena was supplied"
                );
            }
        }
        Some(ids) => {
            if rank_order_capacity == 0 {
                anyhow::bail!(
                    "{context} a rank-order arena was supplied with a zero per-row capacity"
                );
            }
            let values =
                checked_row_values(&format!("{context} rank_order_ids"), rows, rank_order_capacity)?;
            validate_u32_buffer_values(&format!("{context} rank_order_ids"), ids, values)?;
            let max_top_k = params.iter().map(|params| params.top_k).max().unwrap_or(0) as usize;
            if rank_order_capacity < max_top_k {
                anyhow::bail!(
                    "{context} rank_order_capacity {rank_order_capacity} is below the batch's \
                     maximum top_k {max_top_k}"
                );
            }
        }
    }
    Ok(())
}

fn validate_v41_sampling_buffers(
    context: &str,
    logits: CuteafdDeviceBuffer,
    rows: usize,
    vocab: usize,
    logits_stride: usize,
    params: &[CuteafdV41SamplerRow],
    mask_words: Option<CuteafdDeviceBuffer>,
    mask_words_per_row: usize,
    out_indices: CuteafdDeviceBuffer,
    out_status: CuteafdDeviceBuffer,
    out_status_detail: CuteafdDeviceBuffer,
    out_scores: CuteafdDeviceBuffer,
    out_total: Option<CuteafdDeviceBuffer>,
    out_nucleus_count: Option<CuteafdDeviceBuffer>,
    scratch: CuteafdDeviceBuffer,
) -> Result<()> {
    if rows == 0 {
        anyhow::bail!("{context} rows must be positive");
    }
    if vocab == 0 {
        anyhow::bail!("{context} vocab must be positive");
    }
    if vocab > u32::MAX as usize {
        anyhow::bail!("{context} vocab must fit in u32 output indices");
    }
    if logits_stride < vocab {
        anyhow::bail!(
            "{context} logits_stride {logits_stride} is smaller than vocab {vocab}"
        );
    }
    if params.len() != rows {
        anyhow::bail!(
            "{context} params must hold one block per row: have {}, need {rows}",
            params.len()
        );
    }
    let expected_words = vocab.div_ceil(32);
    match (mask_words, mask_words_per_row) {
        (Some(_), 0) => anyhow::bail!("{context} mask_words require a positive mask_words_per_row"),
        (None, 0) => {}
        (None, provided) => anyhow::bail!(
            "{context} mask_words_per_row must be 0 when no mask buffer is supplied, got {provided}"
        ),
        (Some(_), provided) if provided != expected_words => anyhow::bail!(
            "{context} mask_words_per_row must equal ceil(vocab/32) = {expected_words}, got {provided}"
        ),
        (Some(_), _) => {}
    }
    for (row, params) in params.iter().enumerate() {
        if !params.temperature.is_finite() || !(0.0..=2.0).contains(&params.temperature) {
            anyhow::bail!(
                "{context} row {row} temperature must be finite and in [0, 2], got {}",
                params.temperature
            );
        }
        if !params.top_p.is_finite() || params.top_p <= 0.0 || params.top_p > 1.0 {
            anyhow::bail!(
                "{context} row {row} top_p must be finite and in (0, 1], got {}",
                params.top_p
            );
        }
        if !params.min_p.is_finite() || !(0.0..=1.0).contains(&params.min_p) {
            anyhow::bail!(
                "{context} row {row} min_p must be finite and in [0, 1], got {}",
                params.min_p
            );
        }
        // The device never calls `logf`; `ln_min_p` is the host-precomputed
        // threshold. `min_p == 0` must ship `-inf`, otherwise a row silently
        // gets a finite min_p threshold the caller did not ask for.
        if params.min_p == 0.0 {
            if !params.ln_min_p.is_infinite() || params.ln_min_p > 0.0 {
                anyhow::bail!(
                    "{context} row {row} min_p is 0, so ln_min_p must be -inf, got {}",
                    params.ln_min_p
                );
            }
        } else {
            let expected = params.min_p.ln();
            if !params.ln_min_p.is_finite() || params.ln_min_p != expected {
                anyhow::bail!(
                    "{context} row {row} ln_min_p must be f32::ln(min_p) = {expected}, got {}",
                    params.ln_min_p
                );
            }
        }
        if params.reserved0 != 0 || params.reserved1 != 0 || params.reserved2 != 0 {
            anyhow::bail!("{context} row {row} reserved fields must be zero");
        }
        if (params.flags & !CUTEAFD_V41_SAMPLER_FLAG_KNOWN_MASK) != 0 {
            anyhow::bail!(
                "{context} row {row} flags contain unknown bits: {:#x}",
                params.flags
            );
        }
        // `STRICT_FINITE` opts a row into the whole-row finiteness check the
        // greedy and constrained paths always perform. It is meaningless for a
        // plainly stochastic, unconstrained row, and the host does not set it
        // there: requiring greedy or a real mask keeps the header, the host, the
        // kernel and this validator telling the same story.
        if (params.flags & CUTEAFD_V41_SAMPLER_FLAG_STRICT_FINITE) != 0 {
            let greedy = params.temperature < 1e-5
                || params.top_k == 1
                || (params.flags & CUTEAFD_V41_SAMPLER_FLAG_GREEDY) != 0;
            let masked = (params.flags & CUTEAFD_V41_SAMPLER_FLAG_NO_MASK) == 0;
            if !greedy && !masked {
                anyhow::bail!(
                    "{context} row {row} sets STRICT_FINITE but is neither greedy nor masked"
                );
            }
        }
        if params.output_row as usize >= rows {
            anyhow::bail!(
                "{context} row {row} output_row {} is outside the {rows}-row output",
                params.output_row
            );
        }
        if (params.flags & CUTEAFD_V41_SAMPLER_FLAG_NO_MASK) != 0 {
            // A row marked unconstrained must say so in `mask_row`: a caller
            // that set a real index would believe the row is constrained while
            // the kernel ignores the mask.
            if params.mask_row != CUTEAFD_V41_SAMPLER_NO_MASK_ROW {
                anyhow::bail!(
                    "{context} row {row} is unconstrained, so mask_row must be {:#x}, got {:#x}",
                    CUTEAFD_V41_SAMPLER_NO_MASK_ROW,
                    params.mask_row
                );
            }
        } else {
            if mask_words.is_none() {
                anyhow::bail!(
                    "{context} row {row} is masked but no mask buffer was supplied"
                );
            }
            let arena_row = params.mask_row as usize;
            if arena_row >= rows {
                anyhow::bail!(
                    "{context} row {row} mask_row {arena_row} is outside the {rows}-row mask arena"
                );
            }
        }
    }
    let logits_values = checked_row_values(&format!("{context} logits"), rows, logits_stride)?;
    validate_f32_buffer_values(&format!("{context} logits"), logits, logits_values)?;
    if let Some(mask_words) = mask_words {
        let values = checked_row_values(
            &format!("{context} mask_words"),
            rows,
            mask_words_per_row,
        )?;
        validate_u32_buffer_values(&format!("{context} mask_words"), mask_words, values)?;
    }
    validate_u32_buffer_values(&format!("{context} out_indices"), out_indices, rows)?;
    validate_u32_buffer_values(&format!("{context} out_status"), out_status, rows)?;
    validate_u32_buffer_values(&format!("{context} out_status_detail"), out_status_detail, rows)?;
    validate_f32_buffer_values(&format!("{context} out_scores"), out_scores, rows)?;
    if let Some(out_total) = out_total {
        validate_f32_buffer_values(&format!("{context} out_total"), out_total, rows)?;
    }
    if let Some(out_nucleus_count) = out_nucleus_count {
        validate_u32_buffer_values(
            &format!("{context} out_nucleus_count"),
            out_nucleus_count,
            rows,
        )?;
    }
    let scratch_bytes = rows
        .checked_mul(CUTEAFD_V41_SAMPLER_SCRATCH_BYTES)
        .with_context(|| format!("{context} scratch byte count overflows usize"))?;
    validate_device_buffer_bytes(&format!("{context} scratch"), scratch, scratch_bytes)
}

/// Host-side validation for the chunk-3a K3/K4 entry points.
///
/// Every field the chunk-1 validator already pins (logits extent, parameter
/// blocks, mask width and mask semantics, status codes, `output_row` bounds,
/// `ln_min_p` consistency) is reused verbatim: the K1/K2 output buffers do not
/// exist on this entry point, so `scratch` stands in for them. The validator
/// only checks pointer/extent for those, and the scratch extent (`rows * 64`)
/// dominates each of them, so the reuse cannot weaken a check.
///
/// The rank-order arena is all-or-nothing: either both arenas are supplied with
/// a positive per-row capacity that covers every row's `top_k`, or both are
/// absent and `capacity == 0` (selection-only).
fn validate_v41_topk_select_buffers(
    context: &str,
    logits: CuteafdDeviceBuffer,
    rows: usize,
    vocab: usize,
    logits_stride: usize,
    params: &[CuteafdV41SamplerRow],
    mask_words: Option<CuteafdDeviceBuffer>,
    mask_words_per_row: usize,
    rank_order_ids: Option<CuteafdDeviceBuffer>,
    rank_order_scratch: Option<CuteafdDeviceBuffer>,
    rank_order_capacity: usize,
    out_retained_count: CuteafdDeviceBuffer,
    out_pivot_passes: CuteafdDeviceBuffer,
    scratch: CuteafdDeviceBuffer,
) -> Result<()> {
    validate_v41_sampling_buffers(
        context,
        logits,
        rows,
        vocab,
        logits_stride,
        params,
        mask_words,
        mask_words_per_row,
        scratch,
        scratch,
        scratch,
        scratch,
        None,
        None,
        scratch,
    )?;
    validate_u32_buffer_values(
        &format!("{context} out_retained_count"),
        out_retained_count,
        rows,
    )?;
    validate_u32_buffer_values(&format!("{context} out_pivot_passes"), out_pivot_passes, rows)?;
    match (rank_order_ids, rank_order_scratch) {
        (None, None) => {
            if rank_order_capacity != 0 {
                anyhow::bail!(
                    "{context} rank_order_capacity is {rank_order_capacity} but no rank-order \
                     arenas were supplied"
                );
            }
        }
        (Some(ids), Some(rank_scratch)) => {
            if rank_order_capacity == 0 {
                anyhow::bail!(
                    "{context} rank-order arenas were supplied with a zero per-row capacity"
                );
            }
            let values =
                checked_row_values(&format!("{context} rank_order_ids"), rows, rank_order_capacity)?;
            validate_u32_buffer_values(&format!("{context} rank_order_ids"), ids, values)?;
            validate_u64_buffer_values(
                &format!("{context} rank_order_scratch"),
                rank_scratch,
                values,
            )?;
            let max_top_k = params.iter().map(|params| params.top_k).max().unwrap_or(0) as usize;
            if rank_order_capacity < max_top_k {
                anyhow::bail!(
                    "{context} rank_order_capacity {rank_order_capacity} is below the batch's \
                     maximum top_k {max_top_k}"
                );
            }
        }
        _ => anyhow::bail!(
            "{context} the rank-order id and scratch arenas must be supplied together"
        ),
    }
    Ok(())
}

fn validate_logits_argmax_buffers(
    context: &str,
    logits: CuteafdDeviceBuffer,
    out_indices: CuteafdDeviceBuffer,
    out_scores: CuteafdDeviceBuffer,
    rows: usize,
    vocab: usize,
) -> Result<()> {
    if rows == 0 {
        anyhow::bail!("{context} rows must be positive");
    }
    if vocab == 0 {
        anyhow::bail!("{context} vocab must be positive");
    }
    if vocab > u32::MAX as usize {
        anyhow::bail!("{context} vocab must fit in u32 output indices");
    }
    let logits_values = checked_row_values(&format!("{context} logits"), rows, vocab)?;
    validate_f32_buffer_values(&format!("{context} logits"), logits, logits_values)?;
    validate_u32_buffer_values(&format!("{context} out_indices"), out_indices, rows)?;
    validate_f32_buffer_values(&format!("{context} out_scores"), out_scores, rows)
}



fn validate_device_buffer_bytes(
    context: &str,
    buffer: CuteafdDeviceBuffer,
    required_bytes: usize,
) -> Result<()> {
    if buffer.ptr.is_null() {
        anyhow::bail!("{context} buffer pointer is null");
    }
    if buffer.bytes < required_bytes {
        anyhow::bail!(
            "{context} buffer is too small: has {} bytes, needs {required_bytes}",
            buffer.bytes
        );
    }
    Ok(())
}



#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use std::path::PathBuf;
    use std::slice;

    fn native_library_path() -> Option<PathBuf> {
        if let Ok(path) = env::var("CUTEAFD_NATIVE_LIB") {
            return Some(PathBuf::from(path));
        }
        if env::var_os("CUTEAFD_DISABLE_NATIVE_AUTO_DISCOVERY").is_some() {
            return None;
        }
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../..")
            .join("native/build/libcuteafd_native.so");
        path.exists().then_some(path)
    }

    fn load_test_library() -> Result<Option<NativeLibrary>> {
        let Some(path) = native_library_path() else {
            eprintln!("skipping native FFI test because native/build/libcuteafd_native.so is absent");
            return Ok(None);
        };
        let library = unsafe { NativeLibrary::load(path)? };
        Ok(Some(library))
    }

    /// Load the native library for a test that must actually run device code.
    ///
    /// Unlike [`load_test_library`] this never skips: a test whose entire point
    /// is a device/CPU comparison must fail when no library is present, not pass
    /// without running. `CUTEAFD_NATIVE_LIB` names a local build (how this work
    /// was verified); without it the default `native/build/` path must exist.
    fn load_device_test_library() -> Result<NativeLibrary> {
        let path = match std::env::var_os("CUTEAFD_NATIVE_LIB") {
            Some(explicit) => {
                let path = std::path::PathBuf::from(explicit);
                anyhow::ensure!(
                    path.is_file(),
                    "CUTEAFD_NATIVE_LIB points at {} which does not exist",
                    path.display()
                );
                path
            }
            None => {
                let path = native_library_path().context(
                    "the v4.1 sampler device test needs a native library; build native/ or set \
                     CUTEAFD_NATIVE_LIB to a libcuteafd_native.so",
                )?;
                anyhow::ensure!(
                    path.is_file(),
                    "the v4.1 sampler device test needs {} to exist",
                    path.display()
                );
                path
            }
        };
        let library = unsafe { NativeLibrary::load(path)? };
        Ok(library)
    }

    fn synthetic_device_buffer(address: usize, bytes: usize) -> CuteafdDeviceBuffer {
        CuteafdDeviceBuffer {
            ptr: address as *mut c_void,
            bytes,
            device_id: 0,
            flags: 0,
        }
    }

    /* ---- v4.1 GPU target-sampler: ABI pin + validator ---- */

    /// The device ABI is a 64-byte struct with natural alignment. These offsets
    /// are the design's §5.1 table; a field reorder or a size change must fail
    /// here before it can silently corrupt a launch.
    #[test]
    fn v41_sampler_row_abi_layout_is_pinned() {
        use std::mem::{align_of, offset_of, size_of};
        assert_eq!(size_of::<CuteafdV41SamplerRow>(), 64);
        assert_eq!(CUTEAFD_V41_SAMPLER_PARAM_BYTES, 64);
        assert_eq!(CUTEAFD_V41_SAMPLER_SCRATCH_BYTES, 64);
        assert_eq!(align_of::<CuteafdV41SamplerRow>(), 8);
        assert_eq!(offset_of!(CuteafdV41SamplerRow, seed), 0);
        assert_eq!(offset_of!(CuteafdV41SamplerRow, position), 8);
        assert_eq!(offset_of!(CuteafdV41SamplerRow, temperature), 16);
        assert_eq!(offset_of!(CuteafdV41SamplerRow, top_p), 20);
        assert_eq!(offset_of!(CuteafdV41SamplerRow, min_p), 24);
        assert_eq!(offset_of!(CuteafdV41SamplerRow, top_k), 28);
        assert_eq!(offset_of!(CuteafdV41SamplerRow, mask_row), 32);
        assert_eq!(offset_of!(CuteafdV41SamplerRow, flags), 36);
        assert_eq!(offset_of!(CuteafdV41SamplerRow, output_row), 40);
        assert_eq!(offset_of!(CuteafdV41SamplerRow, ln_min_p), 44);
        assert_eq!(offset_of!(CuteafdV41SamplerRow, reserved0), 48);
        assert_eq!(offset_of!(CuteafdV41SamplerRow, reserved1), 52);
        assert_eq!(offset_of!(CuteafdV41SamplerRow, reserved2), 56);
        // The documented flag bits and status codes are part of the ABI too.
        assert_eq!(CUTEAFD_V41_SAMPLER_FLAG_GREEDY, 0x1);
        assert_eq!(CUTEAFD_V41_SAMPLER_FLAG_DIAGNOSE, 0x2);
        assert_eq!(CUTEAFD_V41_SAMPLER_FLAG_NO_MASK, 0x4);
        assert_eq!(CUTEAFD_V41_SAMPLER_FLAG_ORACLE_CROSSCHECK, 0x8);
        assert_eq!(CUTEAFD_V41_SAMPLER_FLAG_STRICT_FINITE, 0x10);
        assert_eq!(CUTEAFD_V41_SAMPLER_STATUS_OK, 0);
        assert_eq!(CUTEAFD_V41_SAMPLER_STATUS_EMPTY_CANDIDATES, 1);
        assert_eq!(CUTEAFD_V41_SAMPLER_STATUS_NONFINITE_LOGIT, 2);
        assert_eq!(CUTEAFD_V41_SAMPLER_STATUS_INVALID_TEMPERATURE, 3);
        assert_eq!(CUTEAFD_V41_SAMPLER_STATUS_MASK_WIDTH, 4);
        assert_eq!(CUTEAFD_V41_SAMPLER_STATUS_INTERNAL, 5);
        assert_eq!(CUTEAFD_V41_SAMPLER_NO_DETAIL, u32::MAX);
        assert_eq!(CUTEAFD_V41_SAMPLER_NO_MASK_ROW, u32::MAX);
        assert_eq!(CUTEAFD_V41_SAMPLER_FLAG_KNOWN_MASK, 0x1F);
        // A default block is a valid unconstrained greedy row.
        let default = CuteafdV41SamplerRow::default();
        assert_eq!(default.flags, 0);
        assert_eq!(default.mask_row, CUTEAFD_V41_SAMPLER_NO_MASK_ROW);
        assert!(default.ln_min_p.is_infinite() && default.ln_min_p < 0.0);
    }

    /// `ceil(vocab/32)` and the §5.3 remainder rule, pinned on the vocabularies
    /// the design names plus the official checkpoint.
    #[test]
    fn v41_sampler_mask_width_and_remainder_rule() {
        assert_eq!(cuteafd_sampler_mask_words(1), 1);
        assert_eq!(cuteafd_sampler_mask_words(32), 1);
        assert_eq!(cuteafd_sampler_mask_words(33), 2);
        assert_eq!(cuteafd_sampler_mask_words(100), 4);
        assert_eq!(cuteafd_sampler_mask_words(127), 4);
        assert_eq!(cuteafd_sampler_mask_words(129_280), 4_040);
        assert_eq!(cuteafd_sampler_mask_words(129_281), 4_041);

        // A checkpoint-width vocabulary has no remainder: leaving the final word
        // alone is the whole rule.
        let mut aligned = vec![u32::MAX; 1];
        cuteafd_sampler_clear_remainder(&mut aligned, 32);
        assert_eq!(aligned, vec![u32::MAX]);

        // vocab 33: bit 0 of the final word is the first out-of-range bit.
        let mut words = vec![u32::MAX; 2];
        cuteafd_sampler_clear_remainder(&mut words, 33);
        assert_eq!(words, vec![u32::MAX, 1]);

        // vocab 127: 31 real bits in word 3, so only bit 31 is cleared.
        let mut words = vec![u32::MAX; 4];
        cuteafd_sampler_clear_remainder(&mut words, 127);
        assert_eq!(words, vec![u32::MAX, u32::MAX, u32::MAX, 0x7FFF_FFFF]);

        // vocab 129281: 1 real bit in the fifth word.
        let mut words = vec![u32::MAX; 4041];
        cuteafd_sampler_clear_remainder(&mut words, 129_281);
        assert_eq!(words[4040], 1);
        assert!(words[..4040].iter().all(|word| *word == u32::MAX));
    }

    fn v41_params(rows: usize, flags: u32, mask_row: u32) -> Vec<CuteafdV41SamplerRow> {
        (0..rows)
            .map(|row| CuteafdV41SamplerRow {
                temperature: 0.0,
                top_p: 1.0,
                min_p: 0.0,
                top_k: 0,
                mask_row,
                flags,
                output_row: row as u32,
                ..CuteafdV41SamplerRow::default()
            })
            .collect()
    }

    /// Shorthand for the validator under test.
    #[allow(clippy::too_many_arguments)]
    fn validate_v41(
        rows: usize,
        vocab: usize,
        logits_stride: usize,
        params: &[CuteafdV41SamplerRow],
        with_mask: bool,
        mask_words_per_row: usize,
    ) -> Result<()> {
        let words = if with_mask {
            vocab.div_ceil(32)
        } else {
            0
        };
        let row_bytes = rows.max(1) * 4096;
        validate_v41_sampling_buffers(
            "test v4.1 sampler",
            synthetic_device_buffer(0x10_0000, rows.max(1) * logits_stride.max(vocab) * 4),
            rows,
            vocab,
            logits_stride,
            params,
            with_mask.then(|| synthetic_device_buffer(0x20_0000, rows.max(1) * words * 4)),
            mask_words_per_row,
            synthetic_device_buffer(0x30_0000, rows.max(1) * 4),
            synthetic_device_buffer(0x31_0000, rows.max(1) * 4),
            synthetic_device_buffer(0x32_0000, rows.max(1) * 4),
            synthetic_device_buffer(0x33_0000, rows.max(1) * 4),
            Some(synthetic_device_buffer(0x34_0000, rows.max(1) * 4)),
            Some(synthetic_device_buffer(0x35_0000, rows.max(1) * 4)),
            synthetic_device_buffer(0x36_0000, rows.max(1) * 64 + row_bytes * 0),
        )
    }

    #[test]
    fn v41_sampler_validator_accepts_a_well_formed_unconstrained_batch() {
        let params = v41_params(
            3,
            CUTEAFD_V41_SAMPLER_FLAG_GREEDY
                | CUTEAFD_V41_SAMPLER_FLAG_NO_MASK
                | CUTEAFD_V41_SAMPLER_FLAG_STRICT_FINITE,
            CUTEAFD_V41_SAMPLER_NO_MASK_ROW,
        );
        validate_v41(3, 129_280, 129_280, &params, false, 0).expect("valid unconstrained batch");
    }

    #[test]
    fn v41_nucleus_validator_requires_identity_output_row() {
        // K5 consumes K3/K4's `rank_retained_count` and the rank-order arena by
        // BLOCK row while writing `out_*` by `output_row`. A swapped pair (one
        // row truncating under `top_k`) would pair a row with another row's
        // retained count, so the nucleus validator must reject it and the
        // identity mapping must pass.
        let rows = 2usize;
        let vocab = 100usize;
        let capacity = 4usize;
        let make_params = |output_rows: [u32; 2]| -> Vec<CuteafdV41SamplerRow> {
            output_rows
                .iter()
                .map(|output_row| CuteafdV41SamplerRow {
                    temperature: 1.0,
                    top_p: 0.9,
                    min_p: 0.0,
                    top_k: capacity as u32,
                    mask_row: CUTEAFD_V41_SAMPLER_NO_MASK_ROW,
                    flags: CUTEAFD_V41_SAMPLER_FLAG_NO_MASK,
                    output_row: *output_row,
                    ln_min_p: f32::NEG_INFINITY,
                    ..CuteafdV41SamplerRow::default()
                })
                .collect()
        };
        let run_validator = |params: &[CuteafdV41SamplerRow]| {
            validate_v41_nucleus_buffers(
                "test K5",
                synthetic_device_buffer(0x10_0000, rows * vocab * 4),
                rows,
                vocab,
                vocab,
                params,
                None,
                0,
                Some(synthetic_device_buffer(0x20_0000, rows * capacity * 4)),
                capacity,
                synthetic_device_buffer(0x30_0000, rows * 4),
                synthetic_device_buffer(0x40_0000, rows * 4),
                synthetic_device_buffer(0x50_0000, rows * 4),
                None,
                None,
                synthetic_device_buffer(0x60_0000, rows * 64),
            )
        };
        let permuted = make_params([1, 0]);
        let error = run_validator(&permuted).expect_err("a permuted output_row must be rejected");
        assert!(
            error.to_string().contains("output_row == block row"),
            "unexpected error: {error}"
        );
        let identity = make_params([0, 1]);
        run_validator(&identity).expect("the identity output_row mapping validates");
    }

    #[test]
    fn v41_sampler_validator_rejects_malformed_batches() {
        let vocab = 100_usize;
        let words = 4_usize;
        let unconstrained = v41_params(
            2,
            CUTEAFD_V41_SAMPLER_FLAG_NO_MASK,
            CUTEAFD_V41_SAMPLER_NO_MASK_ROW,
        );

        // rows / vocab / stride extents.
        assert!(validate_v41(0, vocab, vocab, &[], false, 0).is_err());
        assert!(validate_v41(2, 0, 0, &unconstrained, false, 0).is_err());
        assert!(validate_v41(2, vocab, vocab - 1, &unconstrained, false, 0).is_err());
        // One parameter block per row.
        assert!(validate_v41(2, vocab, vocab, &unconstrained[..1], false, 0).is_err());
        assert!(validate_v41(1, vocab, vocab, &unconstrained, false, 0).is_err());

        // Mask presence and width.
        assert!(validate_v41(2, vocab, vocab, &unconstrained, false, words).is_err());
        // The exact shape a sampled launch produced when it passed the wave's own
        // (always non-zero) mask width with no staged arena: the shipped
        // vocabulary's 4040 words and no mask buffer. This is the pair the
        // launch must keep at (None, 0) for an all-unconstrained round.
        let shipped_vocab = 129_280_usize;
        let shipped_words = shipped_vocab.div_ceil(32);
        assert_eq!(shipped_words, 4040);
        let shipped_unconstrained =
            v41_params(2, CUTEAFD_V41_SAMPLER_FLAG_NO_MASK, CUTEAFD_V41_SAMPLER_NO_MASK_ROW);
        assert!(
            validate_v41(2, shipped_vocab, shipped_vocab, &shipped_unconstrained, false,
                shipped_words).is_err(),
            "a non-zero mask width with no mask buffer must be rejected"
        );
        // And the consistent pair is accepted.
        assert!(validate_v41(2, shipped_vocab, shipped_vocab, &shipped_unconstrained, false, 0)
            .is_ok());
        let masked = v41_params(2, CUTEAFD_V41_SAMPLER_FLAG_STRICT_FINITE, 0);
        assert!(validate_v41(2, vocab, vocab, &masked, true, words).is_ok());
        assert!(validate_v41(2, vocab, vocab, &masked, true, words - 1).is_err());
        assert!(validate_v41(2, vocab, vocab, &masked, true, words + 1).is_err());
        assert!(validate_v41(2, vocab, vocab, &masked, true, 0).is_err());
        assert!(validate_v41(2, vocab, vocab, &masked, false, 0).is_err());
        // mask_row must index the arena when the row is masked.
        let out_of_arena = v41_params(2, CUTEAFD_V41_SAMPLER_FLAG_STRICT_FINITE, 2);
        assert!(validate_v41(2, vocab, vocab, &out_of_arena, true, words).is_err());
        // A NO_MASK row must not claim a real mask row (a caller that did could
        // believe it is constrained when the device ignores the mask).
        let conflicting = v41_params(
            2,
            CUTEAFD_V41_SAMPLER_FLAG_NO_MASK,
            CUTEAFD_V41_SAMPLER_NO_MASK_ROW,
        );
        let mut conflicting = conflicting;
        conflicting[1].mask_row = 0;
        assert!(validate_v41(2, vocab, vocab, &conflicting, false, 0).is_err());
        // STRICT_FINITE is only meaningful for a greedy or constrained row. A
        // stochastic, unconstrained row must not claim it, or the header, the
        // host, the kernel and this validator would disagree about the mode.
        let stochastic_unconstrained = v41_params(2, 0, 0);
        let mut strict_stochastic = stochastic_unconstrained;
        strict_stochastic[1].flags |= CUTEAFD_V41_SAMPLER_FLAG_STRICT_FINITE;
        strict_stochastic[1].temperature = 0.8;
        strict_stochastic[1].mask_row = CUTEAFD_V41_SAMPLER_NO_MASK_ROW;
        strict_stochastic[1].flags |= CUTEAFD_V41_SAMPLER_FLAG_NO_MASK;
        assert!(validate_v41(2, vocab, vocab, &strict_stochastic, true, words).is_err());
        // The same row is accepted once it is greedy, which is where the host
        // sets the bit.
        let mut greedy_strict = strict_stochastic.clone();
        greedy_strict[1].flags |= CUTEAFD_V41_SAMPLER_FLAG_GREEDY;
        assert!(validate_v41(2, vocab, vocab, &greedy_strict, true, words).is_ok());
        // Or once it is constrained (a real mask row, no NO_MASK).
        let mut masked_strict = strict_stochastic;
        masked_strict[1].flags &= !CUTEAFD_V41_SAMPLER_FLAG_NO_MASK;
        masked_strict[1].mask_row = 1;
        assert!(validate_v41(2, vocab, vocab, &masked_strict, true, words).is_ok());

        // Parameter range checks per row.
        let mut params = unconstrained.clone();
        params[1].temperature = 2.5;
        assert!(validate_v41(2, vocab, vocab, &params, false, 0).is_err());
        let mut params = unconstrained.clone();
        params[1].temperature = f32::NAN;
        assert!(validate_v41(2, vocab, vocab, &params, false, 0).is_err());
        let mut params = unconstrained.clone();
        params[1].top_p = 0.0;
        assert!(validate_v41(2, vocab, vocab, &params, false, 0).is_err());
        let mut params = unconstrained.clone();
        params[1].top_p = 1.5;
        assert!(validate_v41(2, vocab, vocab, &params, false, 0).is_err());
        let mut params = unconstrained.clone();
        params[1].min_p = 1.5;
        assert!(validate_v41(2, vocab, vocab, &params, false, 0).is_err());
        let mut params = unconstrained.clone();
        params[1].min_p = f32::NAN;
        assert!(validate_v41(2, vocab, vocab, &params, false, 0).is_err());

        // ln_min_p is the device's only min_p threshold: it must be the host
        // f32::ln for an enabled min_p and -inf when min_p is disabled.
        let mut params = unconstrained.clone();
        params[0].min_p = 0.05;
        params[0].ln_min_p = f32::NEG_INFINITY;
        assert!(validate_v41(2, vocab, vocab, &params, false, 0).is_err());
        let mut params = unconstrained.clone();
        params[0].min_p = 0.05;
        params[0].ln_min_p = 0.05_f32.ln();
        assert!(validate_v41(2, vocab, vocab, &params, false, 0).is_ok());
        let mut params = unconstrained.clone();
        params[0].min_p = 0.05;
        params[0].ln_min_p = f32::from_bits(0.05_f32.ln().to_bits() + 1);
        assert!(validate_v41(2, vocab, vocab, &params, false, 0).is_err());
        let mut params = unconstrained.clone();
        params[0].min_p = 0.0;
        params[0].ln_min_p = 0.0;
        assert!(validate_v41(2, vocab, vocab, &params, false, 0).is_err());

        // Reserved fields, unknown flags and output_row bounds.
        let mut params = unconstrained.clone();
        params[0].reserved0 = 1;
        assert!(validate_v41(2, vocab, vocab, &params, false, 0).is_err());
        let mut params = unconstrained.clone();
        params[0].reserved2 = 1;
        assert!(validate_v41(2, vocab, vocab, &params, false, 0).is_err());
        let mut params = unconstrained.clone();
        params[0].flags |= 0x8000_0000;
        assert!(validate_v41(2, vocab, vocab, &params, false, 0).is_err());
        let mut params = unconstrained.clone();
        params[1].output_row = 2;
        assert!(validate_v41(2, vocab, vocab, &params, false, 0).is_err());
    }

    #[test]
    fn v41_sampler_validator_rejects_short_buffers() {
        let vocab = 33_usize;
        let words = 2_usize;
        let params = v41_params(
            2,
            CUTEAFD_V41_SAMPLER_FLAG_NO_MASK,
            CUTEAFD_V41_SAMPLER_NO_MASK_ROW,
        );
        let logits = synthetic_device_buffer(0x10_0000, 2 * vocab * 4);
        let mask = synthetic_device_buffer(0x20_0000, 2 * words * 4);
        let out = synthetic_device_buffer(0x30_0000, 2 * 4);
        let scratch = synthetic_device_buffer(0x40_0000, 2 * CUTEAFD_V41_SAMPLER_SCRATCH_BYTES);
        let run = |logits, mask, out, scratch| {
            validate_v41_sampling_buffers(
                "test v4.1 sampler",
                logits,
                2,
                vocab,
                vocab,
                &params,
                Some(mask),
                words,
                out,
                out,
                out,
                out,
                Some(out),
                Some(out),
                scratch,
            )
        };
        run(logits, mask, out, scratch).expect("exact extents are valid");
        assert!(run(synthetic_device_buffer(0x10_0000, 2 * vocab * 4 - 4), mask, out, scratch).is_err());
        assert!(run(logits, synthetic_device_buffer(0x20_0000, 2 * words * 4 - 4), out, scratch).is_err());
        assert!(run(logits, mask, synthetic_device_buffer(0x30_0000, 2 * 4 - 4), out).is_err());
        assert!(
            run(logits, mask, out, synthetic_device_buffer(0x40_0000, 2 * 64 - 4)).is_err()
        );
        // A null scratch pointer is rejected like any other required buffer.
        assert!(run(logits, mask, out, CuteafdDeviceBuffer::default()).is_err());
    }

    /// End-to-end ABI check: build the parameter block in Rust, drive the
    /// shipped device kernel through the FFI wrapper, and require the device id
    /// to equal the production CPU oracle exactly (design §12.2).
    ///
    /// `#[ignore]` convention: this test needs a GPU and a built native library,
    /// so it does not run in the default `cargo test` sweep (the design §12.2
    /// pattern, matching the CUDA-ignored daemon tests). Run it explicitly with
    /// the library path, for example:
    ///
    /// ```text
    /// CUTEAFD_NATIVE_LIB=/path/to/libcuteafd_native.so \
    ///   cargo test -p cuteafd-ffi -- --ignored v41_sampler
    /// ```
    ///
    /// The loader is deliberately loud: an explicit `--ignored` run with a
    /// missing library FAILS (`load_device_test_library` errors); it never skips
    /// silently.
    #[test]
    #[ignore = "requires a GPU and a built libcuteafd_native.so; run with --ignored"]
    fn v41_sampler_device_greedy_matches_cpu_oracle() -> Result<()> {
        let library = load_device_test_library()?;
        // Five tiny rows, exercising ties, a mask, and a strict-finites row.
        let vocab = 6_usize;
        let rows = 5_usize;
        let words = vocab.div_ceil(32);
        let logits: Vec<f32> = vec![
            -0.5, 0.1, 0.8, 0.0, 0.8, -0.2, // row 0: tie at 0.8 -> token 2
            -1.0, -0.7, -0.9, -0.8, -0.6, -0.4, // row 1: token 5
            1.25, 1.0, 0.5, 1.25, -2.0, 0.0, // row 2: tie at 1.25 -> token 0
            3.0, 1.0, 2.0, 0.5, 0.25, 0.0, // row 3: masked to tokens 1,3
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // row 4: all equal -> token 0
        ];
        // Row 3 allows only tokens 1 and 3, so token 1 wins over token 3.
        let mask: Vec<u32> = vec![
            u32::MAX,
            u32::MAX,
            u32::MAX,
            (1 << 1) | (1 << 3),
            u32::MAX,
        ];
        let mut params = v41_params(
            rows,
            CUTEAFD_V41_SAMPLER_FLAG_GREEDY | CUTEAFD_V41_SAMPLER_FLAG_STRICT_FINITE,
            CUTEAFD_V41_SAMPLER_NO_MASK_ROW,
        );
        for (row, params) in params.iter_mut().enumerate() {
            params.mask_row = row as u32;
        }
        params[4].flags |= CUTEAFD_V41_SAMPLER_FLAG_NO_MASK;
        params[4].mask_row = CUTEAFD_V41_SAMPLER_NO_MASK_ROW;

        let logits_buffer = library.alloc_device_buffer(rows * vocab * 4)?;
        // K1 dereferences the parameter block on device, so upload it: this is
        // the residency requirement the ABI documents, and the launch below
        // must use this buffer rather than the host slice.
        let params_buffer = library.alloc_device_buffer(rows * CUTEAFD_V41_SAMPLER_PARAM_BYTES)?;
        let mask_buffer = library.alloc_device_buffer(rows * words * 4)?;
        let ids = library.alloc_device_buffer(rows * 4)?;
        let status = library.alloc_device_buffer(rows * 4)?;
        let detail = library.alloc_device_buffer(rows * 4)?;
        let scores = library.alloc_device_buffer(rows * 4)?;
        let scratch = library.alloc_device_buffer(rows * CUTEAFD_V41_SAMPLER_SCRATCH_BYTES)?;

        let result = (|| -> Result<Vec<u32>> {
            // A kernel that never ran would leave these sentinels in place, so
            // "the ids match the oracle" cannot be satisfied by stale memory.
            let sentinel: Vec<u8> = (0..rows).flat_map(|_| 0xDEAD_BEEFu32.to_ne_bytes()).collect();
            library.copy_h2d(ids, &sentinel)?;
            library.copy_h2d(status, &sentinel)?;
            let mut logits_bytes = Vec::with_capacity(logits.len() * 4);
            for value in &logits {
                logits_bytes.extend_from_slice(&value.to_ne_bytes());
            }
            library.copy_h2d(logits_buffer, &logits_bytes)?;
            let mut mask_bytes = Vec::with_capacity(mask.len() * 4);
            for value in &mask {
                mask_bytes.extend_from_slice(&value.to_ne_bytes());
            }
            library.copy_h2d(mask_buffer, &mask_bytes)?;
            // `#[repr(C)]` and pinned to 64 bytes, so its raw bytes are the ABI.
            assert_eq!(
                std::mem::size_of::<CuteafdV41SamplerRow>(),
                CUTEAFD_V41_SAMPLER_PARAM_BYTES
            );
            let param_bytes: Vec<u8> = unsafe {
                std::slice::from_raw_parts(
                    params.as_ptr().cast::<u8>(),
                    params.len() * CUTEAFD_V41_SAMPLER_PARAM_BYTES,
                )
                .to_vec()
            };
            library.copy_h2d(params_buffer, &param_bytes)?;
            library.cuda_v41_target_sample(
                logits_buffer,
                rows,
                vocab,
                vocab,
                &params,
                params_buffer,
                Some(mask_buffer),
                words,
                ids,
                status,
                detail,
                scores,
                None,
                None,
                scratch,
            )?;
            let mut status_bytes = vec![0_u8; rows * 4];
            library.copy_d2h(&mut status_bytes, status)?;
            for row in 0..rows {
                let status = u32::from_ne_bytes(status_bytes[row * 4..row * 4 + 4].try_into().unwrap());
                assert_eq!(status, CUTEAFD_V41_SAMPLER_STATUS_OK, "row {row} status");
            }
            let mut id_bytes = vec![0_u8; rows * 4];
            library.copy_d2h(&mut id_bytes, ids)?;
            Ok((0..rows)
                .map(|row| u32::from_ne_bytes(id_bytes[row * 4..row * 4 + 4].try_into().unwrap()))
                .collect())
        })();

        let mut logits_buffer = logits_buffer;
        let mut params_buffer = params_buffer;
        let mut mask_buffer = mask_buffer;
        let mut ids = ids;
        let mut status = status;
        let mut detail = detail;
        let mut scores = scores;
        let mut scratch = scratch;
        let cleanup = (|| -> Result<()> {
            library.free_device_buffer(&mut logits_buffer)?;
            library.free_device_buffer(&mut params_buffer)?;
            library.free_device_buffer(&mut mask_buffer)?;
            library.free_device_buffer(&mut ids)?;
            library.free_device_buffer(&mut status)?;
            library.free_device_buffer(&mut detail)?;
            library.free_device_buffer(&mut scores)?;
            library.free_device_buffer(&mut scratch)
        })();
        cleanup?;

        // The expected ids come from the production CPU oracle, not from a
        // re-derivation here: `TargetSamplingParams::greedy()` is exact argmax
        // with the lowest id winning a tie, and `select_token_with_uniform`
        // applies the mask first.
        let selected = result?;
        let greedy = cuteafd_core::TargetSamplingParams::greedy();
        let expected: Vec<u32> = (0..rows)
            .map(|row| {
                let row_logits = &logits[row * vocab..(row + 1) * vocab];
                let row_mask = &mask[row * words..(row + 1) * words];
                let mask = if params[row].flags & CUTEAFD_V41_SAMPLER_FLAG_NO_MASK != 0 {
                    None
                } else {
                    Some(row_mask)
                };
                greedy
                    .select_token(row_logits, mask, 0)
                    .expect("oracle selection") as u32
            })
            .collect();
        assert_eq!(selected, expected);
        // Non-vacuity: row 3 is a constrained greedy row whose mask changes the
        // winner (unmasked argmax would be token 0; the grammar allows only
        // tokens 1 and 3), so a kernel that ignored the mask cannot pass this.
        let unmasked_row3 = greedy
            .select_token(&logits[3 * vocab..4 * vocab], None, 0)
            .expect("oracle unmasked selection") as u32;
        assert_eq!(unmasked_row3, 0, "row 3 must differ unmasked");
        assert_eq!(expected[3], 1, "row 3 masked argmax");
        assert_ne!(expected[3], unmasked_row3, "the mask must change row 3's winner");
        // Printed so a captured run is self-evidencing: the ids were produced by
        // the kernel (the sentinels above were overwritten), the parameter block
        // was launched from device memory, and row 3 is a constrained greedy row
        // whose masked winner differs from the unmasked one.
        println!(
            "v4.1 sampler device oracle: rows={rows} vocab={vocab} device_ids={selected:?} \
             cpu_ids={expected:?} masked_row3={} unmasked_row3={unmasked_row3}",
            expected[3]
        );
        Ok(())
    }

    /// Drive one v4.1 sampler batch through the C ABI and return
    /// `(ids, status, total)`. Outputs are pre-filled with a sentinel so a
    /// kernel that never ran cannot satisfy an equality assertion with stale
    /// memory. `total` is the diagnostic `out_total` (so every row here is
    /// launched with `DIAGNOSE`).
    fn run_v41_fast_path_batch(
        library: &NativeLibrary,
        logits: &[f32],
        rows: usize,
        vocab: usize,
        params: &[CuteafdV41SamplerRow],
        mask: Option<&[u32]>,
        mask_words_per_row: usize,
    ) -> Result<(Vec<u32>, Vec<u32>, Vec<f32>)> {
        assert_eq!(params.len(), rows);
        assert_eq!(logits.len(), rows * vocab);
        let words = vocab.div_ceil(32);
        if let Some(mask) = mask {
            assert_eq!(mask.len(), rows * words);
            assert_eq!(mask_words_per_row, words);
        }
        let logits_buffer = library.alloc_device_buffer(rows * vocab * 4)?;
        let params_buffer = library.alloc_device_buffer(rows * CUTEAFD_V41_SAMPLER_PARAM_BYTES)?;
        let mask_buffer = match mask {
            Some(_) => Some(library.alloc_device_buffer(rows * words * 4)?),
            None => None,
        };
        let ids = library.alloc_device_buffer(rows * 4)?;
        let status = library.alloc_device_buffer(rows * 4)?;
        let detail = library.alloc_device_buffer(rows * 4)?;
        let scores = library.alloc_device_buffer(rows * 4)?;
        let total = library.alloc_device_buffer(rows * 4)?;
        let scratch = library.alloc_device_buffer(rows * CUTEAFD_V41_SAMPLER_SCRATCH_BYTES)?;

        let result = (|| -> Result<(Vec<u32>, Vec<u32>, Vec<f32>)> {
            let sentinel: Vec<u8> = (0..rows).flat_map(|_| 0xDEAD_BEEFu32.to_ne_bytes()).collect();
            library.copy_h2d(ids, &sentinel)?;
            library.copy_h2d(status, &sentinel)?;
            let mut logits_bytes = Vec::with_capacity(logits.len() * 4);
            for value in logits {
                logits_bytes.extend_from_slice(&value.to_ne_bytes());
            }
            library.copy_h2d(logits_buffer, &logits_bytes)?;
            if let (Some(mask), Some(mask_buffer)) = (mask, mask_buffer) {
                let mut mask_bytes = Vec::with_capacity(mask.len() * 4);
                for value in mask {
                    mask_bytes.extend_from_slice(&value.to_ne_bytes());
                }
                library.copy_h2d(mask_buffer, &mask_bytes)?;
            }
            let param_bytes: Vec<u8> = unsafe {
                std::slice::from_raw_parts(
                    params.as_ptr().cast::<u8>(),
                    params.len() * CUTEAFD_V41_SAMPLER_PARAM_BYTES,
                )
                .to_vec()
            };
            library.copy_h2d(params_buffer, &param_bytes)?;
            library.cuda_v41_target_sample(
                logits_buffer,
                rows,
                vocab,
                vocab,
                params,
                params_buffer,
                mask_buffer,
                if mask_buffer.is_some() { words } else { 0 },
                ids,
                status,
                detail,
                scores,
                Some(total),
                None,
                scratch,
            )?;
            let read_u32 = |buffer: CuteafdDeviceBuffer| -> Result<Vec<u32>> {
                let mut bytes = vec![0_u8; rows * 4];
                library.copy_d2h(&mut bytes, buffer)?;
                Ok((0..rows)
                    .map(|row| {
                        u32::from_ne_bytes(bytes[row * 4..row * 4 + 4].try_into().unwrap())
                    })
                    .collect())
            };
            let read_f32 = |buffer: CuteafdDeviceBuffer| -> Result<Vec<f32>> {
                let mut bytes = vec![0_u8; rows * 4];
                library.copy_d2h(&mut bytes, buffer)?;
                Ok((0..rows)
                    .map(|row| {
                        f32::from_ne_bytes(bytes[row * 4..row * 4 + 4].try_into().unwrap())
                    })
                    .collect())
            };
            Ok((read_u32(ids)?, read_u32(status)?, read_f32(total)?))
        })();

        let mut logits_buffer = logits_buffer;
        let mut params_buffer = params_buffer;
        let mut mask_buffer = mask_buffer;
        let mut ids = ids;
        let mut status = status;
        let mut detail = detail;
        let mut scores = scores;
        let mut total = total;
        let mut scratch = scratch;
        let cleanup = (|| -> Result<()> {
            library.free_device_buffer(&mut logits_buffer)?;
            library.free_device_buffer(&mut params_buffer)?;
            if let Some(buffer) = mask_buffer.as_mut() {
                library.free_device_buffer(buffer)?;
            }
            library.free_device_buffer(&mut ids)?;
            library.free_device_buffer(&mut status)?;
            library.free_device_buffer(&mut detail)?;
            library.free_device_buffer(&mut scores)?;
            library.free_device_buffer(&mut total)?;
            library.free_device_buffer(&mut scratch)
        })();
        cleanup?;
        result
    }

    /// Build a fast-path (`top_k` disabled, `top_p = 1.0`) per-row block with the
    /// host-precomputed `ln_min_p` the validator requires.
    fn v41_fast_row(
        output_row: u32,
        temperature: f32,
        min_p: f32,
        seed: u64,
        position: u64,
        mask_row: u32,
        flags: u32,
    ) -> CuteafdV41SamplerRow {
        CuteafdV41SamplerRow {
            seed,
            position,
            temperature,
            top_p: 1.0,
            min_p,
            top_k: 0,
            mask_row,
            flags: flags | CUTEAFD_V41_SAMPLER_FLAG_DIAGNOSE,
            output_row,
            ln_min_p: if min_p > 0.0 { min_p.ln() } else { f32::NEG_INFINITY },
            ..CuteafdV41SamplerRow::default()
        }
    }

    /// Chunk-2 FFI oracle: drive the shipped K2 fast path through the C ABI and
    /// compare against the **production** CPU sampler
    /// (`cuteafd_core::TargetSamplingParams`), not a reimplementation. Small
    /// deterministic rows pin the filter/RNG/mask chain exactly; two realistic
    /// 129,280-wide rows pin the full-vocabulary scan. A near-uniform row is
    /// included to record the documented accumulation residual rather than to
    /// assert zero (§6.3c): the device token must still be a real survivor.
    ///
    /// `#[ignore]` convention: this test needs a GPU and a built native library,
    /// so it does not run in the default `cargo test` sweep (the design §12.2
    /// pattern, matching the CUDA-ignored daemon tests). Run it explicitly with
    /// the library path, for example:
    ///
    /// ```text
    /// CUTEAFD_NATIVE_LIB=/path/to/libcuteafd_native.so \
    ///   cargo test -p cuteafd-ffi -- --ignored v41_sampler
    /// ```
    ///
    /// The loader is deliberately loud: an explicit `--ignored` run with a
    /// missing library FAILS (`load_device_test_library` errors); it never skips
    /// silently.
    #[test]
    #[ignore = "requires a GPU and a built libcuteafd_native.so; run with --ignored"]
    fn v41_sampler_device_fast_path_matches_cpu_oracle() -> Result<()> {
        let library = load_device_test_library()?;

        // ---- small deterministic rows, vocab 6 ----
        let vocab = 6_usize;
        let rows = 4_usize;
        let words = vocab.div_ceil(32);
        let logits: Vec<f32> = vec![
            0.5, 1.25, -0.75, 0.25, 2.0, -1.5, // row 0
            -0.5, 0.0, 0.75, -1.25, 1.5, 0.25, // row 1
            1.0, 2.0, 3.0, 0.5, -2.0, 0.0,     // row 2: masked to tokens 0,3,4
            -1.0, -0.5, 0.0, 0.5, 1.0, 1.5,    // row 3
        ];
        // Row 2 allows tokens 0, 3 and 4; unmasked the argmax is token 2 (3.0),
        // so the mask must remove the *winner* to be non-vacuous, leaving token 0
        // (1.0) as the masked winner.
        let mask: Vec<u32> = vec![u32::MAX, u32::MAX, (1 << 0) | (1 << 3) | (1 << 4), u32::MAX];
        let small = vec![
            v41_fast_row(0, 0.7, 0.0, 1, 0, CUTEAFD_V41_SAMPLER_NO_MASK_ROW, CUTEAFD_V41_SAMPLER_FLAG_NO_MASK),
            v41_fast_row(1, 0.7, 0.5, 1, 1, CUTEAFD_V41_SAMPLER_NO_MASK_ROW, CUTEAFD_V41_SAMPLER_FLAG_NO_MASK),
            v41_fast_row(2, 0.2, 0.0, 0xDEAD_BEEF, 2, 2, 0),
            v41_fast_row(3, 2.0, 0.0, 7, 3, CUTEAFD_V41_SAMPLER_NO_MASK_ROW, CUTEAFD_V41_SAMPLER_FLAG_NO_MASK),
        ];
        let (small_ids, small_status, small_total) = run_v41_fast_path_batch(
            &library,
            &logits,
            rows,
            vocab,
            &small,
            Some(&mask),
            words,
        )?;
        for row in 0..rows {
            assert_eq!(
                small_status[row], CUTEAFD_V41_SAMPLER_STATUS_OK,
                "small row {row} status"
            );
            assert_ne!(small_ids[row], 0xDEAD_BEEF, "small row {row} was not written");
            assert!(small_ids[row] < vocab as u32, "small row {row} in-vocab");
            assert!(small_total[row] > 0.0, "small row {row} total");
            let row_mask = &mask[row * words..(row + 1) * words];
            let mask_arg = if small[row].flags & CUTEAFD_V41_SAMPLER_FLAG_NO_MASK != 0 {
                None
            } else {
                Some(row_mask)
            };
            let oracle = cuteafd_core::TargetSamplingParams::new(
                small[row].temperature,
                small[row].top_p,
                None,
                small[row].min_p,
                small[row].seed,
            )
            .expect("oracle params")
            .select_token(&logits[row * vocab..(row + 1) * vocab], mask_arg, small[row].position)
            .expect("oracle selection") as u32;
            assert_eq!(
                small_ids[row], oracle,
                "small row {row} device id must equal the production CPU sampler"
            );
        }
        // Non-vacuity for the mask chain: row 2 allows tokens 0, 3 and 4 (the
        // mask removes the unmasked winner, token 2). If the mask were ignored the
        // ids could still coincide, so pin the masked top set explicitly.
        let masked_oracle = cuteafd_core::TargetSamplingParams::new(0.2, 1.0, None, 0.0, 0xDEAD_BEEF)
            .expect("params")
            .select_token(&logits[2 * vocab..3 * vocab], Some(&mask[2 * words..3 * words]), 2)
            .expect("masked oracle") as u32;
        assert!(
            [0u32, 3, 4].contains(&masked_oracle),
            "masked oracle {masked_oracle} must be one of the allowed tokens"
        );

        // ---- realistic 129,280-wide rows ----
        const WIDE: usize = 129_280;
        // Phase-0 generator, in tree: `-8 + 10*splitmix_unit(i)`, with a +20
        // boost on the first four tokens. Token 0 dominates by many nats at any
        // sampled temperature, so the drawn id is deterministic and exact.
        let splitmix_unit = |index: u64| -> f32 {
            let mut hash = index.wrapping_mul(0x9e37_79b9_7f4a_7c15);
            hash = (hash ^ (hash >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            hash = (hash ^ (hash >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            hash ^= hash >> 31;
            (hash >> 40) as f32 / 16_777_216.0
        };
        let mut peaked = Vec::with_capacity(4 * WIDE);
        let mut moderate = Vec::with_capacity(4 * WIDE);
        let mut near_uniform = Vec::with_capacity(4 * WIDE);
        for _ in 0..4 {
            for token in 0..WIDE {
                let mut value = -8.0 + 10.0 * splitmix_unit(token as u64);
                if token < 4 {
                    value += 20.0 - token as f32 * 2.0;
                }
                peaked.push(value);
                moderate.push(if token < 64 { -0.25 * token as f32 } else { -30.0 });
                near_uniform.push((splitmix_unit(token as u64) - 0.5) * 1.0e-3);
            }
        }
        let wide_rows = 4_usize;
        let wide_seeds = [1_u64, 20_260_922, 0xDEAD_BEEF, 987_654_321];
        let wide = vec![
            v41_fast_row(0, 0.7, 0.0, wide_seeds[0], 0, CUTEAFD_V41_SAMPLER_NO_MASK_ROW, CUTEAFD_V41_SAMPLER_FLAG_NO_MASK),
            v41_fast_row(1, 0.7, 0.05, wide_seeds[1], 1, CUTEAFD_V41_SAMPLER_NO_MASK_ROW, CUTEAFD_V41_SAMPLER_FLAG_NO_MASK),
            v41_fast_row(2, 1.0, 0.0, wide_seeds[2], 2, CUTEAFD_V41_SAMPLER_NO_MASK_ROW, CUTEAFD_V41_SAMPLER_FLAG_NO_MASK),
            v41_fast_row(3, 0.7, 0.0, wide_seeds[3], 5, CUTEAFD_V41_SAMPLER_NO_MASK_ROW, CUTEAFD_V41_SAMPLER_FLAG_NO_MASK),
        ];

        let (peaked_ids, peaked_status, peaked_total) =
            run_v41_fast_path_batch(&library, &peaked, wide_rows, WIDE, &wide, None, 0)?;
        for row in 0..wide_rows {
            assert_eq!(peaked_status[row], CUTEAFD_V41_SAMPLER_STATUS_OK);
            assert!(
                (peaked_ids[row] as usize) < 4,
                "peaked row {row}: the first four tokens carry the +20 boost, device id {}",
                peaked_ids[row]
            );
            assert!(peaked_total[row].is_finite() && peaked_total[row] > 0.0);
            let oracle = cuteafd_core::TargetSamplingParams::new(
                wide[row].temperature,
                wide[row].top_p,
                None,
                wide[row].min_p,
                wide[row].seed,
            )
            .expect("params")
            .select_token(&peaked[row * WIDE..(row + 1) * WIDE], None, wide[row].position)
            .expect("oracle selection") as u32;
            assert_eq!(
                peaked_ids[row], oracle,
                "peaked row {row} device id must equal the production CPU sampler"
            );
        }

        let moderate_rows = 2_usize;
        let moderate_params = vec![
            v41_fast_row(0, 0.7, 0.05, wide_seeds[0], 0, CUTEAFD_V41_SAMPLER_NO_MASK_ROW, CUTEAFD_V41_SAMPLER_FLAG_NO_MASK),
            v41_fast_row(1, 0.7, 0.05, wide_seeds[1], 1, CUTEAFD_V41_SAMPLER_NO_MASK_ROW, CUTEAFD_V41_SAMPLER_FLAG_NO_MASK),
        ];
        let moderate_prefix = &moderate[..moderate_rows * WIDE];
        let (moderate_ids, moderate_status, _) = run_v41_fast_path_batch(
            &library,
            moderate_prefix,
            moderate_rows,
            WIDE,
            &moderate_params,
            None,
            0,
        )?;
        for row in 0..moderate_rows {
            assert_eq!(moderate_status[row], CUTEAFD_V41_SAMPLER_STATUS_OK);
            let oracle = cuteafd_core::TargetSamplingParams::new(
                moderate_params[row].temperature,
                moderate_params[row].top_p,
                None,
                moderate_params[row].min_p,
                moderate_params[row].seed,
            )
            .expect("params")
            .select_token(
                &moderate[row * WIDE..(row + 1) * WIDE],
                None,
                moderate_params[row].position,
            )
            .expect("oracle selection") as u32;
            assert_eq!(
                moderate_ids[row], oracle,
                "moderate row {row}: device id must equal the production CPU sampler"
            );
        }

        // Near-uniform is the pathological wide-support row: the accumulation
        // residual can move the crossing by a token, so this records the
        // divergence (§6.3c) and only requires a valid survivor.
        let near_rows = 2_usize;
        let near_params = vec![
            v41_fast_row(0, 0.7, 0.0, wide_seeds[0], 5, CUTEAFD_V41_SAMPLER_NO_MASK_ROW, CUTEAFD_V41_SAMPLER_FLAG_NO_MASK),
            v41_fast_row(1, 0.7, 0.0, wide_seeds[1], 6, CUTEAFD_V41_SAMPLER_NO_MASK_ROW, CUTEAFD_V41_SAMPLER_FLAG_NO_MASK),
        ];
        let near_prefix = &near_uniform[..near_rows * WIDE];
        let (near_ids, near_status, _) = run_v41_fast_path_batch(
            &library,
            near_prefix,
            near_rows,
            WIDE,
            &near_params,
            None,
            0,
        )?;
        let mut near_differing = 0_usize;
        for row in 0..near_rows {
            assert_eq!(near_status[row], CUTEAFD_V41_SAMPLER_STATUS_OK);
            assert!((near_ids[row] as usize) < WIDE);
            let oracle = cuteafd_core::TargetSamplingParams::new(
                near_params[row].temperature,
                near_params[row].top_p,
                None,
                near_params[row].min_p,
                near_params[row].seed,
            )
            .expect("params")
            .select_token(
                &near_uniform[row * WIDE..(row + 1) * WIDE],
                None,
                near_params[row].position,
            )
            .expect("oracle selection") as u32;
            if near_ids[row] != oracle {
                near_differing += 1;
            }
        }
        println!(
            "v4.1 sampler fast-path FFI oracle: small_rows={rows} vocab={vocab} \
             peaked_rows={wide_rows} moderate_rows={moderate_rows} near_uniform_rows={near_rows} \
             near_uniform_diverged={near_differing}/{near_rows}"
        );
        Ok(())
    }

    /// One chunk-3a K3/K4 per-row parameter block (`top_p = 1.0`, so the row is
    /// the ordered profile).
    fn v41_topk_row(
        output_row: u32,
        temperature: f32,
        top_k: u32,
        min_p: f32,
        mask_row: u32,
        flags: u32,
    ) -> CuteafdV41SamplerRow {
        CuteafdV41SamplerRow {
            seed: 0x7f4a_7c15_9e37_79b9,
            position: output_row as u64,
            temperature,
            top_p: 1.0,
            min_p,
            top_k,
            mask_row,
            flags,
            output_row,
            ln_min_p: if min_p > 0.0 { min_p.ln() } else { f32::NEG_INFINITY },
            ..CuteafdV41SamplerRow::default()
        }
    }

    /// Host port of the shipped `cuteafd_order_key` (design §4.3).
    fn v41_order_key(scaled: f32) -> u32 {
        let value = if scaled == 0.0 { 0.0 } else { scaled };
        let bits = value.to_bits();
        if bits & 0x8000_0000 != 0 {
            !bits
        } else {
            bits ^ 0x8000_0000
        }
    }

    /// The CPU's ordered top-k branch (`target_sampling.rs:477-497`) as a host
    /// oracle: survivors under `min_p` and the optional mask, sorted by the
    /// served comparator (scaled descending, ascending id), truncated to
    /// `top_k`. Returns the exact `ranked[..k]` and `above_count = C_gt(kth)`.
    fn v41_expected_topk(
        logits: &[f32],
        temperature: f32,
        top_k: usize,
        min_p: f32,
        mask: Option<&[u32]>,
    ) -> (Vec<u32>, u32) {
        let inv = 1.0f32 / temperature;
        let mut max_scaled = f32::NEG_INFINITY;
        for (token, &value) in logits.iter().enumerate() {
            if mask.is_none_or(|words| words[token / 32] & (1 << (token % 32)) != 0) {
                max_scaled = max_scaled.max(value * inv);
            }
        }
        let min_scaled = if min_p > 0.0 {
            max_scaled + min_p.ln()
        } else {
            f32::NEG_INFINITY
        };
        let mut ranked: Vec<(f32, u32)> = logits
            .iter()
            .enumerate()
            .filter(|(token, &value)| {
                mask.is_none_or(|words| words[token / 32] & (1 << (token % 32)) != 0)
                    && value * inv >= min_scaled
            })
            .map(|(token, &value)| (value * inv, token as u32))
            .collect();
        ranked.sort_unstable_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.1.cmp(&b.1))
        });
        let kth_key = v41_order_key(ranked[top_k - 1].0);
        let above = ranked
            .iter()
            .filter(|(scaled, _)| v41_order_key(*scaled) > kth_key)
            .count() as u32;
        (
            ranked.into_iter().take(top_k).map(|(_, id)| id).collect(),
            above,
        )
    }

    /// Outputs of one K1 + K3/K4 batch. `ranked` is `rows * capacity` (empty for
    /// a selection-only call); the scratch fields are parsed at the header's
    /// offsets (`kth_value_bits` +16, `above_count` +20).
    struct TopkRun {
        ranked: Vec<u32>,
        retained: Vec<u32>,
        passes: Vec<u32>,
        above: Vec<u32>,
        kth_bits: Vec<u32>,
    }

    /// Drive K1 then the chunk-3a K3/K4 stages through the FFI. Every output is
    /// pre-filled with `0xDEADBEEF`, so a kernel that never ran cannot satisfy an
    /// equality assertion with stale memory.
    fn run_v41_topk_batch(
        library: &NativeLibrary,
        logits: &[f32],
        rows: usize,
        vocab: usize,
        params: &[CuteafdV41SamplerRow],
        mask: Option<&[u32]>,
        capacity: usize,
    ) -> Result<TopkRun> {
        assert_eq!(params.len(), rows);
        assert_eq!(logits.len(), rows * vocab);
        let words = vocab.div_ceil(32);
        if let Some(mask) = mask {
            assert_eq!(mask.len(), rows * words);
        }
        let logits_buffer = library.alloc_device_buffer(rows * vocab * 4)?;
        let params_buffer = library.alloc_device_buffer(rows * CUTEAFD_V41_SAMPLER_PARAM_BYTES)?;
        let mask_buffer = match mask {
            Some(_) => Some(library.alloc_device_buffer(rows * words * 4)?),
            None => None,
        };
        let ids = library.alloc_device_buffer(rows * 4)?;
        let status = library.alloc_device_buffer(rows * 4)?;
        let detail = library.alloc_device_buffer(rows * 4)?;
        let scores = library.alloc_device_buffer(rows * 4)?;
        let scratch = library.alloc_device_buffer(rows * CUTEAFD_V41_SAMPLER_SCRATCH_BYTES)?;
        let retained = library.alloc_device_buffer(rows * 4)?;
        let passes = library.alloc_device_buffer(rows * 4)?;
        let rank_ids = match capacity {
            0 => None,
            _ => Some(library.alloc_device_buffer(rows * capacity * 4)?),
        };
        let rank_scratch = match capacity {
            0 => None,
            _ => Some(library.alloc_device_buffer(rows * capacity * 8)?),
        };

        let result = (|| -> Result<TopkRun> {
            let sentinel: Vec<u8> = (0..rows).flat_map(|_| 0xDEAD_BEEFu32.to_ne_bytes()).collect();
            library.copy_h2d(retained, &sentinel)?;
            library.copy_h2d(passes, &sentinel)?;
            if let Some(rank_ids) = rank_ids {
                let arena: Vec<u8> =
                    (0..rows * capacity).flat_map(|_| 0xDEAD_BEEFu32.to_ne_bytes()).collect();
                library.copy_h2d(rank_ids, &arena)?;
            }
            let mut logits_bytes = Vec::with_capacity(logits.len() * 4);
            for value in logits {
                logits_bytes.extend_from_slice(&value.to_ne_bytes());
            }
            library.copy_h2d(logits_buffer, &logits_bytes)?;
            if let (Some(mask), Some(mask_buffer)) = (mask, mask_buffer) {
                let mut mask_bytes = Vec::with_capacity(mask.len() * 4);
                for value in mask {
                    mask_bytes.extend_from_slice(&value.to_ne_bytes());
                }
                library.copy_h2d(mask_buffer, &mask_bytes)?;
            }
            let param_bytes: Vec<u8> = unsafe {
                std::slice::from_raw_parts(
                    params.as_ptr().cast::<u8>(),
                    params.len() * CUTEAFD_V41_SAMPLER_PARAM_BYTES,
                )
                .to_vec()
            };
            library.copy_h2d(params_buffer, &param_bytes)?;
            library.cuda_v41_target_sample(
                logits_buffer,
                rows,
                vocab,
                vocab,
                params,
                params_buffer,
                mask_buffer,
                if mask_buffer.is_some() { words } else { 0 },
                ids,
                status,
                detail,
                scores,
                None,
                None,
                scratch,
            )?;
            library.cuda_v41_topk_select(
                logits_buffer,
                rows,
                vocab,
                vocab,
                params,
                params_buffer,
                mask_buffer,
                if mask_buffer.is_some() { words } else { 0 },
                rank_ids,
                rank_scratch,
                capacity,
                retained,
                passes,
                scratch,
            )?;
            let read_u32 = |buffer: CuteafdDeviceBuffer, count: usize| -> Result<Vec<u32>> {
                let mut bytes = vec![0_u8; count * 4];
                library.copy_d2h(&mut bytes, buffer)?;
                Ok((0..count)
                    .map(|index| {
                        u32::from_ne_bytes(bytes[index * 4..index * 4 + 4].try_into().unwrap())
                    })
                    .collect())
            };
            let mut scratch_bytes = vec![0_u8; rows * CUTEAFD_V41_SAMPLER_SCRATCH_BYTES];
            library.copy_d2h(&mut scratch_bytes, scratch)?;
            let field = |row: usize, offset: usize| -> u32 {
                u32::from_ne_bytes(
                    scratch_bytes[row * CUTEAFD_V41_SAMPLER_SCRATCH_BYTES + offset
                        ..row * CUTEAFD_V41_SAMPLER_SCRATCH_BYTES + offset + 4]
                        .try_into()
                        .unwrap(),
                )
            };
            Ok(TopkRun {
                ranked: match rank_ids {
                    Some(rank_ids) => read_u32(rank_ids, rows * capacity)?,
                    None => Vec::new(),
                },
                retained: read_u32(retained, rows)?,
                passes: read_u32(passes, rows)?,
                above: (0..rows).map(|row| field(row, 20)).collect(),
                kth_bits: (0..rows).map(|row| field(row, 16)).collect(),
            })
        })();

        let mut logits_buffer = logits_buffer;
        let mut params_buffer = params_buffer;
        let mut mask_buffer = mask_buffer;
        let mut ids = ids;
        let mut status = status;
        let mut detail = detail;
        let mut scores = scores;
        let mut scratch = scratch;
        let mut retained = retained;
        let mut passes = passes;
        let mut rank_ids = rank_ids;
        let mut rank_scratch = rank_scratch;
        let cleanup = (|| -> Result<()> {
            library.free_device_buffer(&mut logits_buffer)?;
            library.free_device_buffer(&mut params_buffer)?;
            if let Some(buffer) = mask_buffer.as_mut() {
                library.free_device_buffer(buffer)?;
            }
            library.free_device_buffer(&mut ids)?;
            library.free_device_buffer(&mut status)?;
            library.free_device_buffer(&mut detail)?;
            library.free_device_buffer(&mut scores)?;
            library.free_device_buffer(&mut scratch)?;
            library.free_device_buffer(&mut retained)?;
            library.free_device_buffer(&mut passes)?;
            if let Some(buffer) = rank_ids.as_mut() {
                library.free_device_buffer(buffer)?;
            }
            if let Some(buffer) = rank_scratch.as_mut() {
                library.free_device_buffer(buffer)?;
            }
            Ok(())
        })();
        cleanup?;
        result
    }

    /// Chunk-3a FFI oracle: drive the shipped K3/K4 stages through the C ABI and
    /// require the **exact retained id set and rank order** of a faithful host
    /// port of the CPU comparator (`v41_expected_topk`) on deterministic in-tree
    /// rows, including no-op, greedy, tie-heavy, all-tied, masked and wide
    /// (129,280) rows. A token-level end-to-end comparison against
    /// `cuteafd_core::TargetSamplingParams` belongs to chunk 3b, which adds K5's
    /// nucleus and draw; K3/K4's contract is the ranked set itself.
    ///
    /// `#[ignore]` convention: this test needs a GPU and a built native library,
    /// so it does not run in the default `cargo test` sweep (the design §12.2
    /// pattern). Run it explicitly with the library path, for example:
    ///
    /// ```text
    /// CUTEAFD_NATIVE_LIB=/path/to/libcuteafd_native.so \
    ///   cargo test -p cuteafd-ffi -- --ignored v41_sampler
    /// ```
    ///
    /// The loader is deliberately loud: an explicit `--ignored` run with a
    /// missing library FAILS (`load_device_test_library` errors); it never skips
    /// silently.
    #[test]
    #[ignore = "requires a GPU and a built libcuteafd_native.so; run with --ignored"]
    fn v41_sampler_device_topk_set_and_order_matches_cpu_oracle() -> Result<()> {
        let library = load_device_test_library()?;

        // ---- small deterministic rows, vocab 8 ----
        let vocab = 8_usize;
        let rows = 7_usize;
        let words = vocab.div_ceil(32);
        let logits: Vec<f32> = vec![
            0.5, 1.25, -0.75, 0.25, 2.0, 2.0, -1.5, 0.0, // row 0: tie at 2.0 -> k=3
            0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,      // row 1: all tied, k=5
            1.0, 2.0, 3.0, 0.5, -2.0, 0.0, 0.25, -0.5,   // row 2: k=0 disabled
            0.5, 1.25, -0.75, 0.25, 2.0, 2.0, -1.5, 0.0, // row 3: k=1 greedy
            -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5,    // row 4: k=V no-op
            2.5, 2.0, 1.5, 1.0, 0.5, 0.0, -0.5, -1.0,    // row 5: k>V no-op
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,      // row 6: masked, k=2
        ];
        let mask: Vec<u32> = vec![
            u32::MAX,
            u32::MAX,
            u32::MAX,
            u32::MAX,
            u32::MAX,
            u32::MAX,
            (1 << 1) | (1 << 4) | (1 << 7),
        ];
        let params = vec![
            v41_topk_row(0, 0.7, 3, 0.0, CUTEAFD_V41_SAMPLER_NO_MASK_ROW, CUTEAFD_V41_SAMPLER_FLAG_NO_MASK),
            v41_topk_row(1, 0.7, 5, 0.0, CUTEAFD_V41_SAMPLER_NO_MASK_ROW, CUTEAFD_V41_SAMPLER_FLAG_NO_MASK),
            v41_topk_row(2, 0.7, 0, 0.0, CUTEAFD_V41_SAMPLER_NO_MASK_ROW, CUTEAFD_V41_SAMPLER_FLAG_NO_MASK),
            v41_topk_row(3, 0.7, 1, 0.0, CUTEAFD_V41_SAMPLER_NO_MASK_ROW, CUTEAFD_V41_SAMPLER_FLAG_NO_MASK),
            v41_topk_row(4, 0.7, 8, 0.0, CUTEAFD_V41_SAMPLER_NO_MASK_ROW, CUTEAFD_V41_SAMPLER_FLAG_NO_MASK),
            v41_topk_row(5, 0.7, 9, 0.0, CUTEAFD_V41_SAMPLER_NO_MASK_ROW, CUTEAFD_V41_SAMPLER_FLAG_NO_MASK),
            v41_topk_row(6, 0.7, 2, 0.0, 6, 0),
        ];
        let capacity = 9_usize;
        let run = run_v41_topk_batch(&library, &logits, rows, vocab, &params, Some(&mask), capacity)?;
        for row in 0..rows {
            let row_logits = &logits[row * vocab..(row + 1) * vocab];
            let top_k = params[row].top_k as usize;
            let runs = top_k > 1 && top_k < vocab && params[row].temperature >= 1e-5;
            if !runs {
                assert_eq!(run.retained[row], 0, "row {row} is a no-op");
                assert_eq!(run.passes[row], 0, "row {row} is a no-op");
                for rank in 0..capacity {
                    assert_eq!(
                        run.ranked[row * capacity + rank],
                        0xDEAD_BEEF,
                        "row {row} must leave the rank-order arena untouched"
                    );
                }
                continue;
            }
            let (expected, above) = {
                let row_mask = if params[row].flags & CUTEAFD_V41_SAMPLER_FLAG_NO_MASK != 0 {
                    None
                } else {
                    Some(&mask[row * words..(row + 1) * words])
                };
                v41_expected_topk(row_logits, params[row].temperature, top_k, 0.0, row_mask)
            };
            let got = &run.ranked[row * capacity..row * capacity + top_k];
            assert_eq!(
                got, expected.as_slice(),
                "row {row} k={top_k} exact CPU rank order"
            );
            let mut got_sorted = got.to_vec();
            let mut want_sorted = expected.clone();
            got_sorted.sort_unstable();
            want_sorted.sort_unstable();
            assert_eq!(got_sorted, want_sorted, "row {row} exact retained id multiset");
            assert_eq!(run.retained[row] as usize, top_k, "row {row} retained count");
            assert!(run.passes[row] >= 1 && run.passes[row] <= 32, "row {row} passes");
            assert_eq!(run.above[row], above, "row {row} above_count");
            let kth_scaled = row_logits[*expected.last().unwrap() as usize]
                * (1.0f32 / params[row].temperature);
            // `order_key` canonicalizes -0.0 to +0.0 (the CPU's
            // `descending_radix_key` does the same), so the device publishes +0.0.
            let kth_scaled = if kth_scaled == 0.0 { 0.0 } else { kth_scaled };
            assert_eq!(
                run.kth_bits[row],
                kth_scaled.to_bits(),
                "row {row} kth_value_bits equals the oracle's k-th value"
            );
        }

        // ---- tie-heavy: 2048 survivors in one flat group, k=1500 ----
        let tie_vocab = 2048_usize;
        let mut tie_logits = vec![0.0f32; tie_vocab];
        tie_logits[0] = 5.0;
        tie_logits[1] = 4.0;
        let tie_params = vec![v41_topk_row(
            0,
            1.0,
            1500,
            0.0,
            CUTEAFD_V41_SAMPLER_NO_MASK_ROW,
            CUTEAFD_V41_SAMPLER_FLAG_NO_MASK,
        )];
        let tie_run = run_v41_topk_batch(&library, &tie_logits, 1, tie_vocab, &tie_params, None, 1500)?;
        let (tie_expected, tie_above) = v41_expected_topk(&tie_logits, 1.0, 1500, 0.0, None);
        assert_eq!(tie_run.above[0], tie_above, "tie-heavy above_count");
        assert_eq!(tie_above, 2, "exactly two leaders above the flat group");
        assert_eq!(
            &tie_run.ranked[..1500],
            tie_expected.as_slice(),
            "tie-heavy exact rank order"
        );
        assert_eq!(tie_run.retained[0], 1500);

        // ---- all-tied: ids 0..k-1 ----
        let flat_vocab = 512_usize;
        let flat_logits = vec![0.25f32; flat_vocab];
        let flat_k = 40_usize;
        let flat_params = vec![v41_topk_row(
            0,
            1.0,
            flat_k as u32,
            0.0,
            CUTEAFD_V41_SAMPLER_NO_MASK_ROW,
            CUTEAFD_V41_SAMPLER_FLAG_NO_MASK,
        )];
        let flat_run = run_v41_topk_batch(&library, &flat_logits, 1, flat_vocab, &flat_params, None, flat_k)?;
        let expected_flat: Vec<u32> = (0..flat_k as u32).collect();
        assert_eq!(&flat_run.ranked[..flat_k], expected_flat.as_slice(), "all-tied ids 0..k-1");
        assert_eq!(flat_run.above[0], 0, "all-tied above_count is zero");

        // ---- wide 129,280 in-tree rows (the phase-0 generator) ----
        const WIDE: usize = 129_280;
        let splitmix_unit = |index: u64| -> f32 {
            let mut hash = index.wrapping_mul(0x9e37_79b9_7f4a_7c15);
            hash = (hash ^ (hash >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            hash = (hash ^ (hash >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            hash ^= hash >> 31;
            (hash >> 40) as f32 / 16_777_216.0
        };
        let wide_logits: Vec<f32> =
            (0..WIDE).map(|token| -8.0 + 10.0 * splitmix_unit(token as u64)).collect();
        for &k in &[40_u32, 1000_u32] {
            let wide_params = vec![v41_topk_row(
                0,
                0.7,
                k,
                0.0,
                CUTEAFD_V41_SAMPLER_NO_MASK_ROW,
                CUTEAFD_V41_SAMPLER_FLAG_NO_MASK,
            )];
            let wide_run = run_v41_topk_batch(
                &library,
                &wide_logits,
                1,
                WIDE,
                &wide_params,
                None,
                k as usize,
            )?;
            let (wide_expected, wide_above) = v41_expected_topk(&wide_logits, 0.7, k as usize, 0.0, None);
            assert_eq!(wide_run.retained[0], k, "wide k={k} retained count");
            assert_eq!(wide_run.above[0], wide_above, "wide k={k} above_count");
            assert_eq!(
                &wide_run.ranked[..k as usize],
                wide_expected.as_slice(),
                "wide k={k} exact rank order"
            );
        }

        // ---- wide selection-only: survivor_count - 1 (materialization is O(k^2)) ----
        let big_k = (WIDE - 1) as u32;
        let big_params = vec![v41_topk_row(
            0,
            0.7,
            big_k,
            0.0,
            CUTEAFD_V41_SAMPLER_NO_MASK_ROW,
            CUTEAFD_V41_SAMPLER_FLAG_NO_MASK,
        )];
        let big_run =
            run_v41_topk_batch(&library, &wide_logits, 1, WIDE, &big_params, None, 0)?;
        assert_eq!(big_run.retained[0], big_k, "selection-only retained count");
        assert!(big_run.ranked.is_empty(), "selection-only writes no arena");
        let (big_expected, big_above) = v41_expected_topk(&wide_logits, 0.7, WIDE - 1, 0.0, None);
        assert_eq!(big_run.above[0], big_above, "selection-only above_count");
        assert_eq!(big_expected.len(), WIDE - 1);

        println!(
            "v4.1 sampler top-k FFI oracle: small_rows={rows} vocab={vocab} \
             tie_rows=1 vocab={tie_vocab} all_tied_k={flat_k} wide_k=[40,1000] wide_selection_k={big_k}"
        );
        Ok(())
    }


    fn protocol_v2_frame(kind: u16, payload_bytes: usize) -> Vec<u8> {
        const HEADER_BYTES: usize = 96;
        let mut frame = vec![0_u8; HEADER_BYTES + payload_bytes];
        frame[..8].copy_from_slice(b"CUTEAFD3");
        frame[8..10].copy_from_slice(&3_u16.to_le_bytes());
        frame[10..12].copy_from_slice(&kind.to_le_bytes());
        frame[12..16].copy_from_slice(&(HEADER_BYTES as u32).to_le_bytes());
        let frame_len = frame.len() as u64;
        let wire_bytes_offset = if kind == 1 { 76 } else { 60 };
        frame[wire_bytes_offset..wire_bytes_offset + 8].copy_from_slice(&frame_len.to_le_bytes());
        for (idx, byte) in frame[HEADER_BYTES..].iter_mut().enumerate() {
            *byte = ((idx * 17 + kind as usize) & 0xff) as u8;
        }
        frame
    }












































    #[test]
    fn native_version_call() -> Result<()> {
        let Some(library) = load_test_library()? else {
            return Ok(());
        };
        let version = library.version()?;
        assert!(version.contains("cuteafd_native"));
        Ok(())
    }


    #[test]
    fn allocate_copy_free_roundtrip() -> Result<()> {
        let Some(library) = load_test_library()? else {
            return Ok(());
        };
        let input = b"cuteafd native ffi roundtrip".to_vec();
        let mut output = vec![0_u8; input.len()];
        let mut buffer = library.alloc_device_buffer(input.len())?;

        library.copy_h2d(buffer, &input)?;
        library.copy_d2h(&mut output, buffer)?;
        assert_eq!(output, input);

        library.free_device_buffer(&mut buffer)?;
        assert!(buffer.ptr.is_null());
        assert_eq!(buffer.bytes, 0);
        Ok(())
    }

    #[test]
    fn managed_device_buffer_copy_free_roundtrip() -> Result<()> {
        let Some(library) = load_test_library()? else {
            return Ok(());
        };
        let input = b"cuteafd managed native ffi roundtrip".to_vec();
        let mut output = vec![0_u8; input.len()];
        let mut buffer = library.alloc_managed_device_buffer(input.len())?;

        assert!(!buffer.ptr.is_null());
        assert!(buffer.bytes >= input.len());
        assert_ne!(buffer.flags & CUTEAFD_DEVICE_BUFFER_FLAG_MANAGED, 0);
        library.copy_h2d(buffer, &input)?;
        library.copy_d2h(&mut output, buffer)?;
        assert_eq!(output, input);

        library.free_device_buffer(&mut buffer)?;
        assert!(buffer.ptr.is_null());
        assert_eq!(buffer.bytes, 0);
        Ok(())
    }

    #[test]
    fn copy_h2d_reuses_synchronous_pinned_staging() -> Result<()> {
        let Some(library) = load_test_library()? else {
            return Ok(());
        };
        let first = b"cuteafd native ffi reusable sync h2d staging".to_vec();
        let second = b"shorter sync h2d payload".to_vec();
        let mut output = vec![0_u8; first.len()];
        let mut buffer = library.alloc_device_buffer(first.len())?;

        library.copy_h2d(buffer, &first)?;
        let first_staging = library
            .sync_h2d_staging_snapshot()
            .expect("copy_h2d allocates reusable staging");
        library.copy_d2h(&mut output, buffer)?;
        assert_eq!(output, first);

        library.copy_h2d(buffer, &second)?;
        let second_staging = library
            .sync_h2d_staging_snapshot()
            .expect("copy_h2d keeps reusable staging");
        output[..second.len()].fill(0);
        library.copy_d2h(&mut output[..second.len()], buffer)?;
        assert_eq!(&output[..second.len()], second.as_slice());
        assert_eq!(second_staging.0, first_staging.0);
        assert_eq!(second_staging.1, first_staging.1);
        assert!(second_staging.1 >= first.len());

        library.free_device_buffer(&mut buffer)?;
        Ok(())
    }

    #[test]
    fn device_to_device_copy_roundtrip() -> Result<()> {
        let Some(library) = load_test_library()? else {
            return Ok(());
        };
        let input = b"cuteafd native ffi d2d roundtrip".to_vec();
        let mut output = vec![0_u8; input.len()];
        let mut src = library.alloc_device_buffer(input.len())?;
        let mut dst = library.alloc_device_buffer(input.len())?;

        library.copy_h2d(src, &input)?;
        library.copy_d2d(dst, src, input.len())?;
        library.copy_d2h(&mut output, dst)?;
        assert_eq!(output, input);

        library.free_device_buffer(&mut src)?;
        library.free_device_buffer(&mut dst)?;
        Ok(())
    }

    #[test]
    fn pinned_host_buffer_copy_roundtrip() -> Result<()> {
        let Some(library) = load_test_library()? else {
            return Ok(());
        };
        let input = b"cuteafd pinned staging ffi roundtrip".to_vec();
        let mut output = vec![0_u8; input.len()];
        let mut host = library.alloc_host_buffer(input.len())?;
        assert!(!host.ptr.is_null());
        assert_eq!(host.bytes, input.len());
        assert_ne!(host.flags, CUTEAFD_HOST_BUFFER_FLAG_NONE);
        assert_ne!(host.flags & CUTEAFD_HOST_BUFFER_FLAG_PINNED, 0);
        assert_ne!(host.flags & CUTEAFD_HOST_BUFFER_FLAG_MAPPED, 0);
        unsafe {
            std::ptr::copy_nonoverlapping(input.as_ptr(), host.ptr.cast::<u8>(), input.len());
        }

        let mut device = library.alloc_device_buffer(input.len())?;
        library.copy_host_buffer_h2d(device, host, input.len())?;
        library.copy_d2h(&mut output, device)?;
        assert_eq!(output, input);

        let alias = library.cuda_host_buffer_device_alias(host)?;
        assert_eq!(alias.bytes, host.bytes);
        assert_ne!(alias.flags & CUTEAFD_DEVICE_BUFFER_FLAG_MAPPED_HOST, 0);
        unsafe {
            library.cuda_zero_bytes_async(alias, input.len(), std::ptr::null_mut())?;
            library.cuda_stream_synchronize(std::ptr::null_mut())?;
        }
        assert!(
            unsafe { std::slice::from_raw_parts(host.ptr.cast::<u8>(), input.len()) }
                .iter()
                .all(|byte| *byte == 0)
        );

        output.fill(0);
        unsafe {
            std::ptr::copy_nonoverlapping(input.as_ptr(), host.ptr.cast::<u8>(), input.len());
        }
        unsafe {
            library.copy_host_buffer_h2d_async(device, host, input.len(), std::ptr::null_mut())?;
        }
        library.copy_d2h(&mut output, device)?;
        assert_eq!(output, input);

        let pitched_source = [1_u8, 2, 3, 90, 91, 4, 5, 6, 92, 93, 7, 8, 9, 94, 95];
        let pitched_expected = [1_u8, 2, 3, 0, 4, 5, 6, 0, 7, 8, 9, 0];
        let mut pitched_host = library.alloc_host_buffer(pitched_source.len())?;
        let mut pitched_device = library.alloc_device_buffer(pitched_expected.len())?;
        library.copy_h2d(pitched_device, &[0_u8; 12])?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                pitched_source.as_ptr(),
                pitched_host.ptr.cast::<u8>(),
                pitched_source.len(),
            );
            library.copy_host_buffer_h2d_2d_async(
                pitched_device,
                4,
                pitched_host,
                5,
                3,
                3,
                std::ptr::null_mut(),
            )?;
        }
        let mut pitched_output = [0_u8; 12];
        library.copy_d2h(&mut pitched_output, pitched_device)?;
        assert_eq!(pitched_output, pitched_expected);
        library.free_device_buffer(&mut pitched_device)?;
        library.free_host_buffer(&mut pitched_host)?;

        library.free_device_buffer(&mut device)?;
        library.free_host_buffer(&mut host)?;
        assert!(host.ptr.is_null());
        assert_eq!(host.bytes, 0);
        assert_eq!(host.flags, CUTEAFD_HOST_BUFFER_FLAG_NONE);
        Ok(())
    }








































































    #[test]
    fn error_propagation() -> Result<()> {
        let Some(library) = load_test_library()? else {
            return Ok(());
        };
        let err = library.alloc_device_buffer(0).unwrap_err().to_string();
        assert!(err.contains("status 1"));
        assert!(err.contains("size is zero"));
        let err = library
            .alloc_managed_device_buffer(0)
            .unwrap_err()
            .to_string();
        assert!(err.contains("status 1"));
        assert!(err.contains("size is zero"));
        Ok(())
    }

    #[test]
    fn rdma_device_info_and_host_buffer_plan() -> Result<()> {
        let Some(library) = load_test_library()? else {
            return Ok(());
        };
        let info = library.rdma_device_info()?;
        assert!(!c_char_array_to_string(&info.first_device_name).is_empty());
        assert!(!c_char_array_to_string(&info.status).is_empty());

        let input = vec![0_u8; 12_288];
        let plan =
            library.rdma_plan_host_buffer_registration(input.as_ptr().cast(), input.len(), 4096)?;
        assert_eq!(plan.original_bytes, input.len());
        assert_eq!(plan.alignment, 4096);
        assert!(plan.registered_span_bytes >= input.len());
        assert_eq!(plan.registered_span_bytes % 4096, 0);
        assert_eq!(plan.span_aligned, 1);
        Ok(())
    }

    #[test]
    fn rdma_register_host_buffer_probe_reports_capability() -> Result<()> {
        let Some(library) = load_test_library()? else {
            return Ok(());
        };
        let mut input = vec![0_u8; 12_288];
        match library.rdma_register_host_buffer_probe(&mut input) {
            Ok(probe) => {
                assert_eq!(probe.bytes, input.len());
                assert_eq!(probe.registered, 1);
                assert!(!c_char_array_to_string(&probe.device_name).is_empty());
            }
            Err(err) => {
                let err = err.to_string();
                assert!(
                    err.contains(&format!("status {CUTEAFD_STATUS_RDMA_UNAVAILABLE}")),
                    "{err}"
                );
                assert!(err.contains("RDMA") || err.contains("rdma"), "{err}");
            }
        }
        Ok(())
    }

    #[test]
    fn rdma_create_rc_qp_probe_reports_capability() -> Result<()> {
        let Some(library) = load_test_library()? else {
            return Ok(());
        };
        match library.rdma_create_rc_qp_probe(1, 16, 16, 1) {
            Ok(probe) => {
                assert_eq!(probe.port_num, 1);
                assert_eq!(probe.requested_send_wr, 16);
                assert_eq!(probe.requested_recv_wr, 16);
                assert_eq!(probe.requested_max_sge, 1);
                assert_eq!(probe.created, 1);
                assert_ne!(probe.qp_num, 0);
                assert!(probe.actual_max_send_wr >= probe.requested_send_wr);
                assert!(probe.actual_max_recv_wr >= probe.requested_recv_wr);
                assert!(probe.actual_max_send_sge >= probe.requested_max_sge);
                assert!(probe.actual_max_recv_sge >= probe.requested_max_sge);
                assert!(!c_char_array_to_string(&probe.device_name).is_empty());
                assert!(!c_char_array_to_string(&probe.status).is_empty());
            }
            Err(err) => {
                let err = err.to_string();
                assert!(
                    err.contains(&format!("status {CUTEAFD_STATUS_RDMA_UNAVAILABLE}")),
                    "{err}"
                );
                assert!(err.contains("RDMA") || err.contains("rdma"), "{err}");
            }
        }
        Ok(())
    }

    #[test]
    fn rdma_rc_send_recv_loopback_probe_reports_capability() -> Result<()> {
        let Some(library) = load_test_library()? else {
            return Ok(());
        };
        match library.rdma_rc_send_recv_loopback_probe(1, 12_288) {
            Ok(probe) => {
                assert_eq!(probe.port_num, 1);
                assert_eq!(probe.bytes, 12_288);
                assert_eq!(probe.completed, 1);
                assert_eq!(probe.payload_matches, 1);
                assert_ne!(probe.sender_qp_num, 0);
                assert_ne!(probe.receiver_qp_num, 0);
                assert_eq!(probe.send_completions, 1);
                assert_eq!(probe.recv_completions, 1);
                assert!(probe.poll_iterations > 0);
                assert!(!c_char_array_to_string(&probe.device_name).is_empty());
                assert!(!c_char_array_to_string(&probe.status).is_empty());
            }
            Err(err) => {
                let err = err.to_string();
                assert!(
                    err.contains(&format!("status {CUTEAFD_STATUS_RDMA_UNAVAILABLE}")),
                    "{err}"
                );
                assert!(err.contains("RDMA") || err.contains("rdma"), "{err}");
            }
        }
        Ok(())
    }

    #[test]
    fn rdma_rc_protocol_v2_loopback_probe_reports_capability() -> Result<()> {
        let Some(library) = load_test_library()? else {
            return Ok(());
        };
        let request = protocol_v2_frame(1, 12_288);
        let response = protocol_v2_frame(2, 12_288);
        match library.rdma_rc_protocol_v2_loopback_probe(1, &request, &response) {
            Ok(probe) => {
                assert_eq!(probe.port_num, 1);
                assert_eq!(probe.request_bytes, request.len());
                assert_eq!(probe.response_bytes, response.len());
                assert_eq!(probe.completed, 1);
                assert_eq!(probe.request_payload_matches, 1);
                assert_eq!(probe.response_payload_matches, 1);
                assert_ne!(probe.client_qp_num, 0);
                assert_ne!(probe.server_qp_num, 0);
                assert_eq!(probe.send_completions, 2);
                assert_eq!(probe.recv_completions, 2);
                assert!(probe.poll_iterations > 0);
                assert!(!c_char_array_to_string(&probe.device_name).is_empty());
                assert!(!c_char_array_to_string(&probe.status).is_empty());
            }
            Err(err) => {
                let err = err.to_string();
                assert!(
                    err.contains(&format!("status {CUTEAFD_STATUS_RDMA_UNAVAILABLE}")),
                    "{err}"
                );
                assert!(err.contains("RDMA") || err.contains("rdma"), "{err}");
            }
        }
        Ok(())
    }
    /// Chunk-3b FFI oracle: drive K1 -> K3/K4 -> K5 through the C ABI and
    /// compare the final token against the **production** CPU sampler
    /// (`cuteafd_core::TargetSamplingParams::select_token`) on identical logits,
    /// masks, params, seeds and positions. Every device output is pre-filled
    /// with `0xDEADBEEF`, so a kernel that never ran cannot pass. Rows whose two
    /// f32 accumulation orders pick different tokens (the declared §6.3c
    /// residual) are counted and reported, not asserted away; the test asserts
    /// the device token is one of the CPU's ordered survivors and, on the
    /// deterministic rows, that it matches exactly.
    ///
    /// `#[ignore]` convention: this test needs a GPU and a built native library,
    /// so it does not run in the default `cargo test` sweep (the design §12.2
    /// pattern). Run it explicitly with the library path:
    ///
    /// ```text
    /// CUTEAFD_NATIVE_LIB=/path/to/libcuteafd_native.so \
    ///   cargo test -p cuteafd-ffi -- --ignored v41_sampler
    /// ```
    ///
    /// The loader is deliberately loud: an explicit `--ignored` run with a
    /// missing library FAILS (`load_device_test_library` errors); it never skips
    /// silently.
    #[test]
    #[ignore = "requires a GPU and a built libcuteafd_native.so; run with --ignored"]
    fn v41_sampler_device_nucleus_ordered_matches_cpu_oracle() -> Result<()> {
        let library = load_device_test_library()?;

        // ---- small deterministic rows, vocab 8 ----
        let vocab = 8_usize;
        let rows = 6_usize;
        let words = vocab.div_ceil(32);
        let logits: Vec<f32> = vec![
            2.0, 1.0, 0.5, 0.0, -1.0, -2.0, -3.0, -4.0, // row 0: strict prefix at top_p=1
            8.0, 0.0, -8.0, -16.0, -24.0, -32.0, -40.0, -48.0, // row 1: peaked, wide margins
            3.0, 3.0, 3.0, 0.0, 0.0, 0.0, -5.0, -9.0,   // row 2: tie group at the top
            6.0, 0.0, -6.0, -12.0, -18.0, -24.0, -30.0, -36.0, // row 3: peaked + top_k 4
            8.0, 5.0, 0.0, -5.0, -10.0, -15.0, -20.0, -25.0, // row 4: masked, top_p 0.5
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,     // row 5: sparse mask + top_k 3
        ];
        let mask: Vec<u32> = vec![
            u32::MAX,
            u32::MAX,
            u32::MAX,
            u32::MAX,
            u32::MAX,
            (1 << 1) | (1 << 4) | (1 << 7),
        ];
        // A deterministic, non-vacuous mask for row 4: it removes two of the
        // three top tokens, so the masked winner differs from the unmasked one.
        let mut mask = mask;
        mask[4] = (1 << 0) | (1 << 2) | (1 << 3) | (1 << 5);
        let params = vec![
            /* `top_p < 1` keeps this row on the ordered path; `top_p = 1.0` with
             * `top_k` disabled would be the disjoint K2 fast path, which K5 must
             * not touch (and does not write a diagnostic for). */
            v41_nucleus_row(0, 1.0, 0.9, 0, 0.0, CUTEAFD_V41_SAMPLER_NO_MASK_ROW, CUTEAFD_V41_SAMPLER_FLAG_NO_MASK, 1, 0),
            v41_nucleus_row(1, 0.7, 0.9, 0, 0.0, CUTEAFD_V41_SAMPLER_NO_MASK_ROW, CUTEAFD_V41_SAMPLER_FLAG_NO_MASK, 20_260_922, 7),
            v41_nucleus_row(2, 0.7, 1.0, 5, 0.0, CUTEAFD_V41_SAMPLER_NO_MASK_ROW, CUTEAFD_V41_SAMPLER_FLAG_NO_MASK, 0xDEAD_BEEF, 11),
            v41_nucleus_row(3, 0.7, 0.9, 4, 0.0, CUTEAFD_V41_SAMPLER_NO_MASK_ROW, CUTEAFD_V41_SAMPLER_FLAG_NO_MASK, 987_654_321, 0),
            v41_nucleus_row(4, 1.0, 0.5, 0, 0.0, 4, 0, 5, 3),
            v41_nucleus_row(5, 1.0, 1.0, 3, 0.0, 5, 0, 29, 2),
        ];
        let capacity = 8_usize;
        let run = run_v41_nucleus_batch(&library, &logits, rows, vocab, &params, Some(&mask), capacity)?;
        assert!(
            run.status.iter().all(|status| *status == CUTEAFD_V41_SAMPLER_STATUS_OK),
            "every valid K5 row must leave the caller-visible status at OK, got {:?}",
            run.status
        );
        let mut exact = 0_usize;
        let mut divergences = Vec::new();
        for row in 0..rows {
            let row_logits = &logits[row * vocab..(row + 1) * vocab];
            let row_mask = if params[row].flags & CUTEAFD_V41_SAMPLER_FLAG_NO_MASK != 0 {
                None
            } else {
                Some(&mask[row * words..(row + 1) * words])
            };
            assert_ne!(
                run.ids[row], 0xDEAD_BEEF,
                "row {row} out_indices was written by K5"
            );
            let oracle = cuteafd_core::TargetSamplingParams::new(
                params[row].temperature,
                params[row].top_p,
                if params[row].top_k == 0 { None } else { Some(params[row].top_k as usize) },
                params[row].min_p,
                params[row].seed,
            )
            .expect("oracle params")
            .select_token(row_logits, row_mask, params[row].position)
            .expect("oracle selection") as u32;
            assert_ne!(run.total[row], -123.0, "row {row} out_total was written");
            assert_ne!(run.nucleus_count[row], 0xDEAD_BEEF, "row {row} nucleus written");
            if run.ids[row] == oracle {
                exact += 1;
            } else {
                // The declared accumulation-order residual, not a filter bug:
                // the device token must still be one of the CPU's ordered
                // survivors.
                let ordered = ordered_ids(row_logits, params[row].temperature, params[row].top_k, row_mask);
                assert!(
                    ordered.contains(&run.ids[row]),
                    "row {row} divergent token {} is not an ordered survivor",
                    run.ids[row]
                );
                divergences.push((row, run.ids[row], oracle));
            }
        }
        assert_eq!(exact, rows, "deterministic small rows must match exactly; divergences {divergences:?}");

        // ---- tie group with an exact-k cut: the nucleus must stay inside the
        // retained list ----
        let tie_vocab = 256_usize;
        let mut tie_logits = vec![0.0f32; tie_vocab];
        for token in 3..tie_vocab {
            tie_logits[token] = -1.0;
        }
        tie_logits[0] = 4.0;
        tie_logits[1] = 3.0;
        tie_logits[2] = 2.0;
        let tie_params = vec![v41_nucleus_row(
            0, 0.7, 0.9, 40, 0.0, CUTEAFD_V41_SAMPLER_NO_MASK_ROW, CUTEAFD_V41_SAMPLER_FLAG_NO_MASK, 7, 5,
        )];
        let tie = run_v41_nucleus_batch(
            &library,
            &tie_logits,
            1,
            tie_vocab,
            &tie_params,
            None,
            40,
        )?;
        let tie_oracle = cuteafd_core::TargetSamplingParams::new(0.7, 0.9, Some(40), 0.0, 7)
            .expect("tie params")
            .select_token(&tie_logits, None, 5)
            .expect("tie oracle") as u32;
        assert_ne!(tie.ids[0], 0xDEAD_BEEF, "tie row wrote a token");
        assert!(tie.nucleus_count[0] >= 1 && tie.nucleus_count[0] <= 40, "tie nucleus in range");
        assert!(
            tie.ranked[..40].contains(&tie.ids[0]),
            "tie token {} is inside the retained list",
            tie.ids[0]
        );
        assert_eq!(
            tie.ids[0], tie_oracle,
            "tie row matches the production sampler exactly"
        );
        Ok(())
    }

    /// One `CuteafdV41SamplerRow` with an explicit `top_p` (the chunk-3a helper
    /// fixes it at 1.0).
    #[allow(clippy::too_many_arguments)]
    /// The loud per-row failure path through the FFI: `top_k = 300` with
    /// `survivor_count = 400` is a retained list wider than `kBlock` (256), so
    /// K5 cannot sample it. Before the fix the entry point returned OK and left
    /// `out_indices` at the caller's sentinel, so a caller reading only the
    /// returned status believed success. Now K5 writes `INTERNAL` to the same
    /// caller-visible per-row status K1 uses.
    ///
    /// `#[ignore]` convention: needs a GPU and a built native library.
    #[test]
    #[ignore = "requires a GPU and a built libcuteafd_native.so; run with --ignored"]
    fn v41_sampler_device_nucleus_reports_internal_for_unsupported_rows() -> Result<()> {
        let library = load_device_test_library()?;
        let vocab = 400_usize;
        let logits: Vec<f32> = (0..vocab).map(|token| -0.01 * token as f32).collect();
        let params = vec![v41_nucleus_row(
            0,
            0.7,
            0.9,
            300,
            0.0,
            CUTEAFD_V41_SAMPLER_NO_MASK_ROW,
            CUTEAFD_V41_SAMPLER_FLAG_NO_MASK,
            5,
            1,
        )];
        let run = run_v41_nucleus_batch(&library, &logits, 1, vocab, &params, None, 300)?;
        assert_eq!(
            run.status[0], CUTEAFD_V41_SAMPLER_STATUS_INTERNAL,
            "a retained list wider than kBlock must reach out_status"
        );
        assert_eq!(
            run.ids[0], 0xDEAD_BEEF,
            "the failed row must leave out_indices unwritten"
        );
        Ok(())
    }

    fn v41_nucleus_row(
        output_row: u32,
        temperature: f32,
        top_p: f32,
        top_k: u32,
        min_p: f32,
        mask_row: u32,
        flags: u32,
        seed: u64,
        position: u64,
    ) -> CuteafdV41SamplerRow {
        CuteafdV41SamplerRow {
            seed,
            position,
            temperature,
            top_p,
            min_p,
            top_k,
            mask_row,
            flags: flags | CUTEAFD_V41_SAMPLER_FLAG_DIAGNOSE,
            output_row,
            ln_min_p: if min_p > 0.0 { min_p.ln() } else { f32::NEG_INFINITY },
            ..CuteafdV41SamplerRow::default()
        }
    }

    /// The CPU's ordered survivor ids (scaled descending, id ascending), used
    /// only to bound a residual divergence.
    fn ordered_ids(logits: &[f32], temperature: f32, top_k: u32, mask: Option<&[u32]>) -> Vec<u32> {
        let inv = 1.0f32 / temperature;
        let mut ranked: Vec<(f32, u32)> = logits
            .iter()
            .enumerate()
            .filter(|(token, _)| {
                mask.is_none_or(|words| words[token / 32] & (1 << (token % 32)) != 0)
            })
            .map(|(token, &value)| (value * inv, token as u32))
            .collect();
        ranked.sort_unstable_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.1.cmp(&b.1))
        });
        let take = if top_k == 0 || top_k as usize >= ranked.len() {
            ranked.len()
        } else {
            top_k as usize
        };
        ranked.into_iter().take(take).map(|(_, id)| id).collect()
    }

    /// Outputs of one K1 + K3/K4 + K5 batch through the C ABI. `ranked` is
    /// `rows * capacity`; the diagnostic outputs are parsed from the K5 stage.
    struct NucleusRun {
        ranked: Vec<u32>,
        ids: Vec<u32>,
        status: Vec<u32>,
        total: Vec<f32>,
        nucleus_count: Vec<u32>,
    }

    /// Drive K1 then K3/K4 then K5 through the FFI. Every K5 output is
    /// pre-filled with `0xDEADBEEF` / `-123.0f`, so a kernel that never ran
    /// cannot satisfy an equality assertion with stale memory.
    #[allow(clippy::too_many_arguments)]
    fn run_v41_nucleus_batch(
        library: &NativeLibrary,
        logits: &[f32],
        rows: usize,
        vocab: usize,
        params: &[CuteafdV41SamplerRow],
        mask: Option<&[u32]>,
        capacity: usize,
    ) -> Result<NucleusRun> {
        assert_eq!(params.len(), rows);
        assert_eq!(logits.len(), rows * vocab);
        let words = vocab.div_ceil(32);
        if let Some(mask) = mask {
            assert_eq!(mask.len(), rows * words);
        }
        let logits_buffer = library.alloc_device_buffer(rows * vocab * 4)?;
        let params_buffer = library.alloc_device_buffer(rows * CUTEAFD_V41_SAMPLER_PARAM_BYTES)?;
        let mask_buffer = match mask {
            Some(_) => Some(library.alloc_device_buffer(rows * words * 4)?),
            None => None,
        };
        let ids = library.alloc_device_buffer(rows * 4)?;
        let status = library.alloc_device_buffer(rows * 4)?;
        let detail = library.alloc_device_buffer(rows * 4)?;
        let scores = library.alloc_device_buffer(rows * 4)?;
        let scratch = library.alloc_device_buffer(rows * CUTEAFD_V41_SAMPLER_SCRATCH_BYTES)?;
        let retained = library.alloc_device_buffer(rows * 4)?;
        let passes = library.alloc_device_buffer(rows * 4)?;
        let rank_ids = match capacity {
            0 => None,
            _ => Some(library.alloc_device_buffer(rows * capacity * 4)?),
        };
        let rank_scratch = match capacity {
            0 => None,
            _ => Some(library.alloc_device_buffer(rows * capacity * 8)?),
        };
        let out_total = library.alloc_device_buffer(rows * 4)?;
        let out_nucleus = library.alloc_device_buffer(rows * 4)?;

        let result = (|| -> Result<NucleusRun> {
            let sentinel: Vec<u8> = (0..rows).flat_map(|_| 0xDEAD_BEEFu32.to_ne_bytes()).collect();
            library.copy_h2d(retained, &sentinel)?;
            library.copy_h2d(passes, &sentinel)?;
            library.copy_h2d(ids, &sentinel)?;
            library.copy_h2d(out_nucleus, &sentinel)?;
            let negative: Vec<u8> = (0..rows).flat_map(|_| (-123.0f32).to_ne_bytes()).collect();
            library.copy_h2d(out_total, &negative)?;
            if let Some(rank_ids) = rank_ids {
                let arena: Vec<u8> =
                    (0..rows * capacity).flat_map(|_| 0xDEAD_BEEFu32.to_ne_bytes()).collect();
                library.copy_h2d(rank_ids, &arena)?;
            }
            let mut logits_bytes = Vec::with_capacity(logits.len() * 4);
            for value in logits {
                logits_bytes.extend_from_slice(&value.to_ne_bytes());
            }
            library.copy_h2d(logits_buffer, &logits_bytes)?;
            if let (Some(mask), Some(mask_buffer)) = (mask, mask_buffer) {
                let mut mask_bytes = Vec::with_capacity(mask.len() * 4);
                for value in mask {
                    mask_bytes.extend_from_slice(&value.to_ne_bytes());
                }
                library.copy_h2d(mask_buffer, &mask_bytes)?;
            }
            let param_bytes: Vec<u8> = unsafe {
                std::slice::from_raw_parts(
                    params.as_ptr().cast::<u8>(),
                    params.len() * CUTEAFD_V41_SAMPLER_PARAM_BYTES,
                )
                .to_vec()
            };
            library.copy_h2d(params_buffer, &param_bytes)?;
            library.cuda_v41_target_sample(
                logits_buffer,
                rows,
                vocab,
                vocab,
                params,
                params_buffer,
                mask_buffer,
                if mask_buffer.is_some() { words } else { 0 },
                ids,
                status,
                detail,
                scores,
                None,
                None,
                scratch,
            )?;
            library.cuda_v41_topk_select(
                logits_buffer,
                rows,
                vocab,
                vocab,
                params,
                params_buffer,
                mask_buffer,
                if mask_buffer.is_some() { words } else { 0 },
                rank_ids,
                rank_scratch,
                capacity,
                retained,
                passes,
                scratch,
            )?;
            library.cuda_v41_nucleus(
                logits_buffer,
                rows,
                vocab,
                vocab,
                params,
                params_buffer,
                mask_buffer,
                if mask_buffer.is_some() { words } else { 0 },
                rank_ids,
                capacity,
                retained,
                ids,
                status,
                Some(out_total),
                Some(out_nucleus),
                scratch,
            )?;
            let read_u32 = |buffer: CuteafdDeviceBuffer, count: usize| -> Result<Vec<u32>> {
                let mut bytes = vec![0_u8; count * 4];
                library.copy_d2h(&mut bytes, buffer)?;
                Ok((0..count)
                    .map(|index| {
                        u32::from_ne_bytes(bytes[index * 4..index * 4 + 4].try_into().unwrap())
                    })
                    .collect())
            };
            let read_f32 = |buffer: CuteafdDeviceBuffer, count: usize| -> Result<Vec<f32>> {
                let mut bytes = vec![0_u8; count * 4];
                library.copy_d2h(&mut bytes, buffer)?;
                Ok((0..count)
                    .map(|index| {
                        f32::from_ne_bytes(bytes[index * 4..index * 4 + 4].try_into().unwrap())
                    })
                    .collect())
            };
            Ok(NucleusRun {
                ranked: match rank_ids {
                    Some(rank_ids) => read_u32(rank_ids, rows * capacity)?,
                    None => Vec::new(),
                },
                ids: read_u32(ids, rows)?,
                status: read_u32(status, rows)?,
                total: read_f32(out_total, rows)?,
                nucleus_count: read_u32(out_nucleus, rows)?,
            })
        })();

        let mut logits_buffer = logits_buffer;
        let mut params_buffer = params_buffer;
        let mut mask_buffer = mask_buffer;
        let mut ids = ids;
        let mut status = status;
        let mut detail = detail;
        let mut scores = scores;
        let mut scratch = scratch;
        let mut retained = retained;
        let mut passes = passes;
        let mut rank_ids = rank_ids;
        let mut rank_scratch = rank_scratch;
        let mut out_total = out_total;
        let mut out_nucleus = out_nucleus;
        library.free_device_buffer(&mut logits_buffer)?;
        library.free_device_buffer(&mut params_buffer)?;
        if let Some(mut buffer) = mask_buffer.take() {
            library.free_device_buffer(&mut buffer)?;
        }
        library.free_device_buffer(&mut ids)?;
        library.free_device_buffer(&mut status)?;
        library.free_device_buffer(&mut detail)?;
        library.free_device_buffer(&mut scores)?;
        library.free_device_buffer(&mut scratch)?;
        library.free_device_buffer(&mut retained)?;
        library.free_device_buffer(&mut passes)?;
        if let Some(mut buffer) = rank_ids.take() {
            library.free_device_buffer(&mut buffer)?;
        }
        if let Some(mut buffer) = rank_scratch.take() {
            library.free_device_buffer(&mut buffer)?;
        }
        library.free_device_buffer(&mut out_total)?;
        library.free_device_buffer(&mut out_nucleus)?;
        result
    }
}
