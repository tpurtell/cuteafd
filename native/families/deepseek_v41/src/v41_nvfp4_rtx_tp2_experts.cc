// NVFP4 (W4A4) TP2 modules. This family is independent of the W4A8 and EXL3
// variants: it consumes BF16 hidden rows, publishes BF16 token-major route planes
// and keeps its own generated bridge, so it exports distinct symbols.
#define CUTEAFD_V41_LOCAL_EXPERTS 1
#define CUTEAFD_V41_TP2_EXPERTS 1
#define CUTEAFD_V41_NVFP4_VARIANTS_HEADER "v41_nvfp4_rtx_tp2_variants.h"
#define cuteafd_expert_shared_input cuteafd_v41_nvfp4_tp2_expert_shared_input
#define cuteafd_expert_info cuteafd_v41_nvfp4_tp2_expert_info
#define cuteafd_expert_initialize cuteafd_v41_nvfp4_tp2_expert_initialize
#define cuteafd_expert_output_kind cuteafd_v41_nvfp4_tp2_expert_output_kind
#define cuteafd_expert_bind_scratch cuteafd_v41_nvfp4_tp2_expert_bind_scratch
#define cuteafd_expert_initialize_scratch_async cuteafd_v41_nvfp4_tp2_expert_initialize_scratch_async
#define cuteafd_expert_launch cuteafd_v41_nvfp4_tp2_expert_launch
#include "v41_experts.cc"
