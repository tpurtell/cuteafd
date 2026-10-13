// Replicated-group Spark TP2 shard modules (native FP8 K32 family, SM121).
// Distinct symbol family from the standard Spark TP4 shard so a single
// libcuteafd_native.so can hold every Spark TP degree. Geometry is baked into
// the exported variant table (intermediate 1152, no storage padding).
#define CUTEAFD_V41_SPARK_TP2_EXPERTS 1
#define cuteafd_expert_info cuteafd_v41_spark_tp2_expert_info
#define cuteafd_expert_initialize cuteafd_v41_spark_tp2_expert_initialize
#define cuteafd_expert_output_kind cuteafd_v41_spark_tp2_expert_output_kind
#define cuteafd_expert_group_rows cuteafd_v41_spark_tp2_expert_group_rows
#define cuteafd_expert_bind_scratch cuteafd_v41_spark_tp2_expert_bind_scratch
#define cuteafd_expert_initialize_scratch_async cuteafd_v41_spark_tp2_expert_initialize_scratch_async
#define cuteafd_expert_launch cuteafd_v41_spark_tp2_expert_launch
#include "v41_experts.cc"
