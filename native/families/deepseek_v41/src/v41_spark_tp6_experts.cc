// Pure TP6 Spark shard modules (native FP8 K32 family, SM121).
// Distinct symbol family from the standard Spark TP4 and replicated-group
// TP2/TP3 shards so a single libcuteafd_native.so can hold every Spark TP
// degree. Geometry is baked into the exported variant table: the official
// intermediate 2304 splits evenly across six ranks, so the local shard is 384
// with no storage padding (kernel_intermediate == intermediate == 384).
#define CUTEAFD_V41_SPARK_TP6_EXPERTS 1
#define cuteafd_expert_info cuteafd_v41_spark_tp6_expert_info
#define cuteafd_expert_initialize cuteafd_v41_spark_tp6_expert_initialize
#define cuteafd_expert_output_kind cuteafd_v41_spark_tp6_expert_output_kind
#define cuteafd_expert_group_rows cuteafd_v41_spark_tp6_expert_group_rows
#define cuteafd_expert_bind_scratch cuteafd_v41_spark_tp6_expert_bind_scratch
#define cuteafd_expert_initialize_scratch_async cuteafd_v41_spark_tp6_expert_initialize_scratch_async
#define cuteafd_expert_launch cuteafd_v41_spark_tp6_expert_launch
#include "v41_experts.cc"
