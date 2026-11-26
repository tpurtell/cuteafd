//! DeepSeek V4.1 coordinator kernels.

pub(crate) mod v41_attention_ops;
pub(crate) mod v41_candidate_blocks;
pub(crate) mod v41_compressor;
pub(crate) mod v41_dfly2;
pub(crate) mod v41_dspark;
pub(crate) mod v41_dspark_attention;
pub(crate) mod v41_dspark_cache;
pub(crate) mod v41_fp8;
pub(crate) mod v41_fp8_plan;
pub(crate) mod v41_grouped_output;
pub(crate) mod v41_hc;
pub(crate) mod v41_index_scores;
pub(crate) mod v41_index_topk;
pub(crate) mod v41_kv;
pub(crate) mod v41_sparse_attention;
pub(crate) mod v41_vision;
