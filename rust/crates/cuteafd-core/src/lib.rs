pub mod coordinator_programs;
pub mod expert_geometry;
pub use expert_geometry::{expert_geometry, set_expert_geometry, ExpertGeometry};
mod draft_policy;
pub use draft_policy::{
    draft_expected_tokens, DraftCandidate, DraftCostSnapshot, DraftPolicy, DraftPolicyError, DraftPolicyStats,
    DraftRoundObservation, DraftSelection, LayerResource, ObservedDraftRequest, PolicyGeometry, ResourceClass,
    MAX_RESOURCE_CLASSES,
};
mod dspark_rng;
pub use dspark_rng::{DsparkRng, DsparkRngReservation};
mod target_sampling;
pub use target_sampling::{
    TargetSamplingError, TargetSamplingParams, GREEDY_TEMPERATURE_EPS, MAX_TARGET_TEMPERATURE,
};
mod dspark_verify;
pub use dspark_verify::{verify_dspark_greedy, GreedyVerification, MAX_DSPARK_PROPOSALS};
mod engram;
pub mod prefix;
pub mod serving_capacity;
pub mod memory_layout;
pub mod media;
pub use media::{AudioKey, ImageKey, MediaKey, MediaSpan};
pub use engram::{EngramBatch, EngramError, EngramHashes, EngramHistory, EngramPrefillCursor, ENGRAM_LAYERS, ENGRAM_ROWS, ENGRAM_COMPRESSED_VOCAB};
mod attention_geometry;
mod constants;
mod debug_expert;
mod errors;
mod expert_batch;
mod expert_host_batch;
mod expert_route_plan;
mod exl3_tp4_ownership;
pub use exl3_tp4_ownership::{
    exl3_tp4_active_blocks, Exl3BoundaryCost, Exl3Tp4OwnershipPlan,
    Exl3Tp4OwnershipPlanner, EXL3_TP4_RESIDENT_BLOCKS,
};
mod ids;
mod kv_cache;
mod layerwave;
mod model;
mod node;
mod placement;
mod replicated_expert_schedule;
mod transport_metrics;

pub use attention_geometry::{
    DeepseekV4AttentionGeometry, DeepseekV4AttentionLayerPlan, DeepseekV4AttentionLayerSource,
    DeepseekV4AttentionPlan, DeepseekV4CompressedSelection,
};
pub use constants::{
    COORDINATOR_HOST, DEFAULT_MODEL_ID, DS4_COMPRESS_ROPE_THETA, DS4_DSPARK_BLOCK_SIZE,
    DS4_DSPARK_NOISE_TOKEN_ID, DS4_EXPERT_TP_WORLD_SIZE, DS4_FLASH_COMPRESS_RATIOS,
    DS4_FLASH_DSPARK_MARKOV_RANK, DS4_FLASH_HIDDEN_BF16_BYTES, DS4_FLASH_HIDDEN_SIZE,
    DS4_FLASH_MODEL_ID, DS4_FLASH_MOE_INTERMEDIATE_SIZE, DS4_FLASH_NUM_HIDDEN_LAYERS,
    DS4_FLASH_Q_LORA_RANK, DS4_FLASH_ROUTED_EXPERTS, DS4_FLASH_TOP_K, DS4_HC_EPS, DS4_HC_MULT,
    DS4_HC_SINKHORN_ITERS, DS4_HEAD_DIM, DS4_INDEX_HEADS, DS4_INDEX_HEAD_DIM, DS4_KV_HEADS,
    DS4_NUM_HASH_LAYERS, DS4_NUM_SHARED_EXPERTS, DS4_ORIGINAL_MAX_POSITION_EMBEDDINGS,
    DS4_O_LORA_RANK, DS4_PRO_COMPRESS_RATIOS, DS4_PRO_DSPARK_MARKOV_RANK,
    DS4_PRO_HIDDEN_BF16_BYTES, DS4_PRO_HIDDEN_SIZE, DS4_PRO_MOE_INTERMEDIATE_SIZE,
    DS4_PRO_NUM_HIDDEN_LAYERS, DS4_PRO_PREVIEW_MODEL_ID, DS4_PRO_Q_LORA_RANK,
    DS4_PRO_ROUTED_EXPERTS, DS4_PRO_TOP_K, DS4_QK_ROPE_HEAD_DIM, DS4_ROPE_BETA_FAST,
    DS4_ROPE_BETA_SLOW, DS4_ROPE_SCALING_FACTOR, DS4_ROPE_THETA, DS4_SLIDING_WINDOW, EXPERT_HOSTS,
    GLM52_COMPRESSED_DSA_BF16_BYTES_PER_TOKEN, GLM52_COMPRESSED_KV_BF16_BYTES_PER_TOKEN,
    GLM52_COMPRESSED_MAIN_MLA_BF16_BYTES_PER_TOKEN, GLM52_DSA_INDEXER_LAYERS,
    GLM52_DSA_INDEXER_LAYER_IDS, GLM52_DSA_INDEXER_LAYER_IDS_WITH_MTP, GLM52_DSA_INDEX_HEAD_DIM,
    GLM52_EXPANDED_DEBUG_KV_BF16_BYTES_PER_TOKEN, GLM52_FIRST_K_DENSE_REPLACE,
    GLM52_HIDDEN_BF16_BYTES, GLM52_HIDDEN_SIZE, GLM52_MLA_FP8_DS_BYTES_PER_TOKEN,
    GLM52_MLA_FP8_DS_SCALE_BYTES_PER_TOKEN, GLM52_MLA_KV_LORA_RANK, GLM52_MLA_MXFP4_BLOCK_SIZE,
    GLM52_MLA_MXFP4_CODE_BYTES_PER_TOKEN, GLM52_MLA_MXFP4_DS_BYTES_PER_TOKEN,
    GLM52_MLA_MXFP4_PADDING_BYTES_PER_TOKEN, GLM52_MLA_MXFP4_SCALE_BYTES_PER_TOKEN,
    GLM52_MLA_QK_ROPE_HEAD_DIM, GLM52_MLA_ROPE_THETA, GLM52_MTP_LAYER_ID, GLM52_NUM_HIDDEN_LAYERS,
    GLM52_NUM_MTP_LAYERS, GLM52_ROUTED_EXPERTS, GLM52_ROUTED_SCALING_FACTOR, GLM52_TOP_K,
    GLM52_TOTAL_LAYERS_WITH_MTP, SUPPORTED_MODEL_IDS,
};
pub use debug_expert::{
    ExpertRequest, ExpertRequestHeader, ExpertResponse, ExpertResponseHeader, ExpertRow,
    ExpertWaveMetadata, RouteEntry,
};
pub use errors::CuteafdError;
pub use expert_batch::{ExpertBatch, ExpertBatchRow};
pub use expert_host_batch::{
    ExpertBatchRoute, ExpertHostBatch, ExpertHostBatchRow, ExpertHostBatchSet,
    ExpertHostBatchSetAccumulation, HostRowToGlobalRowMap, PartialReconstructionPlan,
};
pub use expert_route_plan::{
    plan_completion_first_routes, plan_rolling_expert_row_packs, CompletionFirstRouteGroup,
    CompletionFirstRoutePlan, CompletionRoutePlanEntry, RollingExpertRowPackAccumulator,
    RollingExpertRowPackConfig, RollingExpertRowPackEmission, RollingExpertRowPackPlan,
};
pub use ids::{LayerId, PlacementVersion, PositionId, Priority, RequestId};
pub use kv_cache::{
    KvBackedBlock, KvCacheAllocator, KvCacheBackingStore, KvCacheConfig, KvCacheDType,
    KvCacheSnapshot, KvLayout, KvReservation, KvReservationState, KvWriteRecord, KvWriteState,
    MlaKvCacheRepresentation,
};
pub use layerwave::{
    admit_layerwaves_for_iteration, plan_prefill_chunks, plan_prefill_chunks_with_model,
    DecodeStep, GraphBucket, HiddenShape, KvBlockDescriptor, LayerWave, LayerWaveAdmission,
    LayerWaveMode, MtpVerifyBlock, PrefillChunk, PrefillChunkPolicy, RouteMetadataPlaceholder,
    RowSource, RowSourceKind,
};
pub use model::{
    AttentionKind, DType, ModelFacts, ModelVariant, TensorCatalog, TensorInfo, TensorRole,
};
pub use node::NodeRole;
pub use placement::{
    owner_for_expert, ExpertOwnerLookup, LoadPlan, PlacementPolicy, TensorAssignment,
};
pub use replicated_expert_schedule::{
    replicated_expert_tie_seed, replicated_expert_tie_seed_for, ReplicatedExpertCostModel,
    ReplicatedExpertGroupId, ReplicatedExpertGroupLoad, ReplicatedExpertGroupPlan,
    ReplicatedExpertScheduleConfig, ReplicatedExpertScheduler, ReplicatedExpertTieSeedMode,
    INACTIVE_REPLICATED_EXPERT_GROUP, MAX_REPLICATED_EXPERT_GROUPS, TIE_SEED_FIXED_REQUEST_ID,
};
pub use transport_metrics::{
    TransportCapabilities, TransportPrefillBandwidthMeasurement, TransportRttMeasurement,
};

#[cfg(test)]
mod tests;
