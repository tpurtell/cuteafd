pub mod families;
pub mod media;
pub mod formats;
pub mod serving_capacity;
pub use families::deepseek_v41::engram_pipeline::{EngramPipeline, EngramRequestTokens, EngramWave};
pub use families::deepseek_v41::engram_gather::{
    EngramGatherer, EngramGatherLease, EngramGatherPoll, EngramGatherTicket, EngramGatherTiming,
};
pub use families::deepseek_v41::engram_staging::{EngramBatchStaging, EngramGatherView};
pub use families::deepseek_v41::v41_expert_staging::{V41ExpertSelection, V41ExpertStaging};
pub use families::deepseek_v41::v41_catalog::{
    read_expert_catalog, read_official_v41_catalog, OfficialV41Catalog, RoutedExpertShape,
    V41StorageBudget, V41Tensor, V41TensorPlacement, V41CoordinatorTensorReader,
};
pub use families::deepseek_v41::v41_exl3_residency::{V41Exl3Layer, V41Exl3Load, V41Exl3Residency, V41Exl3ResidentBuffer};
pub use families::deepseek_v41::v41_exl3_staging::V41Exl3TensorSlice;
pub use families::deepseek_v41::v41_exl3::{read_v41_exl3_manifest, V41Exl3Manifest, V41Exl3Naming, V41Exl3Projection,
    V41Exl3ProjectionKind, V41Exl3Partition, V41_EXL3_SCHEMA};
pub use families::deepseek_v41::v41_nvfp4::{
    is_v41_nvfp4_publication, read_v41_nvfp4_contract, V41Nvfp4Contract, V41Nvfp4ExpertLayout,
};
pub use families::deepseek_v41::v41_nvfp4_staging::{V41Nvfp4Staging, V41_NVFP4_STAGING_SLOTS};
pub use families::deepseek_v41::v41_image::{V41Image, V41ImageGrid, V41ImageSpan, V41VisionPrompt, V41ImageTokenType,
    V41_IMAGE_TOKEN_ID, V41_MAX_IMAGES};
pub use families::deepseek_v41::v41_config::{
    read_official_v41_config, OfficialV41Config, V41QuantizationConfig, V41RopeScaling,
    V41TextConfig, V41VisionConfig, OFFICIAL_V41_MODEL_ID, OFFICIAL_V41_REVISION,
};
pub use families::deepseek_v41::engram_tokenizer::EngramTokenMap;
pub use families::deepseek_v41::engram_prefetch::{EngramEncoding, EngramPrefetcher, EngramTable, PrefetchOutcome, PrefetchTicket};
pub use formats::mapped_table::{
    AdviseRows, GatherFailure, GatherLease, GatherPool, GatherPoll, GatherReport, GatherTicket, GatherTiming,
    GatherWorker, HotRowCache, MappedRows, PendingGather, MappedTable, MappedTableError, RowFormat, TablePart, TablePrefetchOutcome,
    TablePrefetchTicket, TablePrefetcher, MappedTableStatsReader, TableStats, TableStatsSnapshot, TableBackend, mapped_table_stats, mapped_table_stats_with_intervals,
};
mod catalog;
mod snapshot;
mod tensors;
mod tokenizer;
pub mod plan;
pub mod page_cache;

pub use formats::attention_format::{
    native_deepseek_v4_attention_tensor_specs, validate_native_deepseek_v4_attention_catalog,
    NativeDeepseekV4AttentionCatalogSummary, NativeDeepseekV4AttentionTensorFamily,
    NativeDeepseekV4AttentionTensorSpec, NATIVE_ATTENTION_FP8_BLOCK,
};
pub use catalog::{
    build_catalog, build_catalog_for_snapshot, classification_summary_markdown, read_model_facts,
    read_safetensors_metadata, SafetensorsTensorMetadata,
};
pub use formats::dspark_format::{
    native_deepseek_v4_dspark_tensor_specs, validate_native_deepseek_v4_dspark_catalog,
    NativeDeepseekV4DsparkCatalogSummary, NativeDeepseekV4DsparkTensorSpec,
};
pub use formats::exl3_format::{
    exl3_expert, exl3_expert_trellis_bits, exl3_projection_trellis_bits,
    exl3_trellis_bits_for_recipe, is_deepseek_v4_exl3_recipe, is_deepseek_v4_mixed_exl3_recipe,
    validate_exl3_expert_catalog, Exl3CatalogSummary, Exl3Expert, Exl3Projection,
    Exl3ProjectionKind, Exl3Tp4ResidentGeometry, DEEPSEEK_V4_EXL3_CODEBOOK,
    DEEPSEEK_V4_EXL3_RECIPE, DEEPSEEK_V4_EXL3_RECIPE_K3_V4, DEEPSEEK_V4_EXL3_RECIPE_MIXED_K2_K3_V1,
    DEEPSEEK_V4_EXL3_RECIPE_V2, DEEPSEEK_V4_EXL3_RECIPE_V3, DEEPSEEK_V4_EXL3_RECIPE_V4,
    DEEPSEEK_V4_EXL3_SCHEMA, DEEPSEEK_V4_EXL3_SCHEMA_VERSION, DEEPSEEK_V4_EXL3_SOURCE_FORMAT,
    DEEPSEEK_V4_EXL3_T12_LUT_BYTES, DEEPSEEK_V4_EXL3_TENSOR_FORMAT, DEEPSEEK_V4_EXL3_TRELLIS_BITS,
    EXLLAMAV3_REPOSITORY, EXLLAMAV3_REVISION, EXLLAMAV3_SOURCE_TREE_SHA256,
};
pub use formats::expert_format::{
    native_fp4_expert, validate_native_fp4_expert_catalog, NativeFp4CatalogSummary,
    NativeFp4Expert, NativeFp4Projection, NativeFp4ProjectionKind, NativeFp4TpExpertShard,
    NativeFp4TpProjectionShard, NativeFp4TpTensorWindow, NATIVE_FP4_K_BLOCK,
};
pub use snapshot::{
    default_hf_home, empty_catalog_for_snapshot, model_cache_dir, resolve_snapshot,
    resolve_snapshot_at_revision, SnapshotResolution,
};
pub use tensors::{
    dtype_byte_width, load_tensor_bytes, load_tensor_bytes_with_options, load_tensor_rows,
    load_tensor_rows_with_options, read_tensor_bytes_into, read_tensor_bytes_into_with_options,
    read_tensor_row_prefix_into, read_tensor_row_prefix_into_with_options,
    read_tensor_row_window_into, read_tensor_row_window_into_with_options, read_tensor_rows_into,
    read_tensor_rows_into_with_options, LoadedTensor, LoadedTensorRows, LoadedTensorRowsSummary,
    LoadedTensorSummary, TensorLoadOptions,
};
pub use tokenizer::{
    decode_tokenizer_ids, encode_tokenizer_text, streaming_token_decoder, LoadedTokenizer,
    StreamingTokenDecoder, TokenizerDecodeSummary, TokenizerEncodingSummary,
};

#[cfg(test)]
mod tests;
