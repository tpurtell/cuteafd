cmake_minimum_required(VERSION 3.24)
project(cuteafd_native LANGUAGES C CXX)

# Single-config serving builds should optimize the native host path too.
# Respect an explicitly selected Debug/RelWithDebInfo or multi-config generator.
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Native build configuration" FORCE)
endif()

option(CUTEAFD_ENABLE_CUDA "Build CUDA kernels and CUDA-backed ABI functions" OFF)
option(CUTEAFD_ENABLE_VISION_ATTENTION_AOT "Build shared Qwen/GLM BF16 vision attention" OFF)
option(CUTEAFD_ENABLE_AUDIO_AOT "Build qualification-only FP32 MiMo audio tower" OFF)
option(CUTEAFD_ENABLE_V41_LOCAL_EXPERT_AOT "Build full RTX backbone expert kernels alongside dSpark" OFF)
option(CUTEAFD_ENABLE_V41_TP2_EXPERT_AOT "Build dual RTX TP2 encoder expert kernels" OFF)
option(CUTEAFD_V41_TP2_COMPACT_EXPERIMENT "Use experimental native compact TP2 experts at capacities 1/16" OFF)
option(CUTEAFD_V41_SPARK_COMPACT_EXPERIMENT "Use experimental compact Spark experts through two live rows" OFF)
# Opt-in replicated-group Spark TP2/TP3 expert shards. Empty (the default)
# compiles/links no new object or symbol, so every release default behaves
# exactly as the historical Spark TP4 shard. Set to a semicolon list of
# `tp2`/`tp3` on the SM121 expert build to pre-compile those AOT capacity
# variants ahead of runtime. (The exporter always adds an additive
# `spark_tp_degree` manifest field, so byte identity of exported JSON is not
# claimed either way.)
set(CUTEAFD_SPARK_TP_ROLES "" CACHE STRING
  "Extra replicated-group Spark expert TP roles to build (tp2;tp3;tp6); empty builds only TP4")
# Deprecated spelling, accepted for one release.
if(DEFINED CUTEAFD_V41_SPARK_TP_ROLES AND NOT CUTEAFD_V41_SPARK_TP_ROLES STREQUAL "")
  message(DEPRECATION "CUTEAFD_V41_SPARK_TP_ROLES is deprecated; use CUTEAFD_SPARK_TP_ROLES")
  if(CUTEAFD_SPARK_TP_ROLES STREQUAL "")
    set(CUTEAFD_SPARK_TP_ROLES "${CUTEAFD_V41_SPARK_TP_ROLES}")
  endif()
endif()
set(CUTEAFD_EXPERT_FAMILIES "" CACHE STRING
  "Extra SM121 routed-expert kernel families as FAMILY:ROLE (e.g. dsv4f:spark)")
option(CUTEAFD_V41_VOCAB_ROW_EXPERIMENT "Use experimental single-row FP32 vocabulary projection" OFF)
option(CUTEAFD_ENABLE_EXL3_PACKAGES "Build loadable mixed EXL3 expert packages" OFF)
# Deprecated spelling, accepted for one release.
if(CUTEAFD_ENABLE_V41_EXL3_AOT)
  message(DEPRECATION "CUTEAFD_ENABLE_V41_EXL3_AOT is deprecated; use CUTEAFD_ENABLE_EXL3_PACKAGES")
  set(CUTEAFD_ENABLE_EXL3_PACKAGES ON)
endif()
option(CUTEAFD_ENABLE_V41_NVFP4_AOT "Build W4A4 ModelOpt NVFP4 expert AOT kernels" OFF)
option(CUTEAFD_ENABLE_V41_EXPERT_AOT "Build native V4.1 expert AOT kernels" OFF)
option(CUTEAFD_ENABLE_V41_NARROW_AOT "Use candidate V4.1 narrow projection scheduling" OFF)
option(CUTEAFD_ENABLE_V41_HC_LAGGED_AOT "Build candidate native lagged mHC begin" OFF)
option(CUTEAFD_ENABLE_V41_FP8_AOT "Build native V4.1 coordinator FP8 AOT kernels" OFF)
option(CUTEAFD_ENABLE_DSV4_AOT "Build the DeepSeek V4 coordinator programs" OFF)
option(CUTEAFD_ENABLE_GLM_AOT "Build the GLM 5.x coordinator programs (same table as DeepSeek V4)" OFF)
set(CUTEAFD_GLM_GEOMETRIES "glm;glm2" CACHE STRING
  "GLM program geometries with CUTEAFD_ENABLE_GLM_AOT: glm (GLM 5.x), glm2 (its two-GPU head split)")
option(CUTEAFD_ENABLE_MIMO_AOT "Build the MiMo V2 coordinator programs (same table as DeepSeek V4)" OFF)
set(CUTEAFD_MIMO_GEOMETRIES "mimo" CACHE STRING
  "MiMo program geometries with CUTEAFD_ENABLE_MIMO_AOT: mimo (V2 Flash), mimof (V2.6 Flash MOPD), mimop (V2.6 Pro); suffix 2 selects each two-GPU head split")
option(CUTEAFD_ENABLE_GLMF_AOT "Build the GLM 5.3 Flash coordinator programs (same table as DeepSeek V4)" OFF)
option(CUTEAFD_ENABLE_QWEN4_AOT "Build the Qwen 3.8 Flash Next coordinator programs (same table as DeepSeek V4)" OFF)
set(CUTEAFD_DSV4_GEOMETRY "flash,pro,flash2,pro2" CACHE STRING
  "DeepSeek V4 program geometries, comma-separated: flash, pro; flash2 / pro2 their two-GPU head splits")
set(CUTEAFD_DSV4_DECODE_ROWS "64" CACHE STRING "DeepSeek V4 decode program capacity (below 96 keeps mHC on its decode route)")
set(CUTEAFD_DSV4_PREFILL_ROWS "4096" CACHE STRING "DeepSeek V4 prefill program capacity")
set(CUTEAFD_GLMF_WIDE_DECODE_ROWS "128" CACHE STRING
  "GLM 5.3 Flash wide decode and verify program capacity (serve --decode-rows 128); 0 exports none")
set(CUTEAFD_CONTEXT_SPLIT_AOT "OFF" CACHE STRING
  "Token-split attention primitives (scored top-k, candidate merge, partial, combine, gather) for the flash/pro/glm/glmf program geometries: OFF, ON (after every other program) or ONLY")
set(CUTEAFD_DSV4_MAX_CONTEXT "1048576" CACHE STRING "Coordinator cache extent compiled into programs (clamped per checkpoint geometry)")
if((CUTEAFD_ENABLE_V41_HC_LAGGED_AOT OR CUTEAFD_ENABLE_V41_NARROW_AOT) AND NOT CUTEAFD_ENABLE_V41_FP8_AOT)
  message(FATAL_ERROR "Candidate mHC/narrow AOT requires V4.1 FP8 AOT")
endif()
option(CUTEAFD_ENABLE_V41_ATTENTION_AOT "Build candidate direct V4.1 attention AOT kernels" OFF)
option(CUTEAFD_ENABLE_RDMA "Build RDMA helper components when libibverbs is available" OFF)
option(CUTEAFD_ENABLE_NCCL "Build multi-node NCCL helper components" OFF)
option(CUTEAFD_ENABLE_XGRAMMAR "Build constrained-decoding grammar support" ON)
set(CUTEAFD_CUDA_ARCHITECTURES "120" CACHE STRING "CUDA architectures for native kernels")
set(CUTEAFD_SPARKINFER_SOURCE_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/sparkinfer"
  CACHE PATH "Pinned SparkInfer source tree used by every SparkInfer-backed AOT export")
set(CUTEAFD_SPARKINFER_LOCK_FILE
  "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/sparkinfer.lock.json"
  CACHE FILEPATH "Provenance lock for CUTEAFD_SPARKINFER_SOURCE_DIR")
set(CUTEAFD_XGRAMMAR_SOURCE_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/xgrammar"
  CACHE PATH "Pinned XGrammar source tree")
set(CUTEAFD_XGRAMMAR_LOCK_FILE
  "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/xgrammar.lock.json"
  CACHE FILEPATH "Provenance lock for CUTEAFD_XGRAMMAR_SOURCE_DIR")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CUTEAFD_NATIVE_SOURCES shared/src/cuteafd_native.cc)
# Headers live beside their code: shared/ and one families/<id>/ tree per
# family. Kernels include common.h/expert_hidden.cuh and the expert family
# sources include v41_experts.cc by name, so shared/cuda and shared/src are
# on the path too.
set(CUTEAFD_NATIVE_INCLUDE_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}/shared/include
  ${CMAKE_CURRENT_SOURCE_DIR}/families/deepseek_v41/include
  ${CMAKE_CURRENT_SOURCE_DIR}/shared/cuda
  ${CMAKE_CURRENT_SOURCE_DIR}/shared/src
)

if(CUTEAFD_ENABLE_XGRAMMAR)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  if(NOT EXISTS "${CUTEAFD_XGRAMMAR_SOURCE_DIR}/include/xgrammar/compiler.h" OR
     NOT EXISTS "${CUTEAFD_XGRAMMAR_SOURCE_DIR}/3rdparty/dlpack/include/dlpack/dlpack.h" OR
     NOT EXISTS "${CUTEAFD_XGRAMMAR_SOURCE_DIR}/3rdparty/picojson/picojson.h")
    message(FATAL_ERROR
      "Pinned XGrammar source is incomplete under ${CUTEAFD_XGRAMMAR_SOURCE_DIR}; "
      "initialize third_party/xgrammar and its nested submodules")
  endif()
  set(CUTEAFD_XGRAMMAR_VERIFY_SCRIPT
    "${CMAKE_CURRENT_SOURCE_DIR}/../scripts/build/verify-xgrammar-source.py")
  if(NOT EXISTS "${CUTEAFD_XGRAMMAR_VERIFY_SCRIPT}")
    message(FATAL_ERROR
      "XGrammar source verifier was not found: ${CUTEAFD_XGRAMMAR_VERIFY_SCRIPT}")
  endif()
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" "${CUTEAFD_XGRAMMAR_VERIFY_SCRIPT}"
      --source "${CUTEAFD_XGRAMMAR_SOURCE_DIR}"
      --lock "${CUTEAFD_XGRAMMAR_LOCK_FILE}"
    RESULT_VARIABLE CUTEAFD_XGRAMMAR_VERIFY_RESULT
    OUTPUT_VARIABLE CUTEAFD_XGRAMMAR_VERIFY_OUTPUT
    ERROR_VARIABLE CUTEAFD_XGRAMMAR_VERIFY_ERROR
  )
  if(NOT CUTEAFD_XGRAMMAR_VERIFY_RESULT EQUAL 0)
    message(FATAL_ERROR
      "XGrammar source verification failed:\n${CUTEAFD_XGRAMMAR_VERIFY_ERROR}")
  endif()
  string(STRIP "${CUTEAFD_XGRAMMAR_VERIFY_OUTPUT}" CUTEAFD_XGRAMMAR_VERIFY_OUTPUT)
  message(STATUS "${CUTEAFD_XGRAMMAR_VERIFY_OUTPUT}")
  file(GLOB_RECURSE CUTEAFD_XGRAMMAR_SOURCES
    CONFIGURE_DEPENDS
    "${CUTEAFD_XGRAMMAR_SOURCE_DIR}/cpp/*.cc"
  )
  list(FILTER CUTEAFD_XGRAMMAR_SOURCES EXCLUDE REGEX "/cpp/tvm_ffi/.*\\.cc$")
  # Extend the pinned schema factory in a generated translation unit. The
  # anonymous upstream parser is reused without changing the vendored tree.
  file(READ "${CUTEAFD_XGRAMMAR_SOURCE_DIR}/cpp/json_schema_converter.cc" CUTEAFD_SCHEMA_SOURCE)
  set(CUTEAFD_SCHEMA_FACTORY "switch (json_format) {\n    case JSONFormat::kJSON:")
  string(FIND "${CUTEAFD_SCHEMA_SOURCE}" "${CUTEAFD_SCHEMA_FACTORY}" CUTEAFD_SCHEMA_FACTORY_AT)
  if(CUTEAFD_SCHEMA_FACTORY_AT EQUAL -1)
    message(FATAL_ERROR "Pinned XGrammar schema factory no longer matches the V4.1 extension")
  endif()
  string(REPLACE "#include \"json_schema_converter.h\""
    "#include \"json_schema_converter.h\"\n#include \"xgrammar_v41_tools.h\""
    CUTEAFD_SCHEMA_SOURCE "${CUTEAFD_SCHEMA_SOURCE}")
  string(REPLACE "${CUTEAFD_SCHEMA_FACTORY}"
    "switch (json_format) {\n    case static_cast<JSONFormat>(5): {\n      V41ToolCallingConverter converter(ref_resolver);\n      return converter.Convert(spec);\n    }\n    case JSONFormat::kJSON:"
    CUTEAFD_SCHEMA_SOURCE "${CUTEAFD_SCHEMA_SOURCE}")
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/v41_xgrammar_schema.cc" "${CUTEAFD_SCHEMA_SOURCE}")
  list(FILTER CUTEAFD_XGRAMMAR_SOURCES EXCLUDE REGEX "/cpp/json_schema_converter\\.cc$")
  # Correct UTF-8 range tails and Unicode negation in a generated copy. Source
  # provenance is checked above; the patcher additionally verifies its anchors.
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/shared/src/xgrammar_unicode_fsm.inc"
    "${CMAKE_CURRENT_SOURCE_DIR}/../scripts/build/prepare-xgrammar-unicode.py")
  execute_process(COMMAND "${Python3_EXECUTABLE}"
    "${CMAKE_CURRENT_SOURCE_DIR}/../scripts/build/prepare-xgrammar-unicode.py"
    --source "${CUTEAFD_XGRAMMAR_SOURCE_DIR}/cpp/grammar_functor.cc"
    --replacement "${CMAKE_CURRENT_SOURCE_DIR}/shared/src/xgrammar_unicode_fsm.inc"
    --output "${CMAKE_CURRENT_BINARY_DIR}/v41_xgrammar_unicode.cc"
    COMMAND_ERROR_IS_FATAL ANY)
  list(FILTER CUTEAFD_XGRAMMAR_SOURCES EXCLUDE REGEX "/cpp/grammar_functor\\.cc$")
  list(APPEND CUTEAFD_XGRAMMAR_SOURCES "${CMAKE_CURRENT_BINARY_DIR}/v41_xgrammar_unicode.cc")
  list(APPEND CUTEAFD_XGRAMMAR_SOURCES "${CMAKE_CURRENT_BINARY_DIR}/v41_xgrammar_schema.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/shared/src/xgrammar_regular.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/shared/src/xgrammar_v41_tools.cc")
  add_library(cuteafd_xgrammar STATIC ${CUTEAFD_XGRAMMAR_SOURCES})
  target_include_directories(cuteafd_xgrammar PRIVATE
    "${CUTEAFD_XGRAMMAR_SOURCE_DIR}/cpp" "${CMAKE_CURRENT_SOURCE_DIR}/shared/src")
  target_include_directories(cuteafd_xgrammar
    PUBLIC "${CUTEAFD_XGRAMMAR_SOURCE_DIR}/include"
  )
  target_include_directories(cuteafd_xgrammar SYSTEM PUBLIC
    "${CUTEAFD_XGRAMMAR_SOURCE_DIR}/3rdparty/dlpack/include"
    "${CUTEAFD_XGRAMMAR_SOURCE_DIR}/3rdparty/picojson"
  )
  target_compile_definitions(cuteafd_xgrammar PUBLIC
    XGRAMMAR_ENABLE_CPPTRACE=0
    XGRAMMAR_ENABLE_INTERNAL_CHECK=0
  )
  find_package(Threads REQUIRED)
  target_link_libraries(cuteafd_xgrammar PUBLIC Threads::Threads)
  list(APPEND CUTEAFD_NATIVE_SOURCES shared/src/xgrammar_adapter.cc)
endif()

set(CUTEAFD_BUILD_SPARKINFER_AOT OFF)
# FAMILY:fp8 and FAMILY:nvfp4 expert families (fp8_moe packages, cmake/shared/fp8_moe.cmake).
set(CUTEAFD_FP8_EXPERT_FAMILIES OFF)
if(CUTEAFD_EXPERT_FAMILIES MATCHES ":(fp8|nvfp4)")
  set(CUTEAFD_FP8_EXPERT_FAMILIES ON)
endif()
if(CUTEAFD_ENABLE_AUDIO_AOT OR CUTEAFD_ENABLE_VISION_ATTENTION_AOT OR CUTEAFD_ENABLE_V41_EXPERT_AOT OR CUTEAFD_ENABLE_V41_LOCAL_EXPERT_AOT OR
   CUTEAFD_ENABLE_V41_TP2_EXPERT_AOT OR CUTEAFD_ENABLE_EXL3_PACKAGES OR CUTEAFD_ENABLE_V41_NVFP4_AOT OR CUTEAFD_ENABLE_V41_FP8_AOT OR CUTEAFD_ENABLE_V41_ATTENTION_AOT OR CUTEAFD_ENABLE_DSV4_AOT OR CUTEAFD_ENABLE_GLM_AOT OR
   CUTEAFD_ENABLE_MIMO_AOT OR CUTEAFD_ENABLE_GLMF_AOT OR CUTEAFD_ENABLE_QWEN4_AOT OR CUTEAFD_FP8_EXPERT_FAMILIES)
  set(CUTEAFD_BUILD_SPARKINFER_AOT ON)
endif()
if(CUTEAFD_BUILD_SPARKINFER_AOT)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  if(NOT EXISTS "${CUTEAFD_SPARKINFER_SOURCE_DIR}/b12x/__init__.py")
    message(FATAL_ERROR
      "SparkInfer was not found under CUTEAFD_SPARKINFER_SOURCE_DIR="
      "${CUTEAFD_SPARKINFER_SOURCE_DIR}. Initialize third_party/sparkinfer or "
      "select an explicit pinned source tree.")
  endif()
  set(CUTEAFD_SPARKINFER_VERIFY_SCRIPT
    "${CMAKE_CURRENT_SOURCE_DIR}/../scripts/build/verify-sparkinfer-source.py")
  set(CUTEAFD_SPARKINFER_PYTHON_BOOTSTRAP
    "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/lib/_pinned_sparkinfer.py")
  if(NOT EXISTS "${CUTEAFD_SPARKINFER_VERIFY_SCRIPT}")
    message(FATAL_ERROR
      "SparkInfer source verifier was not found: ${CUTEAFD_SPARKINFER_VERIFY_SCRIPT}")
  endif()
  if(NOT EXISTS "${CUTEAFD_SPARKINFER_PYTHON_BOOTSTRAP}")
    message(FATAL_ERROR
      "SparkInfer Python bootstrap was not found: "
      "${CUTEAFD_SPARKINFER_PYTHON_BOOTSTRAP}")
  endif()
  set(CUTEAFD_SPARKINFER_VERIFY_COMMAND
    "${Python3_EXECUTABLE}" "${CUTEAFD_SPARKINFER_VERIFY_SCRIPT}"
    --source "${CUTEAFD_SPARKINFER_SOURCE_DIR}"
    --lock "${CUTEAFD_SPARKINFER_LOCK_FILE}"
  )
  set(CUTEAFD_SPARKINFER_PYTHON_ENV
    "CUTEAFD_SPARKINFER_SOURCE_DIR=${CUTEAFD_SPARKINFER_SOURCE_DIR}"
    "CUTEAFD_SPARKINFER_LOCK_FILE=${CUTEAFD_SPARKINFER_LOCK_FILE}"
    # Persisted objects are executable-only ExternalBinaryModules. AOT export
    # needs the live compiler IR consumed by export_to_c(), so never substitute
    # a runtime disk-cache hit in these build-time Python processes.
    "B12X_COMPILE_DISK_CACHE=0"
    "PYTHONDONTWRITEBYTECODE=1"
    "PYTHONPATH=${CUTEAFD_SPARKINFER_SOURCE_DIR}"
  )
  execute_process(
    COMMAND ${CUTEAFD_SPARKINFER_VERIFY_COMMAND}
    RESULT_VARIABLE CUTEAFD_SPARKINFER_VERIFY_RESULT
    OUTPUT_VARIABLE CUTEAFD_SPARKINFER_VERIFY_OUTPUT
    ERROR_VARIABLE CUTEAFD_SPARKINFER_VERIFY_ERROR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_STRIP_TRAILING_WHITESPACE
  )
  if(NOT CUTEAFD_SPARKINFER_VERIFY_RESULT EQUAL 0)
    message(FATAL_ERROR
      "Pinned SparkInfer verification failed:\n${CUTEAFD_SPARKINFER_VERIFY_ERROR}")
  endif()
  message(STATUS "${CUTEAFD_SPARKINFER_VERIFY_OUTPUT}")
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${CUTEAFD_SPARKINFER_LOCK_FILE}"
    "${CUTEAFD_SPARKINFER_VERIFY_SCRIPT}"
  )
  set(CUTEAFD_SPARKINFER_PROVENANCE_INPUTS
    "${CUTEAFD_SPARKINFER_LOCK_FILE}"
    "${CUTEAFD_SPARKINFER_VERIFY_SCRIPT}"
    "${CUTEAFD_SPARKINFER_PYTHON_BOOTSTRAP}"
  )
  file(GLOB_RECURSE CUTEAFD_SPARKINFER_EXPORT_INPUTS
    LIST_DIRECTORIES false
    CONFIGURE_DEPENDS
    "${CUTEAFD_SPARKINFER_SOURCE_DIR}/b12x/*.py"
    "${CUTEAFD_SPARKINFER_SOURCE_DIR}/b12x/*.cu"
    "${CUTEAFD_SPARKINFER_SOURCE_DIR}/b12x/*.cuh"
    "${CUTEAFD_SPARKINFER_SOURCE_DIR}/b12x/*.h"
    "${CUTEAFD_SPARKINFER_SOURCE_DIR}/b12x/*.hpp"
  )
  list(APPEND CUTEAFD_SPARKINFER_EXPORT_INPUTS
    "${CUTEAFD_SPARKINFER_SOURCE_DIR}/pyproject.toml"
  )
  add_custom_target(cuteafd_verify_sparkinfer_source
    COMMAND ${CUTEAFD_SPARKINFER_VERIFY_COMMAND}
    COMMENT "Verifying pinned SparkInfer source"
    VERBATIM
  )
endif()

if(CUTEAFD_ENABLE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  list(APPEND CUTEAFD_NATIVE_SOURCES
    shared/cuda/expert_pack.cu
    shared/cuda/exl3_wire.cu
    shared/cuda/route_reduce.cu
    families/deepseek_v41/cuda/engram.cu
    families/deepseek_v41/cuda/v41_dspark.cu
    families/deepseek_v41/cuda/v41_hc.cu
    shared/cuda/expert_router.cu
    families/deepseek_v4/cuda/dsv4_dspark.cu
    families/glm5/cuda/glm_dflash.cu
    families/glm5_flash/cuda/glmf_dspark.cu
    families/glm5/cuda/glm_mla_prefill.cu
    shared/cuda/vocab_head_rows.cu
    families/mimo_v2/cuda/mimo_dflash.cu
    shared/cuda/fp8_block_dequant.cu
    shared/cuda/fp8_gemv.cu
    shared/cuda/l2_prefetch.cu
    families/deepseek_v41/cuda/v41_dspark_cache.cu
    families/deepseek_v41/cuda/v41_dspark_attention.cu
    families/deepseek_v41/cuda/v41_compressor.cu
    families/deepseek_v41/cuda/v41_vision.cu
    shared/cuda/vision_runtime.cu
    families/deepseek_v41/cuda/v41_kv.cu
    families/deepseek_v41/cuda/v41_sparse_attention.cu
    shared/cuda/peer_copy.cu
    shared/cuda/peer_exchange.cu
    families/deepseek_v41/cuda/v41_index_scores.cu
    families/deepseek_v41/cuda/v41_index_topk.cu
    families/deepseek_v41/cuda/v41_candidate_blocks.cu
    families/deepseek_v41/cuda/v41_attention_ops.cu
    shared/cuda/norm.cu
    shared/cuda/mlp.cu
    shared/cuda/residual.cu
    shared/cuda/kv_cache.cu
    shared/cuda/mla_indexing.cu
    shared/cuda/router.cu
    shared/cuda/linear.cu
    shared/cuda/attention.cu
    shared/cuda/embedding.cu
    shared/cuda/token_io.cu
    shared/cuda/sampling.cu
    shared/cuda/sampling_gpu.cu
    shared/cuda/nibbles.cu
    shared/cuda/nvfp4_payload.cu
    shared/cuda/nvfp4_scale.cu
  )
  # SM120 builds use the family-specific target (sm_120f); every live AOT
  # export and hand kernel is qualified with it.
  if(CUTEAFD_CUDA_ARCHITECTURES STREQUAL "120")
    set(CUTEAFD_CUDA_ARCHITECTURES "120f")
  endif()
endif()

# Every CuTe-DSL AOT export (any program table, expert kernels, EXL3/FP8 packages) calls
# into the CuTe DSL runtime: one condition finds it and links it.
if(CUTEAFD_BUILD_SPARKINFER_AOT)
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -m cutlass.cute.export.aot_config --libdir
    OUTPUT_VARIABLE CUTEAFD_B12X_AOT_RUNTIME_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
  )
  find_library(CUTEAFD_B12X_AOT_RUNTIME_LIBRARY
    NAMES cute_dsl_runtime
    PATHS "${CUTEAFD_B12X_AOT_RUNTIME_DIR}"
    NO_DEFAULT_PATH
    REQUIRED
  )
endif()

if(CUTEAFD_ENABLE_V41_EXPERT_AOT)
  include(cmake/families/deepseek_v41/v41_experts.cmake)
endif()
if(NOT CUTEAFD_SPARK_TP_ROLES STREQUAL "")
  include(cmake/families/deepseek_v41/v41_spark_tp_experts.cmake)
endif()
if(NOT CUTEAFD_EXPERT_FAMILIES STREQUAL "")
  include(cmake/shared/expert_families.cmake)
endif()
if(CUTEAFD_FP8_EXPERT_FAMILIES)
  include(cmake/shared/fp8_moe.cmake)
endif()
if(CUTEAFD_ENABLE_DSV4_AOT OR CUTEAFD_ENABLE_GLM_AOT OR CUTEAFD_ENABLE_MIMO_AOT OR CUTEAFD_ENABLE_GLMF_AOT OR CUTEAFD_ENABLE_QWEN4_AOT)
  include(cmake/shared/dsv4_programs.cmake)
endif()
if(CUTEAFD_ENABLE_V41_LOCAL_EXPERT_AOT)
  include(cmake/families/deepseek_v41/v41_local_experts.cmake)
endif()
if(CUTEAFD_ENABLE_V41_TP2_EXPERT_AOT)
  include(cmake/families/deepseek_v41/v41_tp2_experts.cmake)
  include(cmake/families/deepseek_v41/v41_dspark_tp2_experts.cmake)
endif()
if(CUTEAFD_ENABLE_V41_FP8_AOT)
  include(cmake/families/deepseek_v41/v41_fp8.cmake)
  include(cmake/families/deepseek_v41/v41_router.cmake)
endif()
if(CUTEAFD_ENABLE_V41_ATTENTION_AOT)
  include(cmake/families/deepseek_v41/v41_attention.cmake)
endif()
if(CUTEAFD_ENABLE_V41_NVFP4_AOT)
  include(cmake/families/deepseek_v41/v41_nvfp4_experts.cmake)
endif()
if(CUTEAFD_ENABLE_EXL3_PACKAGES)
  include(cmake/shared/exl3.cmake)
endif()
if(CUTEAFD_ENABLE_VISION_ATTENTION_AOT)
  include(cmake/shared/vision_attention.cmake)
endif()
if(CUTEAFD_ENABLE_AUDIO_AOT)
  include(cmake/shared/audio.cmake)
endif()
add_library(cuteafd_native SHARED ${CUTEAFD_NATIVE_SOURCES})
if(CUTEAFD_ENABLE_AUDIO_AOT)
  target_compile_definitions(cuteafd_native PRIVATE CUTEAFD_AUDIO_AOT_SM=${CUTEAFD_AUDIO_CAPABILITY})
  target_include_directories(cuteafd_native PRIVATE "${CUTEAFD_AUDIO_DIR}")
  target_link_libraries(cuteafd_native PRIVATE CUDA::cufft)
  target_link_options(cuteafd_native PRIVATE "-Wl,-z,defs")
  add_dependencies(cuteafd_native cuteafd_audio_export)
endif()
if(CUTEAFD_ENABLE_VISION_ATTENTION_AOT)
  target_compile_definitions(cuteafd_native PRIVATE CUTEAFD_HAVE_VISION_ATTENTION_AOT=1)
  target_include_directories(cuteafd_native PRIVATE "${CUTEAFD_VISION_ATTENTION_DIR}")
  add_dependencies(cuteafd_native cuteafd_vision_attention_export)
endif()
if(CUTEAFD_ENABLE_EXL3_PACKAGES)
  add_dependencies(cuteafd_native cuteafd_v41_exl3_export)
endif()
if(CUTEAFD_ENABLE_V41_ATTENTION_AOT)
  target_compile_definitions(cuteafd_native PRIVATE CUTEAFD_HAVE_V41_ATTENTION_AOT=1)
  add_dependencies(cuteafd_native cuteafd_v41_attention_export cuteafd_v41_attention_heads32_export)
  target_include_directories(cuteafd_native PRIVATE "${CUTEAFD_V41_ATTENTION_DIR}")
endif()
if(CUTEAFD_ENABLE_V41_FP8_AOT)
  # Packed FP8->BF16 conversion requires the SM120 architecture-specific ISA.
  # Keep the normal target's portable image as a fallback in the same fatbin.
  set_property(SOURCE families/deepseek_v41/cuda/v41_sparse_attention.cu APPEND PROPERTY
    COMPILE_OPTIONS "--generate-code=arch=compute_120a,code=sm_120a")
  target_compile_definitions(cuteafd_native PRIVATE CUTEAFD_HAVE_V41_HC_AOT=1 CUTEAFD_HAVE_V41_ROUTER_AOT=1)
  add_dependencies(cuteafd_native cuteafd_v41_fp8_export cuteafd_v41_index_export cuteafd_v41_router_export)
  target_include_directories(cuteafd_native PRIVATE "${CUTEAFD_V41_FP8_DIR}" "${CUTEAFD_V41_INDEX_DIR}" "${CUTEAFD_V41_ROUTER_DIR}")
endif()
if(CUTEAFD_ENABLE_V41_EXPERT_AOT)
  add_dependencies(cuteafd_native cuteafd_v41_experts_export)
  target_include_directories(cuteafd_native PRIVATE "${CUTEAFD_V41_EXPERT_DIR}")
endif()
if(CUTEAFD_V41_SPARK_TP_EXPERT_TARGETS)
  add_dependencies(cuteafd_native ${CUTEAFD_V41_SPARK_TP_EXPERT_TARGETS})
  target_include_directories(cuteafd_native PRIVATE ${CUTEAFD_V41_SPARK_TP_EXPERT_INCLUDE_DIRS})
endif()
if(CUTEAFD_ENABLE_DSV4_AOT OR CUTEAFD_ENABLE_GLM_AOT OR CUTEAFD_ENABLE_MIMO_AOT OR CUTEAFD_ENABLE_GLMF_AOT OR CUTEAFD_ENABLE_QWEN4_AOT)
  add_dependencies(cuteafd_native cuteafd_dsv4_programs_export)
  target_include_directories(cuteafd_native PRIVATE "${CUTEAFD_DSV4_DIR}")
  # Whole archive: the table in dsv4_programs.cc names every program symbol.
  target_link_libraries(cuteafd_native PRIVATE "-Wl,--whole-archive" "${CUTEAFD_DSV4_ARCHIVE}" "-Wl,--no-whole-archive")
endif()
if(CUTEAFD_EXPERT_FAMILY_TARGETS)
  add_dependencies(cuteafd_native ${CUTEAFD_EXPERT_FAMILY_TARGETS})
  target_include_directories(cuteafd_native PRIVATE ${CUTEAFD_EXPERT_FAMILY_INCLUDE_DIRS})
endif()
if(CUTEAFD_ENABLE_V41_LOCAL_EXPERT_AOT)
  add_dependencies(cuteafd_native cuteafd_v41_local_experts_export)
  target_include_directories(cuteafd_native PRIVATE "${CUTEAFD_V41_LOCAL_EXPERT_DIR}")
endif()
if(CUTEAFD_ENABLE_V41_TP2_EXPERT_AOT)
  add_dependencies(cuteafd_native cuteafd_v41_tp2_experts_export cuteafd_v41_dspark_tp2_experts_export)
  target_include_directories(cuteafd_native PRIVATE "${CUTEAFD_V41_TP2_EXPERT_DIR}" "${CUTEAFD_V41_DSPARK_TP2_EXPERT_DIR}")
endif()
if(CUTEAFD_ENABLE_V41_NVFP4_AOT)
  foreach(nvfp4_role IN LISTS CUTEAFD_V41_NVFP4_ROLES)
    add_dependencies(cuteafd_native cuteafd_v41_nvfp4_${nvfp4_role}_export)
  endforeach()
  target_include_directories(cuteafd_native PRIVATE ${CUTEAFD_V41_NVFP4_INCLUDE_DIRS})
endif()
target_include_directories(cuteafd_native PUBLIC ${CUTEAFD_NATIVE_INCLUDE_DIRS})
if(CUTEAFD_ENABLE_XGRAMMAR)
  target_link_libraries(cuteafd_native PRIVATE cuteafd_xgrammar)
  target_include_directories(cuteafd_native PRIVATE "${CUTEAFD_XGRAMMAR_SOURCE_DIR}/cpp")
  target_compile_definitions(cuteafd_native PRIVATE CUTEAFD_NATIVE_ENABLE_XGRAMMAR=1)
else()
  target_compile_definitions(cuteafd_native PRIVATE CUTEAFD_NATIVE_ENABLE_XGRAMMAR=0)
endif()
if(CUTEAFD_ENABLE_CUDA)
  target_compile_definitions(cuteafd_native PRIVATE CUTEAFD_NATIVE_ENABLE_CUDA=1)
  if(CUTEAFD_V41_VOCAB_ROW_EXPERIMENT)
    target_compile_definitions(cuteafd_native PRIVATE CUTEAFD_V41_VOCAB_ROW_EXPERIMENT=1)
  endif()
  target_link_libraries(cuteafd_native PUBLIC CUDA::cudart CUDA::cublas CUDA::cuda_driver)
  set_target_properties(cuteafd_native PROPERTIES
    CUDA_ARCHITECTURES "${CUTEAFD_CUDA_ARCHITECTURES}"
  )
else()
  target_compile_definitions(cuteafd_native PRIVATE CUTEAFD_NATIVE_ENABLE_CUDA=0)
endif()
if(CUTEAFD_BUILD_SPARKINFER_AOT)
  target_link_libraries(cuteafd_native PUBLIC "${CUTEAFD_B12X_AOT_RUNTIME_LIBRARY}")
  set_property(TARGET cuteafd_native APPEND PROPERTY BUILD_RPATH "${CUTEAFD_B12X_AOT_RUNTIME_DIR}")
endif()
if(CUTEAFD_ENABLE_RDMA)
  find_path(CUTEAFD_IBVERBS_INCLUDE_DIR infiniband/verbs.h)
  find_library(CUTEAFD_IBVERBS_LIBRARY NAMES ibverbs)
  if(NOT CUTEAFD_IBVERBS_INCLUDE_DIR OR NOT CUTEAFD_IBVERBS_LIBRARY)
    message(FATAL_ERROR "CUTEAFD_ENABLE_RDMA=ON requires libibverbs headers and library")
  endif()
  target_compile_definitions(cuteafd_native PRIVATE CUTEAFD_NATIVE_ENABLE_RDMA=1)
  target_include_directories(cuteafd_native PRIVATE ${CUTEAFD_IBVERBS_INCLUDE_DIR})
  target_link_libraries(cuteafd_native PUBLIC ${CUTEAFD_IBVERBS_LIBRARY})
else()
  target_compile_definitions(cuteafd_native PRIVATE CUTEAFD_NATIVE_ENABLE_RDMA=0)
endif()
if(CUTEAFD_ENABLE_NCCL)
  if(NOT CUTEAFD_ENABLE_CUDA)
    message(FATAL_ERROR "CUTEAFD_ENABLE_NCCL=ON requires CUTEAFD_ENABLE_CUDA=ON")
  endif()
  find_path(CUTEAFD_NCCL_INCLUDE_DIR nccl.h REQUIRED)
  find_library(CUTEAFD_NCCL_LIBRARY NAMES nccl REQUIRED)
  target_compile_definitions(cuteafd_native PRIVATE CUTEAFD_NATIVE_ENABLE_NCCL=1)
  target_include_directories(cuteafd_native PRIVATE ${CUTEAFD_NCCL_INCLUDE_DIR})
  target_link_libraries(cuteafd_native PUBLIC ${CUTEAFD_NCCL_LIBRARY})
else()
  target_compile_definitions(cuteafd_native PRIVATE CUTEAFD_NATIVE_ENABLE_NCCL=0)
endif()

add_executable(cuteafd_native_selftest tests/native_selftest.cc)
target_link_libraries(cuteafd_native_selftest PRIVATE cuteafd_native)
# The self-test intentionally uses assertions for both the ABI call and its
# result check. Keep those expressions live in Release builds; defining
# NDEBUG would otherwise skip allocations and make the following copy probe
# dereference an uninitialized buffer.
if(MSVC)
  target_compile_options(cuteafd_native_selftest PRIVATE /UNDEBUG)
else()
  target_compile_options(cuteafd_native_selftest PRIVATE -UNDEBUG)
endif()

enable_testing()
add_test(NAME cuteafd_native_selftest COMMAND cuteafd_native_selftest)

if(CUTEAFD_ENABLE_XGRAMMAR)
  add_executable(cuteafd_xgrammar_regular_selftest tests/xgrammar_regular_selftest.cc)
  target_link_libraries(cuteafd_xgrammar_regular_selftest PRIVATE cuteafd_xgrammar)
  target_include_directories(cuteafd_xgrammar_regular_selftest PRIVATE
    "${CUTEAFD_XGRAMMAR_SOURCE_DIR}/cpp" "${CMAKE_CURRENT_SOURCE_DIR}/shared/src")
  add_test(NAME cuteafd_xgrammar_regular_selftest COMMAND cuteafd_xgrammar_regular_selftest)
  add_executable(cuteafd_xgrammar_selftest tests/xgrammar_selftest.cc)
  target_link_libraries(cuteafd_xgrammar_selftest PRIVATE cuteafd_native)
  target_compile_definitions(cuteafd_xgrammar_selftest PRIVATE
    "CUTEAFD_XGRAMMAR_TEST_TOKENIZER=\"${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/xgrammar_tiny_tokenizer.json\""
  )
  if(MSVC)
    target_compile_options(cuteafd_xgrammar_selftest PRIVATE /UNDEBUG)
  else()
    target_compile_options(cuteafd_xgrammar_selftest PRIVATE -UNDEBUG)
  endif()
  add_test(NAME cuteafd_xgrammar_selftest COMMAND cuteafd_xgrammar_selftest)
endif()

if(CUTEAFD_ENABLE_CUDA)
  add_executable(cuteafd_v41_peer_copy_selftest tests/v41_peer_copy_selftest.cc)
  target_link_libraries(cuteafd_v41_peer_copy_selftest PRIVATE cuteafd_native CUDA::cudart)
  add_test(NAME cuteafd_v41_peer_copy_selftest COMMAND cuteafd_v41_peer_copy_selftest)
  set_tests_properties(cuteafd_v41_peer_copy_selftest PROPERTIES SKIP_RETURN_CODE 77)
  add_executable(cuteafd_v41_route_reduce_planes_selftest
    tests/v41_route_reduce_planes_selftest.cc)
  target_link_libraries(cuteafd_v41_route_reduce_planes_selftest PRIVATE cuteafd_native CUDA::cudart)
  add_test(NAME cuteafd_v41_route_reduce_planes_selftest
    COMMAND cuteafd_v41_route_reduce_planes_selftest)
  set_tests_properties(cuteafd_v41_route_reduce_planes_selftest PROPERTIES SKIP_RETURN_CODE 77)
  add_executable(cuteafd_v41_expert_pack_tp3_selftest
    tests/v41_expert_pack_tp3_selftest.cc)
  target_link_libraries(cuteafd_v41_expert_pack_tp3_selftest PRIVATE cuteafd_native CUDA::cudart)
  add_test(NAME cuteafd_v41_expert_pack_tp3_selftest
    COMMAND cuteafd_v41_expert_pack_tp3_selftest)
  set_tests_properties(cuteafd_v41_expert_pack_tp3_selftest PROPERTIES SKIP_RETURN_CODE 77)
  # Geometry-driven packer check: every native extent, including the padded TP4
  # 576 -> 640 storage, in one parameterized device test.
  add_executable(cuteafd_v41_expert_pack_geometry_selftest
    tests/v41_expert_pack_geometry_selftest.cc)
  target_link_libraries(cuteafd_v41_expert_pack_geometry_selftest PRIVATE cuteafd_native CUDA::cudart)
  add_test(NAME cuteafd_v41_expert_pack_geometry_selftest
    COMMAND cuteafd_v41_expert_pack_geometry_selftest)
  set_tests_properties(cuteafd_v41_expert_pack_geometry_selftest PROPERTIES SKIP_RETURN_CODE 77)
  add_executable(cuteafd_v41_fp4_attention_selftest tests/v41_fp4_attention_selftest.cc)
  target_link_libraries(cuteafd_v41_fp4_attention_selftest PRIVATE cuteafd_native CUDA::cudart)
  if(CUTEAFD_ENABLE_V41_ATTENTION_AOT)
    target_compile_definitions(cuteafd_v41_fp4_attention_selftest PRIVATE CUTEAFD_TEST_ATTENTION_AOT=1)
  endif()
  add_test(NAME cuteafd_v41_fp4_attention_selftest COMMAND cuteafd_v41_fp4_attention_selftest)
  add_executable(cuteafd_cuda_inplace_rope_selftest tests/cuda_inplace_rope_selftest.cc)
  target_link_libraries(cuteafd_cuda_inplace_rope_selftest PRIVATE cuteafd_native)
  add_test(NAME cuteafd_cuda_inplace_rope_selftest COMMAND cuteafd_cuda_inplace_rope_selftest)
  set_tests_properties(cuteafd_cuda_inplace_rope_selftest PROPERTIES TIMEOUT 15)
  add_executable(cuteafd_cuda_checked_argmax_selftest tests/cuda_checked_argmax_selftest.cc)
  target_link_libraries(cuteafd_cuda_checked_argmax_selftest PRIVATE cuteafd_native)
  add_test(NAME cuteafd_cuda_checked_argmax_selftest COMMAND cuteafd_cuda_checked_argmax_selftest)
  set_tests_properties(cuteafd_cuda_checked_argmax_selftest PROPERTIES TIMEOUT 15)
  add_executable(cuteafd_cuda_stream_query_selftest tests/cuda_stream_query_selftest.cc)
  target_link_libraries(cuteafd_cuda_stream_query_selftest PRIVATE cuteafd_native CUDA::cudart)
  add_test(NAME cuteafd_cuda_stream_query_selftest COMMAND cuteafd_cuda_stream_query_selftest)
  set_tests_properties(cuteafd_cuda_stream_query_selftest PROPERTIES TIMEOUT 15)
  add_executable(cuteafd_cuda_peer_selftest tests/cuda_peer_selftest.cc)
  target_link_libraries(cuteafd_cuda_peer_selftest PRIVATE cuteafd_native CUDA::cudart)
  add_test(NAME cuteafd_cuda_peer_selftest COMMAND cuteafd_cuda_peer_selftest)
  set_tests_properties(cuteafd_cuda_peer_selftest PROPERTIES TIMEOUT 30 SKIP_RETURN_CODE 77)
  add_executable(cuteafd_cuda_peer_planes_selftest tests/cuda_peer_planes_selftest.cc)
  target_link_libraries(cuteafd_cuda_peer_planes_selftest PRIVATE cuteafd_native CUDA::cudart)
  add_test(NAME cuteafd_cuda_peer_planes_selftest COMMAND cuteafd_cuda_peer_planes_selftest)
  set_tests_properties(cuteafd_cuda_peer_planes_selftest PROPERTIES TIMEOUT 30 SKIP_RETURN_CODE 77)
  add_executable(cuteafd_cuda_selftest tests/cuda_selftest.cc)
  target_link_libraries(cuteafd_cuda_selftest PRIVATE cuteafd_native CUDA::cudart CUDA::cublas CUDA::cuda_driver)
  add_test(NAME cuteafd_cuda_selftest COMMAND cuteafd_cuda_selftest)
  add_executable(cuteafd_vision_pointwise_selftest tests/vision_pointwise_test.cu)
  set_target_properties(cuteafd_vision_pointwise_selftest PROPERTIES
    CUDA_ARCHITECTURES "${CUTEAFD_CUDA_ARCHITECTURES}")
  target_include_directories(cuteafd_vision_pointwise_selftest PRIVATE ${CUTEAFD_NATIVE_INCLUDE_DIRS})
  target_link_libraries(cuteafd_vision_pointwise_selftest PRIVATE CUDA::cudart CUDA::cublas)
  add_test(NAME cuteafd_vision_pointwise_selftest COMMAND cuteafd_vision_pointwise_selftest)
  set_tests_properties(cuteafd_vision_pointwise_selftest PROPERTIES TIMEOUT 30 SKIP_RETURN_CODE 77)
  # Chunk-1 GPU target-sampler (K1) device tests. They link the shipped native
  # library and therefore also pin the exported C ABI of sampling_gpu.h.
  # `.cu` because the subnormal/FTZ canary defines a probe kernel compiled with
  # the same CUDA flags as the shipped sampler objects.
  add_executable(cuteafd_v41_sampling_selftest tests/v41_sampling_selftest.cu)
  set_source_files_properties(tests/v41_sampling_selftest.cu PROPERTIES LANGUAGE CUDA)
  # The FTZ canary compiles device code in this target, so it must target the
  # same architecture as the kernels it canaries; without this the executable
  # silently defaults to sm_75 and its probe kernel never launches.
  set_target_properties(cuteafd_v41_sampling_selftest PROPERTIES
    CUDA_ARCHITECTURES "${CUTEAFD_CUDA_ARCHITECTURES}")
  target_link_libraries(cuteafd_v41_sampling_selftest PRIVATE cuteafd_native CUDA::cudart)
  add_test(NAME cuteafd_v41_sampling_selftest COMMAND cuteafd_v41_sampling_selftest)
  set_tests_properties(cuteafd_v41_sampling_selftest PROPERTIES SKIP_RETURN_CODE 77)
  # Sparse MLA prefill kernels (F16, E4M3) against an FP64 reference, determinism and times.
  add_executable(cuteafd_glm_mla_prefill_selftest tests/glm_mla_prefill_selftest.cu)
  set_target_properties(cuteafd_glm_mla_prefill_selftest PROPERTIES
    CUDA_ARCHITECTURES "${CUTEAFD_CUDA_ARCHITECTURES}")
  target_link_libraries(cuteafd_glm_mla_prefill_selftest PRIVATE cuteafd_native CUDA::cudart)
  add_test(NAME cuteafd_glm_mla_prefill_selftest COMMAND cuteafd_glm_mla_prefill_selftest 512 656 64 2)
  add_test(NAME cuteafd_glm_mla_prefill_selftest_528 COMMAND cuteafd_glm_mla_prefill_selftest 512 528 64 2)
endif()
