# Coordinator-only native K32 activation and block-FP8 projection artifacts.
if(NOT CUTEAFD_ENABLE_CUDA OR NOT
   (CUTEAFD_CUDA_ARCHITECTURES STREQUAL "120" OR CUTEAFD_CUDA_ARCHITECTURES STREQUAL "120f"))
  message(FATAL_ERROR "V4.1 coordinator FP8 AOT requires a native SM120 CUDA target")
endif()
set(CUTEAFD_V41_FP8_DIR "${CMAKE_CURRENT_BINARY_DIR}/v41_fp8")
set(CUTEAFD_V41_FP8_OBJECTS)
set(CUTEAFD_V41_FP8_HEADERS)
foreach(projection IN ITEMS engram ffn_up ffn_down main q_a q_b kv o_b o_a index_q q_b_tp2 o_b_tp2 ffn_tp2_up ffn_tp2_down)
foreach(rows IN LISTS CUTEAFD_V41_PROJECTION_ROWS)
  set(kinds quant gemm)
  if(projection STREQUAL "o_a")
    list(APPEND kinds quant_rope)
  endif()
  foreach(kind IN LISTS kinds)
    set(stem "${CUTEAFD_V41_FP8_DIR}/v41_${projection}_fp8_m${rows}_${kind}")
    list(APPEND CUTEAFD_V41_FP8_OBJECTS "${stem}.o")
    list(APPEND CUTEAFD_V41_FP8_HEADERS "${stem}.h")
  endforeach()
  if((projection STREQUAL "q_a" OR projection STREQUAL "kv") AND (rows EQUAL 64 OR rows EQUAL 256))
    foreach(kind IN LISTS kinds)
      set(stem "${CUTEAFD_V41_FP8_DIR}/v41_${projection}_fp8_m${rows}_decode_${kind}")
      list(APPEND CUTEAFD_V41_FP8_OBJECTS "${stem}.o")
      list(APPEND CUTEAFD_V41_FP8_HEADERS "${stem}.h")
    endforeach()
  endif()
endforeach()
endforeach()
list(APPEND CUTEAFD_V41_FP8_OBJECTS "${CUTEAFD_V41_FP8_DIR}/v41_hc_project.o")
list(APPEND CUTEAFD_V41_FP8_HEADERS "${CUTEAFD_V41_FP8_DIR}/v41_hc_project.h")
set(CUTEAFD_NARROW_ENV "CUTEAFD_EXPORT_NARROW_AOT=0")
if(CUTEAFD_ENABLE_V41_NARROW_AOT)
  set(CUTEAFD_NARROW_ENV "CUTEAFD_EXPORT_NARROW_AOT=1")
endif()
set(CUTEAFD_HC_LAGGED_ENV "CUTEAFD_EXPORT_HC_LAGGED=0")
if(CUTEAFD_ENABLE_V41_HC_LAGGED_AOT)
  set(CUTEAFD_HC_LAGGED_ENV "CUTEAFD_EXPORT_HC_LAGGED=1")
  list(APPEND CUTEAFD_V41_FP8_OBJECTS "${CUTEAFD_V41_FP8_DIR}/v41_hc_lagged.o")
  list(APPEND CUTEAFD_V41_FP8_HEADERS "${CUTEAFD_V41_FP8_DIR}/v41_hc_lagged.h")
endif()
add_custom_command(
  OUTPUT "${CUTEAFD_V41_FP8_DIR}/v41_fp8.json" "${CUTEAFD_V41_FP8_DIR}/v41_fp8_variants.h"
    ${CUTEAFD_V41_FP8_OBJECTS} ${CUTEAFD_V41_FP8_HEADERS}
  COMMAND ${CUTEAFD_SPARKINFER_VERIFY_COMMAND}
  COMMAND "${CMAKE_COMMAND}" -E env ${CUTEAFD_SPARKINFER_PYTHON_ENV} ${CUTEAFD_HC_LAGGED_ENV} ${CUTEAFD_NARROW_ENV}
    ${CUTEAFD_EXPORT_LAUNCHER} "${Python3_EXECUTABLE}"
    "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/aot/export_b12x_v41_fp8_aot.py"
    --output-dir "${CUTEAFD_V41_FP8_DIR}"
  DEPENDS "${CUTEAFD_V41_CAPACITY_SOURCE}"
    "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/lib/v41_capacities.py" "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/aot/export_b12x_v41_fp8_aot.py"
    ${CUTEAFD_SPARKINFER_PROVENANCE_INPUTS} ${CUTEAFD_SPARKINFER_EXPORT_INPUTS}
  COMMENT "Exporting native V4.1 K32 quantization and FP8 projections"
  VERBATIM
)
add_custom_target(cuteafd_v41_fp8_export DEPENDS "${CUTEAFD_V41_FP8_DIR}/v41_fp8.json" "${CUTEAFD_V41_FP8_DIR}/v41_fp8_variants.h"
  ${CUTEAFD_V41_FP8_OBJECTS} ${CUTEAFD_V41_FP8_HEADERS})
add_dependencies(cuteafd_v41_fp8_export cuteafd_verify_sparkinfer_source)
set_source_files_properties(${CUTEAFD_V41_FP8_OBJECTS} PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
list(APPEND CUTEAFD_NATIVE_SOURCES ${CUTEAFD_V41_FP8_OBJECTS} families/deepseek_v41/src/v41_fp8.cc families/deepseek_v41/cuda/v41_fp8.cu)

# Independent scorer export keeps its pointer ABI auditable.
set(CUTEAFD_V41_INDEX_DIR "${CMAKE_CURRENT_BINARY_DIR}/v41_index")
add_custom_command(
  OUTPUT "${CUTEAFD_V41_INDEX_DIR}/v41_index_score.json" "${CUTEAFD_V41_INDEX_DIR}/v41_index_score.h" "${CUTEAFD_V41_INDEX_DIR}/v41_index_score.o"
  COMMAND ${CUTEAFD_SPARKINFER_VERIFY_COMMAND}
  COMMAND "${CMAKE_COMMAND}" -E env ${CUTEAFD_SPARKINFER_PYTHON_ENV}
    ${CUTEAFD_EXPORT_LAUNCHER} "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/aot/export_b12x_v41_index_aot.py"
    --output-dir "${CUTEAFD_V41_INDEX_DIR}"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/aot/export_b12x_v41_index_aot.py"
    ${CUTEAFD_SPARKINFER_PROVENANCE_INPUTS} ${CUTEAFD_SPARKINFER_EXPORT_INPUTS}
  VERBATIM)
add_custom_target(cuteafd_v41_index_export DEPENDS "${CUTEAFD_V41_INDEX_DIR}/v41_index_score.json")
add_dependencies(cuteafd_v41_index_export cuteafd_verify_sparkinfer_source)
set_source_files_properties("${CUTEAFD_V41_INDEX_DIR}/v41_index_score.o" PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
list(APPEND CUTEAFD_NATIVE_SOURCES "${CUTEAFD_V41_INDEX_DIR}/v41_index_score.o" families/deepseek_v41/src/v41_index_score.cc)
