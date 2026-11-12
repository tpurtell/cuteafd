if(NOT CUTEAFD_ENABLE_CUDA OR NOT
   (CUTEAFD_CUDA_ARCHITECTURES STREQUAL "120" OR CUTEAFD_CUDA_ARCHITECTURES STREQUAL "120f"))
  message(FATAL_ERROR "V4.1 native attention AOT requires SM120")
endif()
set(CUTEAFD_V41_ATTENTION_DIR "${CMAKE_CURRENT_BINARY_DIR}/v41_attention")
add_custom_command(
  OUTPUT "${CUTEAFD_V41_ATTENTION_DIR}/v41_attention.json"
    "${CUTEAFD_V41_ATTENTION_DIR}/v41_attention.h" "${CUTEAFD_V41_ATTENTION_DIR}/v41_attention.o"
  COMMAND ${CUTEAFD_SPARKINFER_VERIFY_COMMAND}
  COMMAND "${CMAKE_COMMAND}" -E env ${CUTEAFD_SPARKINFER_PYTHON_ENV}
    ${CUTEAFD_EXPORT_CACHE_COMMAND} --output "${CUTEAFD_V41_ATTENTION_DIR}/v41_attention.json" --output "${CUTEAFD_V41_ATTENTION_DIR}/v41_attention.h" --output "${CUTEAFD_V41_ATTENTION_DIR}/v41_attention.o"
    -- "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/aot/export_b12x_v41_attention_aot.py"
    --output-dir "${CUTEAFD_V41_ATTENTION_DIR}"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/aot/export_b12x_v41_attention_aot.py"
    ${CUTEAFD_SPARKINFER_PROVENANCE_INPUTS} ${CUTEAFD_SPARKINFER_EXPORT_INPUTS}
  COMMENT "Exporting direct V4.1 FP4 attention and sink merge"
  VERBATIM)
add_custom_target(cuteafd_v41_attention_export DEPENDS "${CUTEAFD_V41_ATTENTION_DIR}/v41_attention.json"
  "${CUTEAFD_V41_ATTENTION_DIR}/v41_attention.h" "${CUTEAFD_V41_ATTENTION_DIR}/v41_attention.o")
add_dependencies(cuteafd_v41_attention_export cuteafd_verify_sparkinfer_source)
set_source_files_properties("${CUTEAFD_V41_ATTENTION_DIR}/v41_attention.o"
  PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
list(APPEND CUTEAFD_NATIVE_SOURCES "${CUTEAFD_V41_ATTENTION_DIR}/v41_attention.o" families/deepseek_v41/src/v41_attention_aot.cc)

# Local TP2 head geometry is exported separately; live row counts stay dynamic.
add_custom_command(
  OUTPUT "${CUTEAFD_V41_ATTENTION_DIR}/v41_attention_heads32.json"
    "${CUTEAFD_V41_ATTENTION_DIR}/v41_attention_heads32.h" "${CUTEAFD_V41_ATTENTION_DIR}/v41_attention_heads32.o"
  COMMAND ${CUTEAFD_SPARKINFER_VERIFY_COMMAND}
  COMMAND "${CMAKE_COMMAND}" -E env ${CUTEAFD_SPARKINFER_PYTHON_ENV}
    ${CUTEAFD_EXPORT_CACHE_COMMAND} --output "${CUTEAFD_V41_ATTENTION_DIR}/v41_attention_heads32.json" --output "${CUTEAFD_V41_ATTENTION_DIR}/v41_attention_heads32.h" --output "${CUTEAFD_V41_ATTENTION_DIR}/v41_attention_heads32.o"
    -- "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/aot/export_b12x_v41_attention_aot.py"
    --output-dir "${CUTEAFD_V41_ATTENTION_DIR}" --heads 32
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/aot/export_b12x_v41_attention_aot.py"
    ${CUTEAFD_SPARKINFER_PROVENANCE_INPUTS} ${CUTEAFD_SPARKINFER_EXPORT_INPUTS}
  COMMENT "Exporting 32-head direct V4.1 FP4 attention and sink merge"
  VERBATIM)
add_custom_target(cuteafd_v41_attention_heads32_export DEPENDS
  "${CUTEAFD_V41_ATTENTION_DIR}/v41_attention_heads32.json"
  "${CUTEAFD_V41_ATTENTION_DIR}/v41_attention_heads32.h" "${CUTEAFD_V41_ATTENTION_DIR}/v41_attention_heads32.o")
add_dependencies(cuteafd_v41_attention_heads32_export cuteafd_verify_sparkinfer_source)
set_source_files_properties("${CUTEAFD_V41_ATTENTION_DIR}/v41_attention_heads32.o"
  PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
list(APPEND CUTEAFD_NATIVE_SOURCES "${CUTEAFD_V41_ATTENTION_DIR}/v41_attention_heads32.o" families/deepseek_v41/src/v41_attention_heads32_aot.cc)
