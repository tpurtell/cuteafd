if(NOT CUTEAFD_ENABLE_CUDA)
  message(FATAL_ERROR "Vision attention AOT requires CUDA")
endif()
if(CUTEAFD_CUDA_ARCHITECTURES MATCHES "^120(f|a)?$")
  set(CUTEAFD_VISION_CAPABILITY 120)
elseif(CUTEAFD_CUDA_ARCHITECTURES MATCHES "^121(a)?$")
  set(CUTEAFD_VISION_CAPABILITY 121)
else()
  message(FATAL_ERROR "Vision attention AOT requires SM120 or SM121")
endif()
set(CUTEAFD_VISION_ATTENTION_DIR "${CMAKE_CURRENT_BINARY_DIR}/vision_attention")
set(CUTEAFD_VISION_ATTENTION_OUTPUTS)
foreach(dim 64 72)
  foreach(ext json h o)
    list(APPEND CUTEAFD_VISION_ATTENTION_OUTPUTS "${CUTEAFD_VISION_ATTENTION_DIR}/vision_attention_d${dim}.${ext}")
  endforeach()
  set_source_files_properties("${CUTEAFD_VISION_ATTENTION_DIR}/vision_attention_d${dim}.o"
    PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
  list(APPEND CUTEAFD_NATIVE_SOURCES "${CUTEAFD_VISION_ATTENTION_DIR}/vision_attention_d${dim}.o")
endforeach()
add_custom_command(
  OUTPUT ${CUTEAFD_VISION_ATTENTION_OUTPUTS}
  COMMAND ${CUTEAFD_SPARKINFER_VERIFY_COMMAND}
  COMMAND "${CMAKE_COMMAND}" -E env ${CUTEAFD_SPARKINFER_PYTHON_ENV}
    "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/aot/export_b12x_vision_aot.py"
    --output-dir "${CUTEAFD_VISION_ATTENTION_DIR}" --capability ${CUTEAFD_VISION_CAPABILITY}
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/aot/export_b12x_vision_aot.py"
    ${CUTEAFD_SPARKINFER_PROVENANCE_INPUTS} ${CUTEAFD_SPARKINFER_EXPORT_INPUTS}
  COMMENT "Exporting full-image head64/head72 BF16 vision attention offline"
  VERBATIM)
add_custom_target(cuteafd_vision_attention_export DEPENDS ${CUTEAFD_VISION_ATTENTION_OUTPUTS})
add_dependencies(cuteafd_vision_attention_export cuteafd_verify_sparkinfer_source)
list(APPEND CUTEAFD_NATIVE_SOURCES shared/src/vision_attention_aot.cc)
