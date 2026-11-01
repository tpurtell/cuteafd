if(NOT CUTEAFD_ENABLE_CUDA)
  message(FATAL_ERROR "PCM audio AOT requires CUDA")
endif()
if(CUTEAFD_CUDA_ARCHITECTURES MATCHES "^120(f|a)?$")
  set(CUTEAFD_PCM_CAPABILITY 120)
elseif(CUTEAFD_CUDA_ARCHITECTURES MATCHES "^121(a)?$")
  set(CUTEAFD_PCM_CAPABILITY 121)
else()
  message(FATAL_ERROR "PCM audio AOT requires SM120 or SM121")
endif()
set(CUTEAFD_PCM_DIR "${CMAKE_CURRENT_BINARY_DIR}/audio_pcm")
set(CUTEAFD_PCM_STEMS)
foreach(width 256 512 1024 2048)
  math(EXPR intermediate "${width} * 4")
  list(APPEND CUTEAFD_PCM_STEMS causal_conv7_h${width} rms_h${width}
    gelu_h${intermediate} residual_h${width}
    projection_k${width}_n${intermediate} projection_k${intermediate}_n${width})
endforeach()
list(APPEND CUTEAFD_PCM_STEMS rms_h8192 projection_k16_n256 projection_k1536_n512
  projection_k2560_n1024 projection_k4096_n2048 projection_k2048_n8192)
list(REMOVE_DUPLICATES CUTEAFD_PCM_STEMS)
set(CUTEAFD_PCM_OUTPUTS "${CUTEAFD_PCM_DIR}/audio_pcm.json"
  "${CUTEAFD_PCM_DIR}/audio_pcm_support.cc" "${CUTEAFD_PCM_DIR}/cuteafd_pcm_support_internal.h")
foreach(stem IN LISTS CUTEAFD_PCM_STEMS)
  foreach(ext h o)
    list(APPEND CUTEAFD_PCM_OUTPUTS "${CUTEAFD_PCM_DIR}/audio_pcm_${stem}.${ext}")
  endforeach()
  set_source_files_properties("${CUTEAFD_PCM_DIR}/audio_pcm_${stem}.o" PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
  list(APPEND CUTEAFD_NATIVE_SOURCES "${CUTEAFD_PCM_DIR}/audio_pcm_${stem}.o")
endforeach()
add_custom_command(OUTPUT ${CUTEAFD_PCM_OUTPUTS}
  COMMAND ${CUTEAFD_SPARKINFER_VERIFY_COMMAND}
  COMMAND "${CMAKE_COMMAND}" -E env ${CUTEAFD_SPARKINFER_PYTHON_ENV}
    "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/aot/export_b12x_audio_aot.py"
    --output-dir "${CUTEAFD_PCM_DIR}" --capability ${CUTEAFD_PCM_CAPABILITY} --group pcm
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/aot/export_b12x_audio_aot.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/aot/export_b12x_vision_aot.py"
    ${CUTEAFD_SPARKINFER_PROVENANCE_INPUTS} ${CUTEAFD_SPARKINFER_EXPORT_INPUTS}
  COMMENT "Exporting shared BF16 PCM pyramid programs offline" VERBATIM)
set_source_files_properties("${CUTEAFD_PCM_DIR}/audio_pcm_support.cc" PROPERTIES GENERATED TRUE)
add_custom_target(cuteafd_pcm_export DEPENDS ${CUTEAFD_PCM_OUTPUTS})
add_dependencies(cuteafd_pcm_export cuteafd_verify_sparkinfer_source)
list(APPEND CUTEAFD_NATIVE_SOURCES shared/cuda/pcm_audio_runtime.cu "${CUTEAFD_PCM_DIR}/audio_pcm_support.cc")
