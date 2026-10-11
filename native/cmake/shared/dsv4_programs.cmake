# Coordinator programs (SM120): b12x.integration.cuteafd exports launched
# through the generic table in shared/src/dsv4_programs.cc. DeepSeek V4
# (CUTEAFD_ENABLE_DSV4_AOT, CUTEAFD_DSV4_GEOMETRY), GLM 5.x
# (CUTEAFD_ENABLE_GLM_AOT, geometry "glm") and MiMo V2 Flash
# (CUTEAFD_ENABLE_MIMO_AOT, geometries CUTEAFD_MIMO_GEOMETRIES: "mimo", V2.6 Pro "mimop") and GLM 5.3 Flash
# (CUTEAFD_ENABLE_GLMF_AOT, geometry "glmf") and Qwen 3.8 Flash Next
# (CUTEAFD_ENABLE_QWEN4_AOT, geometry "qwen4") share one table and manifest.
if(NOT CUTEAFD_CUDA_ARCHITECTURES MATCHES "^120")
  message(FATAL_ERROR "coordinator programs require the SM120 build")
endif()
set(CUTEAFD_PROGRAM_GEOMETRY "")
if(CUTEAFD_ENABLE_DSV4_AOT)
  set(CUTEAFD_PROGRAM_GEOMETRY "${CUTEAFD_DSV4_GEOMETRY}")
endif()
if(CUTEAFD_ENABLE_GLM_AOT)
  # glm = GLM 5.x, glm2 = one GPU of its two-GPU head split.
  string(REPLACE ";" "," glm_geometries "${CUTEAFD_GLM_GEOMETRIES}")
  if(CUTEAFD_PROGRAM_GEOMETRY STREQUAL "")
    set(CUTEAFD_PROGRAM_GEOMETRY "${glm_geometries}")
  else()
    set(CUTEAFD_PROGRAM_GEOMETRY "${CUTEAFD_PROGRAM_GEOMETRY},${glm_geometries}")
  endif()
endif()
if(CUTEAFD_ENABLE_MIMO_AOT)
  # mimo = MiMo V2 Flash, mimop = MiMo V2.6 Pro (family `mimop`).
  string(REPLACE ";" "," mimo_geometries "${CUTEAFD_MIMO_GEOMETRIES}")
  if(CUTEAFD_PROGRAM_GEOMETRY STREQUAL "")
    set(CUTEAFD_PROGRAM_GEOMETRY "${mimo_geometries}")
  else()
    set(CUTEAFD_PROGRAM_GEOMETRY "${CUTEAFD_PROGRAM_GEOMETRY},${mimo_geometries}")
  endif()
endif()
if(CUTEAFD_ENABLE_GLMF_AOT)
  # glmf = GLM 5.3 Flash, glmf2 = one GPU of its two-GPU head split.
  if(CUTEAFD_PROGRAM_GEOMETRY STREQUAL "")
    set(CUTEAFD_PROGRAM_GEOMETRY "glmf,glmf2")
  else()
    set(CUTEAFD_PROGRAM_GEOMETRY "${CUTEAFD_PROGRAM_GEOMETRY},glmf,glmf2")
  endif()
endif()
if(CUTEAFD_ENABLE_QWEN4_AOT)
  if(CUTEAFD_PROGRAM_GEOMETRY STREQUAL "")
    set(CUTEAFD_PROGRAM_GEOMETRY "qwen4")
  else()
    set(CUTEAFD_PROGRAM_GEOMETRY "${CUTEAFD_PROGRAM_GEOMETRY},qwen4")
  endif()
endif()
set(CUTEAFD_DSV4_DIR "${CMAKE_CURRENT_BINARY_DIR}/dsv4_programs")
string(TOLOWER "${CUTEAFD_CONTEXT_SPLIT_AOT}" context_split_aot)
if(NOT context_split_aot MATCHES "^(off|on|only)$")
  message(FATAL_ERROR "CUTEAFD_CONTEXT_SPLIT_AOT must be OFF, ON or ONLY")
endif()
set(CUTEAFD_DSV4_EXPORT_ARGS --geometry "${CUTEAFD_PROGRAM_GEOMETRY}"
  --decode-rows "${CUTEAFD_DSV4_DECODE_ROWS}" --prefill-rows "${CUTEAFD_DSV4_PREFILL_ROWS}"
  --glmf-wide-decode-rows "${CUTEAFD_GLMF_WIDE_DECODE_ROWS}"
  --max-context "${CUTEAFD_DSV4_MAX_CONTEXT}" --context-split "${context_split_aot}")
set(stamp "${CMAKE_CURRENT_BINARY_DIR}/dsv4_programs.stamp")
file(GENERATE OUTPUT "${stamp}" CONTENT "${CUTEAFD_DSV4_EXPORT_ARGS}\n")
# The program list lives in the exporter; objects are collected into one
# archive so CMake need not know each stem.
set(CUTEAFD_DSV4_ARCHIVE "${CUTEAFD_DSV4_DIR}/libcuteafd_dsv4_programs.a")
add_custom_command(
  OUTPUT "${CUTEAFD_DSV4_DIR}/dsv4_programs.h" "${CUTEAFD_DSV4_DIR}/dsv4_programs.json"
    "${CUTEAFD_DSV4_ARCHIVE}"
  COMMAND ${CUTEAFD_SPARKINFER_VERIFY_COMMAND}
  COMMAND "${CMAKE_COMMAND}" -E rm -rf "${CUTEAFD_DSV4_DIR}"
  COMMAND "${CMAKE_COMMAND}" -E env ${CUTEAFD_SPARKINFER_PYTHON_ENV}
    "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/aot/export_b12x_dsv4_aot.py"
    --output-dir "${CUTEAFD_DSV4_DIR}" ${CUTEAFD_DSV4_EXPORT_ARGS}
  COMMAND sh -c "${CMAKE_AR} qcs '${CUTEAFD_DSV4_ARCHIVE}' '${CUTEAFD_DSV4_DIR}'/*.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/aot/export_b12x_dsv4_aot.py" "${stamp}"
    ${CUTEAFD_SPARKINFER_PROVENANCE_INPUTS} ${CUTEAFD_SPARKINFER_EXPORT_INPUTS}
  COMMENT "Exporting coordinator programs (${CUTEAFD_PROGRAM_GEOMETRY})"
  VERBATIM)
add_custom_target(cuteafd_dsv4_programs_export DEPENDS
  "${CUTEAFD_DSV4_DIR}/dsv4_programs.h" "${CUTEAFD_DSV4_ARCHIVE}")
add_dependencies(cuteafd_dsv4_programs_export cuteafd_verify_sparkinfer_source)
set_source_files_properties("${CUTEAFD_DSV4_DIR}/dsv4_programs.h" PROPERTIES GENERATED TRUE)
list(APPEND CUTEAFD_NATIVE_SOURCES shared/src/dsv4_programs.cc)
