# Each native architecture builds a self-contained set of loadable modules.
# Kernels remain unloaded unless an EXL3 checkpoint is selected at runtime.
if(NOT CUTEAFD_ENABLE_CUDA)
  message(FATAL_ERROR "EXL3 AOT requires CUDA")
endif()
if(CUTEAFD_CUDA_ARCHITECTURES STREQUAL "120" OR CUTEAFD_CUDA_ARCHITECTURES STREQUAL "120f")
  set(CUTEAFD_EXL3_ROLE coordinator)
  set(CUTEAFD_EXL3_LAYOUTS rtx-tp1 rtx-tp2 dspark)
elseif(CUTEAFD_CUDA_ARCHITECTURES STREQUAL "121")
  set(CUTEAFD_EXL3_ROLE spark)
  set(CUTEAFD_EXL3_LAYOUTS tp4-rank0 tp4-rank1 tp4-rank2 tp4-rank3)
else()
  message(FATAL_ERROR "EXL3 AOT requires a single native SM120 or SM121 target")
endif()
set(CUTEAFD_V41_EXL3_CAPACITIES "1;16;80;256;1024;4096" CACHE STRING "EXL3 batch capacities to package")
# V7 ships two decoder families side by side: the uniform K=2 raw
# publication family (resident tiers [2,3]) and the staged K3.25 family
# ([3,4]). Each family lands in its own sibling package directory
# (exl3-k23, exl3-k34) and the daemon selects by checkpoint tiers.
set(CUTEAFD_V41_EXL3_BIT_FAMILIES "2,3;3,4" CACHE STRING "EXL3 decoder tier families to package; one comma-joined tier list per family")
set(CUTEAFD_V41_EXL3_BITS "" CACHE STRING "Single EXL3 decoder tier family override (replaces CUTEAFD_V41_EXL3_BIT_FAMILIES)")
option(CUTEAFD_V41_EXL3_PAIRED_TP4 "Build paired H128 ownership modules for Spark TP4" OFF)
set(CUTEAFD_V41_EXL3_TILES "" CACHE STRING "Per-profile EXL3 tile overrides for a controlled A/B (for example tp3-width768=all:128,128,128,128, or tp3-width768=16:64,256,64,256)")
set(CUTEAFD_V41_EXL3_RESIDENCY "" CACHE STRING "Explicit paired EXL3 capacity=blocks/SM overrides (for example 80=2)")
if(CUTEAFD_V41_EXL3_BITS)
  list(JOIN CUTEAFD_V41_EXL3_BITS "," CUTEAFD_V41_EXL3_BITS_JOINED)
  set(CUTEAFD_V41_EXL3_BIT_FAMILIES "${CUTEAFD_V41_EXL3_BITS_JOINED}")
endif()
set(CUTEAFD_EXL3_LAYOUT_ARGS)
if(CUTEAFD_V41_EXL3_PAIRED_TP4)
  list(LENGTH CUTEAFD_V41_EXL3_BIT_FAMILIES CUTEAFD_EXL3_FAMILY_COUNT)
  if(NOT CUTEAFD_EXL3_FAMILY_COUNT EQUAL 1)
    message(FATAL_ERROR "Paired EXL3 TP4 requires exactly one decoder tier family")
  endif()
  list(GET CUTEAFD_V41_EXL3_BIT_FAMILIES 0 CUTEAFD_EXL3_PAIRED_FAMILY)
  string(REPLACE "," ";" CUTEAFD_EXL3_PAIRED_TIERS "${CUTEAFD_EXL3_PAIRED_FAMILY}")
  list(LENGTH CUTEAFD_EXL3_PAIRED_TIERS CUTEAFD_EXL3_TIER_COUNT)
  if(NOT CUTEAFD_EXL3_ROLE STREQUAL "spark" OR NOT CUTEAFD_EXL3_TIER_COUNT EQUAL 2)
    message(FATAL_ERROR "Paired EXL3 TP4 requires SM121 and exactly two decoder tiers")
  endif()
  list(APPEND CUTEAFD_EXL3_LAYOUT_ARGS --paired-tp4)
elseif(CUTEAFD_V41_EXL3_RESIDENCY)
  message(FATAL_ERROR "EXL3 residency overrides require paired TP4")
endif()
# Disjoint Spark packages serve TP4, equal-width TP2 and exact TP3 ownership: a
# release image must be able to answer every approved topology. Paired H128
# packages remain TP4-only, so they declare neither TP2 nor TP3 byproducts.
if(CUTEAFD_EXL3_ROLE STREQUAL "spark" AND NOT CUTEAFD_V41_EXL3_PAIRED_TP4)
  list(APPEND CUTEAFD_EXL3_LAYOUTS tp2-rank0 tp2-rank1 tp3-rank0 tp3-rank1 tp3-rank2)
endif()
set(CUTEAFD_EXL3_OVERRIDE_CAPACITIES)
foreach(override IN LISTS CUTEAFD_V41_EXL3_RESIDENCY)
  if(NOT override MATCHES "^([1-9][0-9]*)=([12])$")
    message(FATAL_ERROR "EXL3 residency must be capacity=1 or capacity=2")
  endif()
  set(capacity "${CMAKE_MATCH_1}")
  if(NOT capacity IN_LIST CUTEAFD_V41_EXL3_CAPACITIES OR capacity IN_LIST CUTEAFD_EXL3_OVERRIDE_CAPACITIES)
    message(FATAL_ERROR "EXL3 residency requires a selected, nonduplicate capacity")
  endif()
  list(APPEND CUTEAFD_EXL3_OVERRIDE_CAPACITIES "${capacity}")
  list(APPEND CUTEAFD_EXL3_LAYOUT_ARGS --residency "${override}")
endforeach()
set(CUTEAFD_EXL3_TILE_ARGS)
foreach(tile IN LISTS CUTEAFD_V41_EXL3_TILES)
  if(NOT tile MATCHES "^([A-Za-z0-9._+-]+)=(all|[0-9]+(\\+[0-9]+)*):([0-9]+,[0-9]+,[0-9]+,[0-9]+)$")
    message(FATAL_ERROR "EXL3 tile override must be PROFILE=CAPACITIES:FC1_K,FC1_N,FC2_K,FC2_N: ${tile}")
  endif()
  list(APPEND CUTEAFD_EXL3_TILE_ARGS --tile "${tile}")
endforeach()
# The package must contain every layout this build advertises. Recording the
# request in the manifest is what lets a partial Spark export fail the build (and a
# v10 image verify) instead of shipping a package the launcher cannot serve.
list(JOIN CUTEAFD_EXL3_LAYOUTS "," CUTEAFD_EXL3_REQUIRE_LAYOUTS)
set(CUTEAFD_EXL3_REQUIRE_ARGS --require-layout "${CUTEAFD_EXL3_REQUIRE_LAYOUTS}")
# A Make-based tree never restates a custom command's command line, so changing
# capacities, layouts or tiles would otherwise reuse a warm package. This generated
# file exists only as a dependency, exactly like the expert width stamp:
# `file(GENERATE)` rewrites it only when its content changes, so the timestamp moves
# precisely when the resolved configuration for this family moves.
list(JOIN CUTEAFD_V41_EXL3_CAPACITIES "," CUTEAFD_EXL3_CAPACITIES_ARG)
list(GET CUDAToolkit_INCLUDE_DIRS 0 CUTEAFD_EXL3_CUDA_INCLUDE)
set(CUTEAFD_EXL3_TOOL "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/aot/package_exl3_aot.py")
# Large prefill export arenas must not overlap other GPU compiler jobs.
get_property(CUTEAFD_EXL3_PREDECESSORS DIRECTORY PROPERTY BUILDSYSTEM_TARGETS)
list(FILTER CUTEAFD_EXL3_PREDECESSORS INCLUDE REGEX "_export$")
# Family exports must also serialize against each other: chain each family
# on the previous family's package manifest.
set(CUTEAFD_EXL3_FAMILY_MANIFESTS)
set(CUTEAFD_EXL3_FAMILY_CHAIN ${CUTEAFD_EXL3_PREDECESSORS})
foreach(family IN LISTS CUTEAFD_V41_EXL3_BIT_FAMILIES)
  string(REPLACE "," "" CUTEAFD_EXL3_FAMILY_TAG "${family}")
  string(REPLACE "," ";" CUTEAFD_EXL3_FAMILY_TIERS "${family}")
  if(NOT CUTEAFD_EXL3_FAMILY_TAG MATCHES "^[0-9]+$")
    message(FATAL_ERROR "EXL3 tier family '${family}' must be comma-joined integers")
  endif()
  set(CUTEAFD_EXL3_PACKAGE "${CMAKE_CURRENT_BINARY_DIR}/exl3-k${CUTEAFD_EXL3_FAMILY_TAG}")
  string(JOIN "|" CUTEAFD_EXL3_CONFIG_KEY "role=${CUTEAFD_EXL3_ROLE}"
    "layouts=${CUTEAFD_EXL3_REQUIRE_LAYOUTS}" "capacities=${CUTEAFD_V41_EXL3_CAPACITIES}"
    "tiers=${family}" "args=${CUTEAFD_EXL3_LAYOUT_ARGS}" "tiles=${CUTEAFD_EXL3_TILE_ARGS}")
  set(CUTEAFD_EXL3_CONFIG_STAMP "${CMAKE_CURRENT_BINARY_DIR}/v41_exl3_k${CUTEAFD_EXL3_FAMILY_TAG}_config.stamp")
  file(GENERATE OUTPUT "${CUTEAFD_EXL3_CONFIG_STAMP}" CONTENT "${CUTEAFD_EXL3_CONFIG_KEY}\n")
  set(CUTEAFD_EXL3_BYPRODUCTS)
  foreach(layout IN LISTS CUTEAFD_EXL3_LAYOUTS)
    foreach(rows IN LISTS CUTEAFD_V41_EXL3_CAPACITIES)
      foreach(name v41_exl3.json trellis_lut.bin libcuteafd_exl3.so)
        list(APPEND CUTEAFD_EXL3_BYPRODUCTS "${CUTEAFD_EXL3_PACKAGE}/${layout}/m${rows}/${name}")
      endforeach()
    endforeach()
  endforeach()
  add_custom_command(
    OUTPUT "${CUTEAFD_EXL3_PACKAGE}/manifest.json"
    BYPRODUCTS ${CUTEAFD_EXL3_BYPRODUCTS}
    COMMAND ${CUTEAFD_SPARKINFER_VERIFY_COMMAND}
    COMMAND "${CMAKE_COMMAND}" -E env ${CUTEAFD_SPARKINFER_PYTHON_ENV}
      "${Python3_EXECUTABLE}" "${CUTEAFD_EXL3_TOOL}" build
      --role "${CUTEAFD_EXL3_ROLE}" --capacities "${CUTEAFD_EXL3_CAPACITIES_ARG}"
      --bits ${CUTEAFD_EXL3_FAMILY_TIERS}
      ${CUTEAFD_EXL3_LAYOUT_ARGS} ${CUTEAFD_EXL3_REQUIRE_ARGS} ${CUTEAFD_EXL3_TILE_ARGS}
      --build-dir "${CMAKE_CURRENT_BINARY_DIR}/v41_exl3_exports/k${CUTEAFD_EXL3_FAMILY_TAG}"
      --output "${CUTEAFD_EXL3_PACKAGE}"
      --cxx "${CMAKE_CXX_COMPILER}" --cuda-include "${CUTEAFD_EXL3_CUDA_INCLUDE}"
      --cuda-libdir "$<TARGET_FILE_DIR:CUDA::cudart>"
      --cuda-driver "$<TARGET_FILE:CUDA::cuda_driver>"
      --runtime "${CUTEAFD_B12X_AOT_RUNTIME_LIBRARY}"
    DEPENDS "${CUTEAFD_EXL3_TOOL}" "${CUTEAFD_EXL3_CONFIG_STAMP}"
      "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/aot/export_b12x_exl3_aot.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/aot/export_b12x_exl3_routes_aot.py"
      ${CUTEAFD_SPARKINFER_PROVENANCE_INPUTS} ${CUTEAFD_SPARKINFER_EXPORT_INPUTS}
      ${CUTEAFD_EXL3_FAMILY_CHAIN}
    COMMENT "Building native EXL3 modules and verified runtime package (tiers ${family})"
    VERBATIM
  )
  list(APPEND CUTEAFD_EXL3_FAMILY_MANIFESTS "${CUTEAFD_EXL3_PACKAGE}/manifest.json")
  set(CUTEAFD_EXL3_FAMILY_CHAIN "${CUTEAFD_EXL3_PACKAGE}/manifest.json")
endforeach()
# Other expert geometries: CUTEAFD_EXPERT_FAMILIES entries FAMILY:exl3-kTIERS
# (for example dsv4p:exl3-k23 for DeepSeek V4 Pro EXL3 K2) each build one
# exl3-FAMILY-kTIERS package with the same layouts, sliced from the family's own
# intermediate size; the daemon selects it from the checkpoint geometry.
foreach(entry IN LISTS CUTEAFD_EXPERT_FAMILIES)
  if(NOT entry MATCHES ":exl3-k")
    continue()
  endif()
  if(NOT entry MATCHES "^(dsv4f|dsv4p|glm|glmf|qwen4):exl3-k([2-5][2-5]+)$")
    message(FATAL_ERROR "EXL3 expert family ${entry} must be (dsv4f|dsv4p|glm|glmf|qwen4):exl3-k<tiers>, for example dsv4p:exl3-k23")
  endif()
  if(CUTEAFD_V41_EXL3_PAIRED_TP4)
    message(FATAL_ERROR "Paired EXL3 TP4 packages exist only for DeepSeek V4.1")
  endif()
  set(CUTEAFD_EXL3_GEOMETRY "${CMAKE_MATCH_1}")
  set(CUTEAFD_EXL3_FAMILY_TAG "${CMAKE_MATCH_2}")
  string(REGEX REPLACE "([2-5])" "\\1;" CUTEAFD_EXL3_FAMILY_TIERS "${CUTEAFD_EXL3_FAMILY_TAG}")
  list(FILTER CUTEAFD_EXL3_FAMILY_TIERS EXCLUDE REGEX "^$")
  set(CUTEAFD_EXL3_PACKAGE "${CMAKE_CURRENT_BINARY_DIR}/exl3-${CUTEAFD_EXL3_GEOMETRY}-k${CUTEAFD_EXL3_FAMILY_TAG}")
  set(CUTEAFD_EXL3_CONFIG_STAMP "${CMAKE_CURRENT_BINARY_DIR}/exl3_${CUTEAFD_EXL3_GEOMETRY}_k${CUTEAFD_EXL3_FAMILY_TAG}_config.stamp")
  set(CUTEAFD_EXL3_GEOMETRY_LAYOUTS ${CUTEAFD_EXL3_LAYOUTS})
  list(REMOVE_ITEM CUTEAFD_EXL3_GEOMETRY_LAYOUTS dspark)
  # Qwen's five H128 blocks split into native 384/256 coordinator profiles.
  if(CUTEAFD_EXL3_GEOMETRY STREQUAL "qwen4")
    if("rtx-tp2" IN_LIST CUTEAFD_EXL3_GEOMETRY_LAYOUTS)
      list(REMOVE_ITEM CUTEAFD_EXL3_GEOMETRY_LAYOUTS rtx-tp2)
      list(APPEND CUTEAFD_EXL3_GEOMETRY_LAYOUTS rtx-tp2-rank0 rtx-tp2-rank1)
    endif()
    if(CUTEAFD_EXL3_ROLE STREQUAL "spark")
      list(APPEND CUTEAFD_EXL3_GEOMETRY_LAYOUTS tp1-rank0)
    endif()
  endif()
  # Six Sparks: V4 Pro's 24 H128 blocks (tp6-width512) and the 16 of a 2048
  # intermediate (GLM 5.3, GLM 5.3 Flash: tp6-width384 x4 + tp6-width256 x2).
  if(CUTEAFD_EXL3_ROLE STREQUAL "spark" AND NOT CUTEAFD_EXL3_GEOMETRY STREQUAL "qwen4")
    list(APPEND CUTEAFD_EXL3_GEOMETRY_LAYOUTS tp6-rank0 tp6-rank1 tp6-rank2 tp6-rank3 tp6-rank4 tp6-rank5)
  endif()
  list(JOIN CUTEAFD_EXL3_GEOMETRY_LAYOUTS "," CUTEAFD_EXL3_GEOMETRY_REQUIRE)
  string(JOIN "|" CUTEAFD_EXL3_CONFIG_KEY "geometry=${CUTEAFD_EXL3_GEOMETRY}" "role=${CUTEAFD_EXL3_ROLE}"
    "layouts=${CUTEAFD_EXL3_GEOMETRY_REQUIRE}" "capacities=${CUTEAFD_V41_EXL3_CAPACITIES}"
    "tiers=${CUTEAFD_EXL3_FAMILY_TIERS}")
  file(GENERATE OUTPUT "${CUTEAFD_EXL3_CONFIG_STAMP}" CONTENT "${CUTEAFD_EXL3_CONFIG_KEY}\n")
  add_custom_command(
    OUTPUT "${CUTEAFD_EXL3_PACKAGE}/manifest.json"
    COMMAND ${CUTEAFD_SPARKINFER_VERIFY_COMMAND}
    COMMAND "${CMAKE_COMMAND}" -E env ${CUTEAFD_SPARKINFER_PYTHON_ENV}
      "${Python3_EXECUTABLE}" "${CUTEAFD_EXL3_TOOL}" build
      --role "${CUTEAFD_EXL3_ROLE}" --geometry "${CUTEAFD_EXL3_GEOMETRY}"
      --capacities "${CUTEAFD_EXL3_CAPACITIES_ARG}"
      --bits ${CUTEAFD_EXL3_FAMILY_TIERS}
      --require-layout "${CUTEAFD_EXL3_GEOMETRY_REQUIRE}"
      --build-dir "${CMAKE_CURRENT_BINARY_DIR}/exl3_exports/${CUTEAFD_EXL3_GEOMETRY}-k${CUTEAFD_EXL3_FAMILY_TAG}"
      --output "${CUTEAFD_EXL3_PACKAGE}"
      --cxx "${CMAKE_CXX_COMPILER}" --cuda-include "${CUTEAFD_EXL3_CUDA_INCLUDE}"
      --cuda-libdir "$<TARGET_FILE_DIR:CUDA::cudart>"
      --cuda-driver "$<TARGET_FILE:CUDA::cuda_driver>"
      --runtime "${CUTEAFD_B12X_AOT_RUNTIME_LIBRARY}"
    DEPENDS "${CUTEAFD_EXL3_TOOL}" "${CUTEAFD_EXL3_CONFIG_STAMP}"
      "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/aot/export_b12x_exl3_aot.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/aot/export_b12x_exl3_routes_aot.py"
      ${CUTEAFD_SPARKINFER_PROVENANCE_INPUTS} ${CUTEAFD_SPARKINFER_EXPORT_INPUTS}
      ${CUTEAFD_EXL3_FAMILY_CHAIN}
    COMMENT "Building ${CUTEAFD_EXL3_GEOMETRY} EXL3 modules and verified runtime package (tiers ${CUTEAFD_EXL3_FAMILY_TIERS})"
    VERBATIM
  )
  list(APPEND CUTEAFD_EXL3_FAMILY_MANIFESTS "${CUTEAFD_EXL3_PACKAGE}/manifest.json")
  set(CUTEAFD_EXL3_FAMILY_CHAIN "${CUTEAFD_EXL3_PACKAGE}/manifest.json")
endforeach()
add_custom_target(cuteafd_v41_exl3_export DEPENDS ${CUTEAFD_EXL3_FAMILY_MANIFESTS})
add_dependencies(cuteafd_v41_exl3_export cuteafd_verify_sparkinfer_source)
