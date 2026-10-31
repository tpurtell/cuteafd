# Each native architecture builds a self-contained set of loadable modules.
# Kernels remain unloaded unless an EXL3 checkpoint is selected at runtime.
if(NOT CUTEAFD_ENABLE_CUDA)
  message(FATAL_ERROR "EXL3 AOT requires CUDA")
endif()
if(CUTEAFD_CUDA_ARCHITECTURES STREQUAL "120" OR CUTEAFD_CUDA_ARCHITECTURES STREQUAL "120f")
  set(CUTEAFD_EXL3_ROLE coordinator)
  set(CUTEAFD_EXL3_LAYOUTS rtx-tp1 rtx-tp2)
elseif(CUTEAFD_CUDA_ARCHITECTURES STREQUAL "121")
  set(CUTEAFD_EXL3_ROLE spark)
  set(CUTEAFD_EXL3_LAYOUTS tp4-rank0 tp4-rank1 tp4-rank2 tp4-rank3)
else()
  message(FATAL_ERROR "EXL3 AOT requires a single native SM120 or SM121 target")
endif()
set(CUTEAFD_EXL3_CAPACITIES "1;16;80;256;1024;2048;2752;4096" CACHE STRING "EXL3 batch capacities to package")
set(CUTEAFD_EXL3_TILES "" CACHE STRING "Per-profile EXL3 tiles, for example tp3-width768=16+80:64,256,64,256 or tp3-width768=all:128,128,128,128")
# Spark packages include every approved whole-block topology.
if(CUTEAFD_EXL3_ROLE STREQUAL "spark")
  list(APPEND CUTEAFD_EXL3_LAYOUTS tp2-rank0 tp2-rank1 tp3-rank0 tp3-rank1 tp3-rank2)
endif()
set(CUTEAFD_EXL3_TILE_ARGS)
foreach(tile IN LISTS CUTEAFD_EXL3_TILES)
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
list(JOIN CUTEAFD_EXL3_CAPACITIES "," CUTEAFD_EXL3_CAPACITIES_ARG)
list(GET CUDAToolkit_INCLUDE_DIRS 0 CUTEAFD_EXL3_CUDA_INCLUDE)
set(CUTEAFD_EXL3_TOOL "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/aot/package_exl3_aot.py")
# Large prefill export arenas must not overlap other GPU compiler jobs.
get_property(CUTEAFD_EXL3_PREDECESSORS DIRECTORY PROPERTY BUILDSYSTEM_TARGETS)
list(FILTER CUTEAFD_EXL3_PREDECESSORS INCLUDE REGEX "_export$")
# Family exports must also serialize against each other: chain each family
# on the previous family's package manifest.
set(CUTEAFD_EXL3_FAMILY_MANIFESTS)
set(CUTEAFD_EXL3_FAMILY_CHAIN ${CUTEAFD_EXL3_PREDECESSORS})
# Other expert geometries: CUTEAFD_EXPERT_FAMILIES entries FAMILY:exl3-kTIERS
# (for example dsv4p:exl3-k23 for DeepSeek V4 Pro EXL3 K2) each build one
# exl3-FAMILY-kTIERS package with the same layouts, sliced from the family's own
# intermediate size; the daemon selects it from the checkpoint geometry.
foreach(entry IN LISTS CUTEAFD_EXPERT_FAMILIES)
  if(NOT entry MATCHES ":exl3-k")
    continue()
  endif()
  if(NOT entry MATCHES "^(dsv4p|glm|glmf|qwen4|kimik3|step5):exl3-k([2-6][2-6])$")
    message(FATAL_ERROR "EXL3 expert family ${entry} must be (dsv4p|glm|glmf|qwen4|kimik3|step5):exl3-k<tiers>, for example dsv4p:exl3-k23")
  endif()
  set(CUTEAFD_EXL3_GEOMETRY "${CMAKE_MATCH_1}")
  set(CUTEAFD_EXL3_FAMILY_TAG "${CMAKE_MATCH_2}")
  string(REGEX REPLACE "([2-6])" "\\1;" CUTEAFD_EXL3_FAMILY_TIERS "${CUTEAFD_EXL3_FAMILY_TAG}")
  list(FILTER CUTEAFD_EXL3_FAMILY_TIERS EXCLUDE REGEX "^$")
  set(CUTEAFD_EXL3_COMPACT_ARGS)
  list(LENGTH CUTEAFD_EXL3_FAMILY_TIERS CUTEAFD_EXL3_TIER_COUNT)
  if(CUTEAFD_EXL3_ROLE STREQUAL "spark" AND CUTEAFD_EXL3_TIER_COUNT EQUAL 2)
    set(CUTEAFD_EXL3_COMPACT_ARGS --compact-companions)
  endif()
  set(CUTEAFD_EXL3_PACKAGE "${CMAKE_CURRENT_BINARY_DIR}/exl3-${CUTEAFD_EXL3_GEOMETRY}-k${CUTEAFD_EXL3_FAMILY_TAG}")
  set(CUTEAFD_EXL3_CONFIG_STAMP "${CMAKE_CURRENT_BINARY_DIR}/exl3_${CUTEAFD_EXL3_GEOMETRY}_k${CUTEAFD_EXL3_FAMILY_TAG}_config.stamp")
  set(CUTEAFD_EXL3_GEOMETRY_LAYOUTS ${CUTEAFD_EXL3_LAYOUTS})
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
  set(CUTEAFD_EXL3_PROFILE_ARGS)
  # Entries are COUNT, FAMILY=COUNT or FAMILY:FORMAT=COUNT; a
  # family none of them names keeps its packaged default layouts.
  set(requested_counts)
  foreach(count IN LISTS CUTEAFD_GENERIC_SPARK_COUNTS)
    if(count MATCHES "^([a-z0-9]+):([a-z0-9]+)=(.*)$")
      if(NOT CMAKE_MATCH_1 STREQUAL CUTEAFD_EXL3_GEOMETRY OR NOT CMAKE_MATCH_2 STREQUAL "exl3")
        continue()
      endif()
      set(count "${CMAKE_MATCH_3}")
    elseif(count MATCHES "^([a-z0-9]+)=(.*)$")
      if(NOT CMAKE_MATCH_1 STREQUAL CUTEAFD_EXL3_GEOMETRY)
        continue()
      endif()
      set(count "${CMAKE_MATCH_2}")
    endif()
    list(APPEND requested_counts "${count}")
  endforeach()
  if(CUTEAFD_EXL3_GEOMETRY STREQUAL "kimik3")
    if(CUTEAFD_EXL3_ROLE STREQUAL "spark")
      foreach(count IN LISTS requested_counts)
        if(NOT count STREQUAL "6")
          message(FATAL_ERROR "kimik3 EXL3 supports TP6 only")
        endif()
      endforeach()
      set(requested_counts 6)
    else()
      set(CUTEAFD_EXL3_GEOMETRY_LAYOUTS rtx-tp1)
    endif()
  endif()
  if(CUTEAFD_EXL3_ROLE STREQUAL "spark" AND NOT "${requested_counts}" STREQUAL "")
    if(CUTEAFD_EXL3_GEOMETRY STREQUAL "qwen4")
      set(intermediate_blocks 5)
    elseif(CUTEAFD_EXL3_GEOMETRY STREQUAL "dsv4p" OR CUTEAFD_EXL3_GEOMETRY STREQUAL "kimik3")
      set(intermediate_blocks 24)
    elseif(CUTEAFD_EXL3_GEOMETRY STREQUAL "step5")
      set(intermediate_blocks 12)
    else()
      set(intermediate_blocks 16)
    endif()
    set(CUTEAFD_EXL3_GEOMETRY_LAYOUTS)
    set(requested_profiles)
    foreach(count IN LISTS requested_counts)
      if(NOT count MATCHES "^[1-6]$" OR count GREATER intermediate_blocks)
        message(FATAL_ERROR "${CUTEAFD_EXL3_GEOMETRY}: TP${count} requires a nonempty H128 slice on each of 1..6 ranks")
      endif()
      math(EXPR last_rank "${count} - 1")
      foreach(rank RANGE 0 ${last_rank})
        list(APPEND CUTEAFD_EXL3_GEOMETRY_LAYOUTS "tp${count}-rank${rank}")
        math(EXPR blocks "${intermediate_blocks} / ${count}")
        math(EXPR remainder "${intermediate_blocks} % ${count}")
        if(rank LESS remainder)
          math(EXPR blocks "${blocks} + 1")
        endif()
        math(EXPR width "${blocks} * 128")
        list(APPEND requested_profiles "tp${count}-width${width}")
      endforeach()
    endforeach()
    list(REMOVE_DUPLICATES CUTEAFD_EXL3_GEOMETRY_LAYOUTS)
    list(REMOVE_DUPLICATES requested_profiles)
    foreach(profile IN LISTS requested_profiles)
      list(APPEND CUTEAFD_EXL3_PROFILE_ARGS --profile "${profile}")
    endforeach()
  endif()
  list(JOIN CUTEAFD_EXL3_GEOMETRY_LAYOUTS "," CUTEAFD_EXL3_GEOMETRY_REQUIRE)
  # Kimi K3 wave merging hands the RTX expert kernel one <=2752-row unit or
  # two or three merged units, so its coordinator package carries the 5504 and
  # 8256 extents; Spark ranks keep the per-unit ladder.
  set(CUTEAFD_EXL3_GEOMETRY_CAPACITY_LIST ${CUTEAFD_EXL3_CAPACITIES})
  if(CUTEAFD_EXL3_GEOMETRY STREQUAL "kimik3" AND CUTEAFD_EXL3_ROLE STREQUAL "coordinator")
    list(APPEND CUTEAFD_EXL3_GEOMETRY_CAPACITY_LIST 5504 8256)
    list(REMOVE_DUPLICATES CUTEAFD_EXL3_GEOMETRY_CAPACITY_LIST)
  endif()
  list(JOIN CUTEAFD_EXL3_GEOMETRY_CAPACITY_LIST "," CUTEAFD_EXL3_GEOMETRY_CAPACITIES_ARG)
  string(JOIN "|" CUTEAFD_EXL3_CONFIG_KEY "geometry=${CUTEAFD_EXL3_GEOMETRY}" "role=${CUTEAFD_EXL3_ROLE}"
    "layouts=${CUTEAFD_EXL3_GEOMETRY_REQUIRE}" "capacities=${CUTEAFD_EXL3_GEOMETRY_CAPACITY_LIST}"
    "tiers=${CUTEAFD_EXL3_FAMILY_TIERS}" "compact=${CUTEAFD_EXL3_COMPACT_ARGS}" "tiles=${CUTEAFD_EXL3_TILES}")
  file(GENERATE OUTPUT "${CUTEAFD_EXL3_CONFIG_STAMP}" CONTENT "${CUTEAFD_EXL3_CONFIG_KEY}\n")
  add_custom_command(
    OUTPUT "${CUTEAFD_EXL3_PACKAGE}/manifest.json"
    COMMAND ${CUTEAFD_SPARKINFER_VERIFY_COMMAND}
    COMMAND "${CMAKE_COMMAND}" -E env ${CUTEAFD_SPARKINFER_PYTHON_ENV}
      "${Python3_EXECUTABLE}" "${CUTEAFD_EXL3_TOOL}" build
      --role "${CUTEAFD_EXL3_ROLE}" --geometry "${CUTEAFD_EXL3_GEOMETRY}"
      --capacities "${CUTEAFD_EXL3_GEOMETRY_CAPACITIES_ARG}"
      --bits ${CUTEAFD_EXL3_FAMILY_TIERS}
      --require-layout "${CUTEAFD_EXL3_GEOMETRY_REQUIRE}" ${CUTEAFD_EXL3_PROFILE_ARGS}
      ${CUTEAFD_EXL3_COMPACT_ARGS} ${CUTEAFD_EXL3_TILE_ARGS}
      --build-dir "${CMAKE_CURRENT_BINARY_DIR}/exl3_exports/${CUTEAFD_EXL3_GEOMETRY}-k${CUTEAFD_EXL3_FAMILY_TAG}"
      --output "${CUTEAFD_EXL3_PACKAGE}"
      --cxx "${CMAKE_CXX_COMPILER}" --cuda-include "${CUTEAFD_EXL3_CUDA_INCLUDE}"
      --cuda-libdir "$<TARGET_FILE_DIR:CUDA::cudart>"
      --cuda-driver "$<TARGET_FILE:CUDA::cuda_driver>"
      --runtime "${CUTEAFD_B12X_AOT_RUNTIME_LIBRARY}"
    DEPENDS "${CUTEAFD_EXL3_TOOL}" "${CUTEAFD_EXL3_CONFIG_STAMP}"
      "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/lib/_export_jobs.py"
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
add_custom_target(cuteafd_exl3_export DEPENDS ${CUTEAFD_EXL3_FAMILY_MANIFESTS})
add_dependencies(cuteafd_exl3_export cuteafd_verify_sparkinfer_source)
