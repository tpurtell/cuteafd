# Routed-expert kernel families beyond DeepSeek V4.1 (SM121 Spark shards).
#
# Each CUTEAFD_EXPERT_FAMILIES entry is FAMILY:ROLE, for example `dsv4f:spark`
# (DeepSeek V4 Flash TP4 shard) or `dsv4f:spark_tp2`. The exporter derives the
# role's slice from the family geometry and names every symbol
# `cuteafd_{family}[_{role}]_expert_*`; the runtime selects the family from the
# checkpoint's routed-expert geometry (`ExpertGeometry::family`). Spark roles
# build into the SM121 image, `rtx_backbone` (complete experts) and `rtx_tp2`
# (half-width experts) into the SM120 one; entries for the other architecture are
# skipped so one list serves both builds.
#
# FAMILY:exl3-kTIERS entries (for example `dsv4p:exl3-k23`, `glm:exl3-k45`) are EXL3 packages,
# built by cmake/shared/exl3.cmake; FAMILY:fp8 entries (`mimo:fp8`, `glm:fp8`) are
# exact FP8 packages and FAMILY:nvfp4 entries (`glmf:nvfp4`) ModelOpt NVFP4 ones,
# both built by cmake/shared/fp8_moe.cmake; this file builds only the native families.
set(CUTEAFD_NATIVE_EXPERT_FAMILIES)
foreach(entry IN LISTS CUTEAFD_EXPERT_FAMILIES)
  if(entry MATCHES ":(fp8|nvfp4|nvfp4a4)$")
    continue()
  endif()
  if(NOT entry MATCHES ":exl3-k")
    list(APPEND CUTEAFD_NATIVE_EXPERT_FAMILIES "${entry}")
  elseif(NOT CUTEAFD_ENABLE_EXL3_PACKAGES)
    message(FATAL_ERROR "EXL3 expert family ${entry} requires CUTEAFD_ENABLE_EXL3_PACKAGES=ON")
  endif()
endforeach()
if(NOT CUTEAFD_NATIVE_EXPERT_FAMILIES)
  return()
endif()
if(NOT CUTEAFD_ENABLE_V41_EXPERT_AOT)
  message(FATAL_ERROR "Expert families require the native expert AOT build")
endif()

set(CUTEAFD_EXPERT_FAMILY_TARGETS)
set(CUTEAFD_EXPERT_FAMILY_INCLUDE_DIRS)
set(CUTEAFD_EXPERT_FAMILY_ROWS 1 16 80 256 1024 4096)
# 192-wide slices do not tile these 128-aligned extents (512, 1024); capacity 1
# keeps the narrow decode tile.
set(CUTEAFD_EXPERT_FAMILY_WIDTH "1:64,16:128,80:128,256:128,1024:128,4096:128" CACHE STRING
  "Slice width map for non-V4.1 expert families")
set(expert_ops info initialize output_kind bind_scratch initialize_scratch_async launch)

foreach(entry IN LISTS CUTEAFD_NATIVE_EXPERT_FAMILIES)
  if(NOT entry MATCHES "^(dsv4f|dsv4p):(spark|spark_tp2|rtx_backbone|rtx_tp2)$")
    message(FATAL_ERROR "CUTEAFD_EXPERT_FAMILIES entry ${entry} must be (dsv4f|dsv4p):(spark|spark_tp2|rtx_backbone|rtx_tp2)")
  endif()
  set(family "${CMAKE_MATCH_1}")
  set(role "${CMAKE_MATCH_2}")
  set(device_handles "")
  if(role MATCHES "^rtx_")
    set(wanted coordinator)
    set(device_handles "#define CUTEAFD_EXPERT_PER_DEVICE_HANDLES 1\n")
  else()
    set(wanted spark)
  endif()
  if(NOT CUTEAFD_V41_EXPERT_ROLE STREQUAL wanted)
    continue()
  endif()
  # Symbol names follow the V4.1 roles: the Spark TP4 shard is unsuffixed and
  # the resident coordinator experts are `local`.
  if(role STREQUAL "spark")
    set(symbol "cuteafd_${family}")
  elseif(role STREQUAL "rtx_backbone")
    set(symbol "cuteafd_${family}_local")
  elseif(role STREQUAL "rtx_tp2")
    set(symbol "cuteafd_${family}_tp2")
  else()
    set(symbol "cuteafd_${family}_${role}")
  endif()
  set(dir "${CMAKE_CURRENT_BINARY_DIR}/${family}_${role}_experts")
  set(variant_header "${family}_${role}_expert_variants.h")
  set(stamp "${CMAKE_CURRENT_BINARY_DIR}/${family}_${role}_width.stamp")
  file(GENERATE OUTPUT "${stamp}"
    CONTENT "family=${family}\nrole=${role}\nwidth=${CUTEAFD_EXPERT_FAMILY_WIDTH}\natomic_min_capacity=256\n")
  set(renames "")
  foreach(op IN LISTS expert_ops)
    string(APPEND renames "#define cuteafd_expert_${op} ${symbol}_expert_${op}\n")
  endforeach()
  set(wrapper "${dir}/${family}_${role}_experts.cc")
  file(GENERATE OUTPUT "${wrapper}" CONTENT
"// Generated: ${family} ${role} routed-expert family (native FP8 K32).
#define CUTEAFD_EXPERT_VARIANTS_HEADER \"${variant_header}\"
${device_handles}${renames}#include \"${CMAKE_CURRENT_SOURCE_DIR}/shared/src/v41_experts.cc\"
")
  set(objects)
  set(headers)
  foreach(rows IN LISTS CUTEAFD_EXPERT_FAMILY_ROWS)
    list(APPEND objects "${dir}/${family}_${role}_m${rows}.o")
    list(APPEND headers "${dir}/${family}_${role}_m${rows}.h")
  endforeach()
  list(JOIN CUTEAFD_EXPERT_FAMILY_ROWS "," rows_arg)
  add_custom_command(
    OUTPUT "${dir}/v41_experts.json" "${dir}/${variant_header}" ${objects} ${headers}
    COMMAND ${CUTEAFD_SPARKINFER_VERIFY_COMMAND}
    COMMAND "${CMAKE_COMMAND}" -E env ${CUTEAFD_SPARKINFER_PYTHON_ENV}
      "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/aot/export_b12x_slices_aot.py"
      --output-dir "${dir}" --geometry "${family}" --role "${role}"
      --rows "${rows_arg}" --width "${CUTEAFD_EXPERT_FAMILY_WIDTH}"
      --atomic-min-capacity 256 --standard-names
    COMMAND "${CMAKE_COMMAND}" -E copy
      "${dir}/v41_expert_variants.h" "${dir}/${variant_header}"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/aot/export_b12x_slices_aot.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/shared/src/v41_experts.cc"
      "${stamp}"
      ${CUTEAFD_SPARKINFER_PROVENANCE_INPUTS} ${CUTEAFD_SPARKINFER_EXPORT_INPUTS}
    COMMENT "Exporting ${family} ${role} routed-expert kernels"
    VERBATIM
  )
  add_custom_target(cuteafd_${family}_${role}_experts_export DEPENDS
    "${dir}/${variant_header}" "${dir}/v41_experts.json" ${objects} ${headers})
  add_dependencies(cuteafd_${family}_${role}_experts_export cuteafd_verify_sparkinfer_source)
  set_source_files_properties(${objects} PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
  set_source_files_properties("${wrapper}" PROPERTIES GENERATED TRUE)
  list(APPEND CUTEAFD_EXPERT_FAMILY_INCLUDE_DIRS "${dir}")
  list(APPEND CUTEAFD_EXPERT_FAMILY_TARGETS "cuteafd_${family}_${role}_experts_export")
  list(APPEND CUTEAFD_NATIVE_SOURCES ${objects} "${wrapper}")
endforeach()
