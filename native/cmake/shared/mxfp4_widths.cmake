# Unequal native whole-scale-unit ownership reuses one artifact for each adjacent logical width.
# Neither rank nor absolute layer is part of the executable key.
if(NOT CUTEAFD_ENABLE_V41_EXPERT_AOT OR NOT CUTEAFD_V41_EXPERT_ROLE STREQUAL "spark")
  return()
endif()
set(native_width_requests)
foreach(tp IN LISTS CUTEAFD_SPARK_TP_ROLES)
  string(REGEX REPLACE "^tp" "" world "${tp}")
  list(APPEND native_width_requests "v41:${world}")
endforeach()
foreach(entry IN LISTS CUTEAFD_EXPERT_FAMILIES)
  if(entry MATCHES "^(dsv4f|dsv4p):spark_tp([2-8])$")
    list(APPEND native_width_requests "${CMAKE_MATCH_1}:${CMAKE_MATCH_2}")
  endif()
endforeach()
set(native_width_keys)
foreach(request IN LISTS native_width_requests)
  string(REPLACE ":" ";" parts "${request}")
  list(GET parts 0 family)
  list(GET parts 1 world)
  set(artifacts_var "CUTEAFD_NATIVE_ARTIFACTS_${family}_${world}")
  if(NOT DEFINED ${artifacts_var})
    message(FATAL_ERROR "No native Spark artifacts for ${request}")
  endif()
  foreach(role IN LISTS ${artifacts_var})
    if(role MATCHES "^spark_w([0-9]+)$")
      list(APPEND native_width_keys "${family}:${CMAKE_MATCH_1}")
    endif()
  endforeach()
endforeach()
list(REMOVE_DUPLICATES native_width_keys)
foreach(key IN LISTS native_width_keys)
  string(REPLACE ":" ";" parts "${key}")
  list(GET parts 0 family)
  list(GET parts 1 width)
  set(role "spark_w${width}")
  set(dir "${CMAKE_CURRENT_BINARY_DIR}/${family}_${role}_experts")
  set(symbol "cuteafd_${family}_${role}")
  set(variant_header "${family}_${role}_expert_variants.h")
  set(renames "")
  foreach(op info initialize output_kind group_rows bind_scratch initialize_scratch_async launch)
    string(APPEND renames "#define cuteafd_expert_${op} ${symbol}_expert_${op}\n")
  endforeach()
  set(wrapper "${dir}/${family}_${role}_experts.cc")
  file(GENERATE OUTPUT "${wrapper}" CONTENT
"#define CUTEAFD_EXPERT_VARIANTS_HEADER \"${variant_header}\"
${renames}#include \"${CMAKE_CURRENT_SOURCE_DIR}/shared/src/v41_experts.cc\"
")
  set(objects)
  set(headers)
  foreach(rows 1 16 80 256 1024 4096)
    list(APPEND objects "${dir}/${family}_${role}_m${rows}.o")
    list(APPEND headers "${dir}/${family}_${role}_m${rows}.h")
  endforeach()
  # Keep the known arithmetic on unmeasured widths. The shared measured table
  # overrides only explicitly qualified (geometry, hardware, capacity) recipes.
  math(EXPR divisible192 "${width} % 192")
  if(family STREQUAL "v41" AND divisible192 EQUAL 0)
    set(slice 192)
  else()
    set(slice 128)
  endif()
  add_custom_command(
    OUTPUT "${dir}/v41_experts.json" "${dir}/${variant_header}" ${objects} ${headers}
    COMMAND ${CUTEAFD_SPARKINFER_VERIFY_COMMAND}
    COMMAND "${CMAKE_COMMAND}" -E env ${CUTEAFD_SPARKINFER_PYTHON_ENV}
      "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/aot/export_b12x_slices_aot.py"
      --geometry "${family}" --role "${role}" --output-dir "${dir}"
      --rows 1,16,80,256,1024,4096 --width "${slice}" --standard-names
    COMMAND "${CMAKE_COMMAND}" -E copy
      "${dir}/v41_expert_variants.h" "${dir}/${variant_header}"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/aot/export_b12x_slices_aot.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/lib/mxfp4_layouts.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/lib/spark_packages.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/../scripts/build/spark-package-ranges.json"
      "${CMAKE_CURRENT_SOURCE_DIR}/../scripts/build/mxfp4-layouts.json"
      "${CMAKE_CURRENT_SOURCE_DIR}/shared/src/v41_experts.cc"
      ${CUTEAFD_SPARKINFER_PROVENANCE_INPUTS} ${CUTEAFD_SPARKINFER_EXPORT_INPUTS}
    COMMENT "Exporting ${family} adjacent K32 width ${width}"
    VERBATIM)
  add_custom_target(cuteafd_${family}_${role}_experts_export DEPENDS
    "${dir}/v41_experts.json" "${dir}/${variant_header}" ${objects} ${headers})
  add_dependencies(cuteafd_${family}_${role}_experts_export cuteafd_verify_sparkinfer_source)
  set_source_files_properties(${objects} PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
  list(APPEND CUTEAFD_EXPERT_FAMILY_TARGETS "cuteafd_${family}_${role}_experts_export")
  list(APPEND CUTEAFD_EXPERT_FAMILY_INCLUDE_DIRS "${dir}")
  list(APPEND CUTEAFD_NATIVE_SOURCES ${objects} "${wrapper}")
endforeach()
