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
# Register one geometry and its modules. DIRECTORY points at an existing export
# command (legacy keeps its byte-identical two-object command below). Without it,
# ARGS exports the supplied PROGRAMS as one group in an isolated directory.
function(cuteafd_vision_owner_entry name)
  cmake_parse_arguments(E "WINDOWED" "HEAD_DIM;POINTWISE_GROUP;DIRECTORY" "PROGRAMS;ARGS;POINTWISE_PROGRAMS;POINTWISE_ARGS" ${ARGN})
  if(NOT E_HEAD_DIM OR NOT E_PROGRAMS OR E_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR "Incomplete vision owner entry ${name}")
  endif()
  if(E_DIRECTORY)
    set(directory "${E_DIRECTORY}")
  else()
    set(directory "${CUTEAFD_VISION_ATTENTION_DIR}/${name}")
  endif()
  set(outputs)
  set(programs "[]")
  foreach(stem IN LISTS E_PROGRAMS)
    foreach(ext h o)
      list(APPEND outputs "${directory}/${stem}.${ext}")
    endforeach()
    set_source_files_properties("${directory}/${stem}.o" PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
    set_property(GLOBAL APPEND PROPERTY CUTEAFD_VISION_OWNER_OBJECTS "${directory}/${stem}.o")
    string(JSON size LENGTH "${programs}")
    string(JSON programs SET "${programs}" ${size} "\"${stem}\"")
  endforeach()
  if(NOT E_DIRECTORY)
    add_custom_command(OUTPUT ${outputs}
      COMMAND ${CUTEAFD_SPARKINFER_VERIFY_COMMAND}
      COMMAND "${CMAKE_COMMAND}" -E env ${CUTEAFD_SPARKINFER_PYTHON_ENV}
        "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/aot/export_b12x_vision_aot.py"
        --output-dir "${directory}" --capability ${CUTEAFD_VISION_CAPABILITY} ${E_ARGS}
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/aot/export_b12x_vision_aot.py"
        ${CUTEAFD_SPARKINFER_PROVENANCE_INPUTS} ${CUTEAFD_SPARKINFER_EXPORT_INPUTS}
      COMMENT "Exporting vision owner ${name}" VERBATIM)
  endif()
  if(E_WINDOWED)
    set(windowed true)
  else()
    set(windowed false)
  endif()
  if(E_POINTWISE_PROGRAMS)
    if(NOT E_POINTWISE_GROUP OR NOT E_POINTWISE_ARGS)
      message(FATAL_ERROR "Vision pointwise entry ${name} needs group and exporter arguments")
    endif()
    if(NOT E_POINTWISE_GROUP MATCHES "^[A-Za-z0-9_]+$")
      message(FATAL_ERROR "Invalid vision pointwise group ${E_POINTWISE_GROUP}")
    endif()
    set(pointwise_directory "${CUTEAFD_VISION_ATTENTION_DIR}/pointwise_${E_POINTWISE_GROUP}")
    set(pointwise_outputs)
    foreach(stem IN LISTS E_POINTWISE_PROGRAMS)
      foreach(ext h o)
        list(APPEND pointwise_outputs "${pointwise_directory}/${stem}.${ext}")
      endforeach()
      set_source_files_properties("${pointwise_directory}/${stem}.o" PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
      set_property(GLOBAL APPEND PROPERTY CUTEAFD_VISION_OWNER_OBJECTS "${pointwise_directory}/${stem}.o")
      string(JSON size LENGTH "${programs}")
      string(JSON programs SET "${programs}" ${size}
        "{\"directory\":\"${pointwise_directory}\",\"name\":\"${stem}\"}")
    endforeach()
    string(JOIN ";" pointwise_signature ${E_POINTWISE_PROGRAMS} ${E_POINTWISE_ARGS})
    get_property(existing_signature GLOBAL PROPERTY "CUTEAFD_VISION_POINTWISE_${E_POINTWISE_GROUP}")
    if(existing_signature AND NOT existing_signature STREQUAL pointwise_signature)
      message(FATAL_ERROR "Conflicting vision pointwise group ${E_POINTWISE_GROUP}")
    endif()
    if(NOT existing_signature)
      set_property(GLOBAL PROPERTY "CUTEAFD_VISION_POINTWISE_${E_POINTWISE_GROUP}" "${pointwise_signature}")
      add_custom_command(OUTPUT ${pointwise_outputs}
      COMMAND ${CUTEAFD_SPARKINFER_VERIFY_COMMAND}
      COMMAND "${CMAKE_COMMAND}" -E env ${CUTEAFD_SPARKINFER_PYTHON_ENV}
        "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/aot/export_b12x_vision_aot.py"
        --output-dir "${pointwise_directory}" --capability ${CUTEAFD_VISION_CAPABILITY} ${E_POINTWISE_ARGS}
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/aot/export_b12x_vision_aot.py"
        ${CUTEAFD_SPARKINFER_PROVENANCE_INPUTS} ${CUTEAFD_SPARKINFER_EXPORT_INPUTS}
      COMMENT "Exporting vision owner ${name} pointwise group" VERBATIM)
    endif()
    list(APPEND outputs ${pointwise_outputs})
  endif()
  set(entry "{\"head_dim\":${E_HEAD_DIM},\"windowed\":${windowed},\"pointwise_group\":\"${E_POINTWISE_GROUP}\",\"directory\":\"${directory}\",\"programs\":${programs}}")
  set_property(GLOBAL APPEND PROPERTY CUTEAFD_VISION_OWNER_ENTRIES "${entry}")
  set_property(GLOBAL APPEND PROPERTY CUTEAFD_VISION_OWNER_OUTPUTS ${outputs})
endfunction()

foreach(dim 64 72)
  cuteafd_vision_owner_entry(d${dim} HEAD_DIM ${dim} DIRECTORY "${CUTEAFD_VISION_ATTENTION_DIR}"
    PROGRAMS vision_attention_d${dim})
  foreach(ext h o json)
    list(APPEND CUTEAFD_VISION_ATTENTION_OUTPUTS "${CUTEAFD_VISION_ATTENTION_DIR}/vision_attention_d${dim}.${ext}")
  endforeach()
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
set(muse_exporter "${CMAKE_CURRENT_LIST_DIR}/../../../python/tools/aot/export_b12x_vision_aot.py")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${muse_exporter}")
execute_process(COMMAND "${Python3_EXECUTABLE}" "${muse_exporter}" --list-transformer-programs
  RESULT_VARIABLE muse_inventory_status OUTPUT_VARIABLE muse_transformer_programs
  OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT muse_inventory_status EQUAL 0 OR NOT muse_transformer_programs)
  message(FATAL_ERROR "Cannot read Muse transformer export inventory")
endif()
foreach(mode full windows)
  set(windowed)
  set(windowed_args)
  if(mode STREQUAL "windows")
    set(windowed WINDOWED)
    set(windowed_args --windowed)
  endif()
  cuteafd_vision_owner_entry(d96_${mode} HEAD_DIM 96 ${windowed}
    PROGRAMS vision_attention_d96_${mode}
    ARGS --group attention --head-dim 96 ${windowed_args}
    POINTWISE_GROUP muse
    POINTWISE_PROGRAMS vision_axial_rotary_d96_f32 vision_position_gather_h1536 ${muse_transformer_programs}
    POINTWISE_ARGS --group pointwise --head-dim 96 --hidden 1536 --learned-position --with-transformer)
endforeach()
get_property(owner_outputs GLOBAL PROPERTY CUTEAFD_VISION_OWNER_OUTPUTS)
list(APPEND CUTEAFD_VISION_ATTENTION_OUTPUTS ${owner_outputs})
list(REMOVE_DUPLICATES CUTEAFD_VISION_ATTENTION_OUTPUTS)
get_property(owner_entries GLOBAL PROPERTY CUTEAFD_VISION_OWNER_ENTRIES)
list(JOIN owner_entries ",\n" owner_entries)
file(GENERATE OUTPUT "${CUTEAFD_VISION_ATTENTION_DIR}/entries.json" CONTENT "[${owner_entries}]\n")
set(registry "${CUTEAFD_VISION_ATTENTION_DIR}/vision_owner_entries.h")
add_custom_command(OUTPUT "${registry}"
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/../scripts/build/vision-owner-registry.py"
    --entries "${CUTEAFD_VISION_ATTENTION_DIR}/entries.json" --output "${registry}"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../scripts/build/vision-owner-registry.py"
    "${CUTEAFD_VISION_ATTENTION_DIR}/entries.json" ${CUTEAFD_VISION_ATTENTION_OUTPUTS}
  VERBATIM)
add_custom_target(cuteafd_vision_attention_export DEPENDS "${registry}" ${CUTEAFD_VISION_ATTENTION_OUTPUTS})
add_dependencies(cuteafd_vision_attention_export cuteafd_verify_sparkinfer_source)
get_property(owner_objects GLOBAL PROPERTY CUTEAFD_VISION_OWNER_OBJECTS)
list(REMOVE_DUPLICATES owner_objects)
list(APPEND CUTEAFD_NATIVE_SOURCES shared/src/vision_attention_aot.cc ${owner_objects})
