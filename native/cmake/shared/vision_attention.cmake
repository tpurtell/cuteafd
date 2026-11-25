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
  cmake_parse_arguments(E "WINDOWED" "HEAD_DIM;POINTWISE_GROUP;DIRECTORY" "PROGRAMS;ARGS;POINTWISE_PROGRAMS;POINTWISE_ARGS;BORROW_POINTWISE_GROUP;BORROW_POINTWISE_PROGRAMS" ${ARGN})
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
        ${CUTEAFD_EXPORT_LAUNCHER} "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/aot/export_b12x_vision_aot.py"
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
    set(pointwise_export_args ${E_POINTWISE_ARGS})
    foreach(stem IN LISTS E_BORROW_POINTWISE_PROGRAMS)
      if(NOT stem IN_LIST E_POINTWISE_PROGRAMS)
        message(FATAL_ERROR "Borrowed vision program ${stem} missing from owner inventory")
      endif()
    endforeach()
    foreach(stem IN LISTS E_POINTWISE_PROGRAMS)
      if(stem IN_LIST E_BORROW_POINTWISE_PROGRAMS)
        set(found_borrow_group "")
        foreach(borrow_group IN LISTS E_BORROW_POINTWISE_GROUP)
          get_property(borrowed_programs GLOBAL PROPERTY "CUTEAFD_VISION_GROUP_PROGRAMS_${borrow_group}")
          if(stem IN_LIST borrowed_programs)
            set(found_borrow_group "${borrow_group}")
            break()
          endif()
        endforeach()
        if(NOT found_borrow_group)
          message(FATAL_ERROR "Vision group ${E_BORROW_POINTWISE_GROUP} does not export ${stem}")
        endif()
        get_property(borrowed_directory GLOBAL PROPERTY "CUTEAFD_VISION_PROGRAM_DIR_${stem}")
        if(NOT borrowed_directory)
          set(borrowed_directory "${CUTEAFD_VISION_ATTENTION_DIR}/pointwise_${found_borrow_group}")
        endif()
        string(JSON size LENGTH "${programs}")
        string(JSON programs SET "${programs}" ${size}
          "{\"directory\":\"${borrowed_directory}\",\"name\":\"${stem}\"}")
        list(APPEND pointwise_export_args --omit-program ${stem})
        continue()
      endif()
      set_property(GLOBAL PROPERTY "CUTEAFD_VISION_PROGRAM_DIR_${stem}" "${pointwise_directory}")
      foreach(ext h o)
        list(APPEND pointwise_outputs "${pointwise_directory}/${stem}.${ext}")
      endforeach()
      set_source_files_properties("${pointwise_directory}/${stem}.o" PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
      set_property(GLOBAL APPEND PROPERTY CUTEAFD_VISION_OWNER_OBJECTS "${pointwise_directory}/${stem}.o")
      string(JSON size LENGTH "${programs}")
      string(JSON programs SET "${programs}" ${size}
        "{\"directory\":\"${pointwise_directory}\",\"name\":\"${stem}\"}")
    endforeach()
    set_property(GLOBAL PROPERTY "CUTEAFD_VISION_GROUP_PROGRAMS_${E_POINTWISE_GROUP}" "${E_POINTWISE_PROGRAMS}")
    string(JOIN ";" pointwise_signature ${E_POINTWISE_PROGRAMS} ${pointwise_export_args})
    get_property(existing_signature GLOBAL PROPERTY "CUTEAFD_VISION_POINTWISE_${E_POINTWISE_GROUP}")
    if(existing_signature AND NOT existing_signature STREQUAL pointwise_signature)
      message(FATAL_ERROR "Conflicting vision pointwise group ${E_POINTWISE_GROUP}")
    endif()
    if(NOT existing_signature)
      set_property(GLOBAL PROPERTY "CUTEAFD_VISION_POINTWISE_${E_POINTWISE_GROUP}" "${pointwise_signature}")
      add_custom_command(OUTPUT ${pointwise_outputs}
      COMMAND ${CUTEAFD_SPARKINFER_VERIFY_COMMAND}
      COMMAND "${CMAKE_COMMAND}" -E env ${CUTEAFD_SPARKINFER_PYTHON_ENV}
        ${CUTEAFD_EXPORT_LAUNCHER} "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/aot/export_b12x_vision_aot.py"
        --output-dir "${pointwise_directory}" --capability ${CUTEAFD_VISION_CAPABILITY} ${pointwise_export_args}
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
    ${CUTEAFD_EXPORT_CACHE_COMMAND} --output "${CUTEAFD_VISION_ATTENTION_DIR}/vision_attention_d64.h" --output "${CUTEAFD_VISION_ATTENTION_DIR}/vision_attention_d64.o" --output "${CUTEAFD_VISION_ATTENTION_DIR}/vision_attention_d64.json" --output "${CUTEAFD_VISION_ATTENTION_DIR}/vision_attention_d72.h" --output "${CUTEAFD_VISION_ATTENTION_DIR}/vision_attention_d72.o" --output "${CUTEAFD_VISION_ATTENTION_DIR}/vision_attention_d72.json"
    -- "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/aot/export_b12x_vision_aot.py"
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
# Pixtral retains its geometry-keyed exports; the shared owner manages their lifetime.
cuteafd_vision_owner_entry(d128 HEAD_DIM 128 POINTWISE_GROUP pixtral
  PROGRAMS vision_attention_d128_full ARGS --group attention --head-dim 128
  POINTWISE_PROGRAMS vision_axial_rotary_d128_bf16 vision_rms_bf16_gamma_h2048
  POINTWISE_ARGS --group pointwise --head-dim 128 --hidden 2048 --bf16-products --bf16-before-gamma)
# Lender groups must precede their borrowers; the helper validates availability.
foreach(kind inkling2 inkling2_mini step5)
  if(kind STREQUAL "inkling2" OR kind STREQUAL "inkling2_mini")
    set(dim 64)
  else()
    set(dim 96)
  endif()
  execute_process(COMMAND "${Python3_EXECUTABLE}" "${muse_exporter}" --list-perception-programs --perception ${kind}
    RESULT_VARIABLE pe_status OUTPUT_VARIABLE pe_programs OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(NOT pe_status EQUAL 0 OR NOT pe_programs)
    message(FATAL_ERROR "Cannot read ${kind} PE export inventory")
  endif()
  set(shared_attention)
  set(borrow_group muse)
  set(borrowed_programs vision_gelu_h4096)
  if(kind STREQUAL "inkling2_mini")
    set(shared_attention DIRECTORY "${CUTEAFD_VISION_ATTENTION_DIR}/pe_inkling2")
    set(borrow_group pe_inkling2)
    set(borrowed_programs
      vision_pe_norm_h1024 vision_pe_scale_h1024 vision_pe_position_h1024 vision_pe_rotary_d64
      vision_biased_projection_h1024_o3072 vision_biased_projection_h1024_o1024
      vision_biased_projection_h1024_o4096 vision_biased_projection_h4096_o1024
      vision_projection_h1200_o1024 vision_pe_rgb_p20 vision_pe_merge_h1024
      vision_pe_patch_projection_h4096_o4096 vision_gelu_h4096)
  elseif(kind STREQUAL "step5")
    set(shared_attention DIRECTORY "${CUTEAFD_VISION_ATTENTION_DIR}/d96_full")
    set(borrowed_programs
      vision_biased_projection_h1536_o4608 vision_biased_projection_h1536_o1536
      vision_biased_projection_h1536_o8960 vision_biased_projection_h8960_o1536)
  endif()
  cuteafd_vision_owner_entry(pe_${kind} HEAD_DIM ${dim} POINTWISE_GROUP pe_${kind} ${shared_attention}
    PROGRAMS vision_attention_d${dim}_full ARGS --group attention --head-dim ${dim}
    BORROW_POINTWISE_GROUP ${borrow_group} BORROW_POINTWISE_PROGRAMS ${borrowed_programs}
    POINTWISE_PROGRAMS ${pe_programs} POINTWISE_ARGS --group perception --perception ${kind})
endforeach()
# Generic still towers own dynamic-row attention and an independent program group.
# Registration alone does not qualify runtime dispatch, which stays oracle-gated.
set(tower_inventory "${CMAKE_CURRENT_LIST_DIR}/../../../python/tools/aot/vision_tower_programs.py")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${tower_inventory}")
foreach(kind kimi minimax hy4f hy4)
  if(kind STREQUAL "kimi")
    set(dim 128)
  elseif(kind STREQUAL "hy4f")
    set(dim 72)
  elseif(kind STREQUAL "hy4")
    set(dim 96)
  else()
    set(dim 80)
  endif()
  execute_process(COMMAND "${Python3_EXECUTABLE}" "${muse_exporter}" --list-tower-programs --tower ${kind}
    RESULT_VARIABLE tower_status OUTPUT_VARIABLE tower_programs OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(NOT tower_status EQUAL 0 OR NOT tower_programs)
    message(FATAL_ERROR "Cannot read ${kind} generic tower export inventory")
  endif()
  list(GET tower_programs 0 first_program)
  if(NOT first_program MATCHES "^vision_dynamic_attention_h[0-9]+_d${dim}$")
    message(FATAL_ERROR "Generic tower inventory must start with geometry-keyed dynamic attention")
  endif()
  list(REMOVE_AT tower_programs 0)
  set(borrow_groups muse)
  set(borrowed_programs)
  if(kind STREQUAL "kimi")
    # These use the same media/pointwise builders and keys as Muse.
    set(borrowed_programs vision_projection_h4096_o4096 vision_gelu_h4096)
  elseif(kind STREQUAL "hy4")
    # Reuse identical compiled objects from earlier groups to keep symbols unique.
    set(borrow_groups muse tower_minimax tower_hy4f)
    set(borrowed_programs
      vision_patches_raster_p16_t1
      vision_residual_h1536
      vision_gelu_h6144
      vision_biased_projection_h1536_o1536
      vision_biased_projection_h6144_o6144)
  endif()
  set(attention_args --group tower --tower ${kind})
  foreach(stem IN LISTS tower_programs)
    list(APPEND attention_args --omit-program ${stem})
  endforeach()
  cuteafd_vision_owner_entry(tower_${kind} HEAD_DIM ${dim}
    PROGRAMS ${first_program} ARGS ${attention_args}
    POINTWISE_GROUP tower_${kind} POINTWISE_PROGRAMS ${tower_programs}
    BORROW_POINTWISE_GROUP ${borrow_groups} BORROW_POINTWISE_PROGRAMS ${borrowed_programs}
    POINTWISE_ARGS --group tower --tower ${kind} --omit-program ${first_program})
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
