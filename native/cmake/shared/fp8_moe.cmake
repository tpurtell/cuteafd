# Exact FP8 routed-expert packages (FAMILY:fp8 entries of CUTEAFD_EXPERT_FAMILIES,
# for example mimo:fp8 or glm:fp8): the checkpoint's E4M3 experts with FP32
# 128x128 block scales, run by b12x fp8_moe programs. Each entry builds
# fp8-FAMILY/ next to the native library: tp1 in the SM120 coordinator build
# (RTX local / MTP layers), tp4, tp2 and tp6 in the SM121 Spark build. The daemon
# resolves <libdir>/fp8/fp8-FAMILY/tp<world>; the artifact scripts install
# the packages there. FAMILY:nvfp4 entries (glm, glmf, qwen4) build
# fp8-FAMILY-nvfp4/ the same way for NVIDIA ModelOpt NVFP4 releases (W4A16:
# packed E2M1 x E4M3 per-16 scales widened exactly, FP32 alpha per expert;
# exporter geometry FAMILY_nvfp4); FAMILY:nvfp4a4 builds fp8-FAMILY-nvfp4a4/,
# whose large-row steps run W4A4 (activations quantized with the checkpoint's
# input_scale, block-scaled FP4 MMAs; decode rows stay W4A16). A FAMILY:nvfp4
# entry builds both, since the daemon serves W4A4 by default. glmfdense:nvfp4[a4]
# is GLM 5.3 Flash's NVFP4 dense MLP (one always-selected expert, coordinator tp1).
if(CUTEAFD_CUDA_ARCHITECTURES MATCHES "^120")
  set(CUTEAFD_FP8_MOE_ROLE coordinator)
elseif(CUTEAFD_CUDA_ARCHITECTURES STREQUAL "121")
  set(CUTEAFD_FP8_MOE_ROLE spark)
else()
  message(FATAL_ERROR "FP8 expert packages require a single native SM120 or SM121 target")
endif()
set(CUTEAFD_FP8_MOE_CAPACITIES "1,16,80,256,1024,4096" CACHE STRING "FP8 expert package capacities")
# Spark builds may add fp8-FAMILY-bf16 beside each package: the same slices
# taking BF16 rows instead of FP8 K32 wire rows (the ranks then accept both;
# the coordinator chooses per step, e.g. serve-mimo --expert-input).
set(CUTEAFD_FP8_MOE_BF16_FAMILIES "" CACHE STRING
  "FP8 expert families (mimo;glm;...) that also get a BF16-input Spark package")
list(GET CUDAToolkit_INCLUDE_DIRS 0 CUTEAFD_FP8_MOE_CUDA_INCLUDE)
set(CUTEAFD_FP8_MOE_TOOL "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/aot/package_fp8_moe_aot.py")
set(CUTEAFD_FP8_MOE_MANIFESTS)
set(CUTEAFD_FP8_MOE_ENTRIES)
foreach(entry IN LISTS CUTEAFD_EXPERT_FAMILIES)
  list(APPEND CUTEAFD_FP8_MOE_ENTRIES "${entry}")
  # ModelOpt Qwen keeps MTP experts in FP8 beside NVFP4 routed layers.
  # Local MTP therefore needs the TP1 FP8 package in the same image.
  if(entry MATCHES "^qwen4:nvfp4(a4)?$" AND CUTEAFD_FP8_MOE_ROLE STREQUAL "coordinator")
    list(APPEND CUTEAFD_FP8_MOE_ENTRIES "qwen4:fp8")
  endif()
  if(entry MATCHES ":nvfp4$")
    list(APPEND CUTEAFD_FP8_MOE_ENTRIES "${entry}a4")
  endif()
endforeach()
list(REMOVE_DUPLICATES CUTEAFD_FP8_MOE_ENTRIES)
foreach(entry IN LISTS CUTEAFD_FP8_MOE_ENTRIES)
  if(NOT entry MATCHES ":(fp8|nvfp4|nvfp4a4)$")
    continue()
  endif()
  if(entry MATCHES "^(mimo|mimop|mimof|glm|glmf|qwen4):fp8$")
    set(geometry "${CMAKE_MATCH_1}")
    set(package "${CMAKE_CURRENT_BINARY_DIR}/fp8/fp8-${geometry}")
  elseif(entry MATCHES "^(glm|glmf|glmfdense|qwen4):(nvfp4|nvfp4a4)$")
    if(CMAKE_MATCH_1 STREQUAL "glmfdense" AND CUTEAFD_FP8_MOE_ROLE STREQUAL "spark")
      continue()  # dense MLPs run on the coordinator only
    endif()
    set(geometry "${CMAKE_MATCH_1}_${CMAKE_MATCH_2}")
    set(package "${CMAKE_CURRENT_BINARY_DIR}/fp8/fp8-${CMAKE_MATCH_1}-${CMAKE_MATCH_2}")
  else()
    message(FATAL_ERROR "FP8 expert family ${entry} must be (mimo|mimop|mimof|glm|glmf|qwen4):fp8 (mimof/mimop: MXFP4 weights) \
or (glm|glmf|qwen4):nvfp4[a4] (ModelOpt NVFP4, W4A16 or W4A4 large-row steps)")
  endif()
  # Spark packages also carry exact layouts (tp<n>-w<width>: ranks own whole
  # 128-row blocks without zero padding; the worker prefers them).
  set(exact_slices "")
  if(CUTEAFD_FP8_MOE_ROLE STREQUAL "spark")
    set(exact_slices "--exact-slices")
  endif()
  set(requested_layouts)
  if(CUTEAFD_FP8_MOE_ROLE STREQUAL "spark" AND NOT CUTEAFD_GENERIC_SPARK_COUNTS STREQUAL "")
    foreach(count IN LISTS CUTEAFD_GENERIC_SPARK_COUNTS)
      if(NOT count MATCHES "^[1-8]$")
        message(FATAL_ERROR "CUTEAFD_GENERIC_SPARK_COUNTS requires counts 1..8, got ${count}")
      endif()
      list(APPEND requested_layouts "tp${count}")
    endforeach()
    list(REMOVE_DUPLICATES requested_layouts)
    list(JOIN requested_layouts "," requested_layout_csv)
    set(requested_layouts --layouts "${requested_layout_csv}")
  endif()
  set(stamp "${CMAKE_CURRENT_BINARY_DIR}/fp8_moe_${geometry}.stamp")
  file(GENERATE OUTPUT "${stamp}" CONTENT "role=${CUTEAFD_FP8_MOE_ROLE}|capacities=${CUTEAFD_FP8_MOE_CAPACITIES}|${exact_slices}|${requested_layouts}\n")
  add_custom_command(
    OUTPUT "${package}/manifest.json"
    COMMAND ${CUTEAFD_SPARKINFER_VERIFY_COMMAND}
    COMMAND "${CMAKE_COMMAND}" -E rm -rf "${package}"
    COMMAND "${CMAKE_COMMAND}" -E env ${CUTEAFD_SPARKINFER_PYTHON_ENV}
      "${Python3_EXECUTABLE}" "${CUTEAFD_FP8_MOE_TOOL}" build
      --role "${CUTEAFD_FP8_MOE_ROLE}" --geometry "${geometry}" --capacities "${CUTEAFD_FP8_MOE_CAPACITIES}" ${exact_slices} ${requested_layouts}
      --build-dir "${CMAKE_CURRENT_BINARY_DIR}/fp8_moe_exports" --output "${package}"
      --cxx "${CMAKE_CXX_COMPILER}" --cuda-include "${CUTEAFD_FP8_MOE_CUDA_INCLUDE}"
      --cuda-libdir "$<TARGET_FILE_DIR:CUDA::cudart>" --runtime "${CUTEAFD_B12X_AOT_RUNTIME_LIBRARY}"
    DEPENDS "${CUTEAFD_FP8_MOE_TOOL}" "${stamp}"
      ${CUTEAFD_SPARKINFER_PROVENANCE_INPUTS} ${CUTEAFD_SPARKINFER_EXPORT_INPUTS}
    COMMENT "Building exact FP8 expert package fp8-${geometry} (${CUTEAFD_FP8_MOE_ROLE})"
    VERBATIM)
  list(APPEND CUTEAFD_FP8_MOE_MANIFESTS "${package}/manifest.json")
  if(CUTEAFD_FP8_MOE_ROLE STREQUAL "spark" AND geometry IN_LIST CUTEAFD_FP8_MOE_BF16_FAMILIES)
    set(bf16_package "${package}-bf16")
    add_custom_command(
      OUTPUT "${bf16_package}/manifest.json"
      COMMAND ${CUTEAFD_SPARKINFER_VERIFY_COMMAND}
      COMMAND "${CMAKE_COMMAND}" -E rm -rf "${bf16_package}"
      COMMAND "${CMAKE_COMMAND}" -E env ${CUTEAFD_SPARKINFER_PYTHON_ENV}
        "${Python3_EXECUTABLE}" "${CUTEAFD_FP8_MOE_TOOL}" build
        --role spark --input bf16 --geometry "${geometry}" --capacities "${CUTEAFD_FP8_MOE_CAPACITIES}" ${exact_slices} ${requested_layouts}
        --build-dir "${CMAKE_CURRENT_BINARY_DIR}/fp8_moe_exports_bf16" --output "${bf16_package}"
        --cxx "${CMAKE_CXX_COMPILER}" --cuda-include "${CUTEAFD_FP8_MOE_CUDA_INCLUDE}"
        --cuda-libdir "$<TARGET_FILE_DIR:CUDA::cudart>" --runtime "${CUTEAFD_B12X_AOT_RUNTIME_LIBRARY}"
      DEPENDS "${CUTEAFD_FP8_MOE_TOOL}" "${stamp}"
        ${CUTEAFD_SPARKINFER_PROVENANCE_INPUTS} ${CUTEAFD_SPARKINFER_EXPORT_INPUTS}
      COMMENT "Building exact FP8 expert package fp8-${geometry}-bf16 (spark, BF16 input)"
      VERBATIM)
    list(APPEND CUTEAFD_FP8_MOE_MANIFESTS "${bf16_package}/manifest.json")
  endif()
endforeach()
add_custom_target(cuteafd_fp8_moe_packages ALL DEPENDS ${CUTEAFD_FP8_MOE_MANIFESTS})
