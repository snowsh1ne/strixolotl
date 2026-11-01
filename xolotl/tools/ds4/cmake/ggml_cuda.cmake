# tools/ds4/cmake/ggml_cuda.cmake - fork: ggml's own CUDA backend for the DS4 dense half (one include() line in the
# root CMakeLists.txt, right after upstream sets GGML_CUDA OFF; docs/UPSTREAM_SYNC.md).
# ggml's own CUDA backend is off unless asked for: the DS4 dense half (tools/ds4/ds4_dense) runs its graphs on a
# ggml backend and needs ggml-cuda to run them on the GPU (-DSTRATA_GGML_CUDA=ON, build-ds4-gpu).
option(STRATA_GGML_CUDA "Build ggml's CUDA backend too (for the DS4 dense half on the GPU)" OFF)
if(STRATA_GGML_CUDA)
  set(GGML_CUDA ON CACHE BOOL "" FORCE)
  # ggml defaults CUDA graphs OFF when it is not llama.cpp's own build; without them the DS4 dense half's ~17k tiny
  # kernels per token run at launch-overhead speed (measured: 97 ms/token of attention, 2.1M cudaLaunchKernel)
  set(GGML_CUDA_GRAPHS ON CACHE BOOL "" FORCE)
endif()
# strix-port: ggml's HIP backend (ggml-hip, which compiles the ggml-cuda sources - and so xolotl-ggml.patch's GPU
# changes - for AMD) for the DS4 / V4.1 / GLM dense halves on an AMD GPU (-DSTRATA_GGML_HIP=ON, with
# STRATA_ENABLE_HIP=ON; build-strix).  ggml-hip refuses GGML_STATIC ("Static linking not supported for HIP/ROCm"),
# which the root sets ON for the CPU build: here it is OFF again.  The ggml libraries stay static archives
# (BUILD_SHARED_LIBS OFF); GGML_STATIC only adds -static to ggml's own link lines.  ggml-hip takes the HIP targets
# from CMAKE_HIP_ARCHITECTURES (cmake/hip_backend.cmake set it before this file).  HIP graphs are compiled in
# (ggml's default; GGML_CUDA_DISABLE_GRAPHS=1 turns them off at run time) and HIP VMM stays off (ggml's default
# GGML_HIP_NO_VMM=ON: on the APU every allocation comes from GTT through plain hipMalloc).
option(STRATA_GGML_HIP "Build ggml's HIP backend too (for the DS4 / V4.1 / GLM dense half on an AMD GPU)" OFF)
if(STRATA_GGML_HIP)
  if(NOT STRATA_ENABLE_HIP)
    message(FATAL_ERROR "STRATA_GGML_HIP=ON needs STRATA_ENABLE_HIP=ON")
  endif()
  if(STRATA_GGML_CUDA)
    message(FATAL_ERROR "STRATA_GGML_HIP and STRATA_GGML_CUDA are exclusive")
  endif()
  set(GGML_HIP ON CACHE BOOL "" FORCE)
  set(GGML_STATIC OFF CACHE BOOL "" FORCE)
endif()
