#pragma once

// strix-port: <cuda_profiler_api.h> for the HIP build (ds41_moe.cpp / ds4_moe.cpp bracket a timed region with
// cudaProfilerStart / Stop for nsys).  HIP's hipProfilerStart / Stop are deprecated; rocprof does not need the
// markers, so they are no-ops that succeed.
#include "cuda_runtime.h"

static inline cudaError_t cudaProfilerStart() { return cudaSuccess; }
static inline cudaError_t cudaProfilerStop() { return cudaSuccess; }
