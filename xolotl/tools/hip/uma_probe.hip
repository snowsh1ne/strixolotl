// strix-port: new file (not upstream). Build in the toolbox: hipcc -O2 --offload-arch=gfx1151 -x hip uma_probe.hip -o uma_probe
// uma_probe: runtime checks for the Xolotl Strix Halo port (STATUS.md §6 item 6, §7 item 7, §8 Phase 4).
//   1. GPU read bandwidth from hipMalloc memory
//   2. hipHostRegister(Portable|Mapped) of a large mmap (the PinnedArena path), GPU read bandwidth through the
//      mapped device pointer (zero-copy) vs hipMalloc
//   3. hipMemcpyAsync from a pageable buffer that is overwritten right after the call returns (h2d_pageable)
// usage: uma_probe <arena GiB>
#include <hip/hip_runtime.h>
#include <sys/mman.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    printf("FAIL %s: %s\n", #x, hipGetErrorString(e_)); return 1; } } while (0)

__global__ void rd(const uint4* __restrict__ p, size_t n, unsigned* out) {
    uint4 acc = {0, 0, 0, 0};
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        uint4 v = p[i];
        acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
    }
    if ((acc.x ^ acc.y ^ acc.z ^ acc.w) == 0x12345678u) atomicAdd(out, 1u);
}
__global__ void sum_words(const unsigned* p, size_t n, unsigned long long* out) {
    unsigned long long s = 0;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) s += p[i];
    atomicAdd(out, s);
}

static long memavail_gib() {
    FILE* f = fopen("/proc/meminfo", "r"); char k[64]; long v = 0; char u[16];
    while (f && fscanf(f, "%63s %ld %15s", k, &v, u) == 3) if (!strcmp(k, "MemAvailable:")) break;
    if (f) fclose(f);
    return v / 1048576;
}

static double bw(const void* p, size_t bytes, unsigned* d_out, int reps = 5) {
    size_t n = bytes / 16;
    rd<<<40 * 32, 256>>>((const uint4*)p, n, d_out);
    (void)hipDeviceSynchronize();
    double best = 0;
    for (int r = 0; r < reps; r++) {
        auto t0 = std::chrono::steady_clock::now();
        rd<<<40 * 32, 256>>>((const uint4*)p, n, d_out);
        if (hipDeviceSynchronize() != hipSuccess) return -1;
        double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        best = std::max(best, bytes / s / 1e9);
    }
    return best;
}

int main(int argc, char** argv) {
    size_t arena_gib = argc > 1 ? strtoul(argv[1], nullptr, 10) : 8;
    hipDeviceProp_t pr; CK(hipGetDeviceProperties(&pr, 0));
    printf("device %s %s, %d CUs, integrated=%d, canMapHostMemory=%d, pageableMemoryAccess=%d\n", pr.name,
           pr.gcnArchName, pr.multiProcessorCount, pr.integrated, pr.canMapHostMemory, pr.pageableMemoryAccess);
    size_t fr, tot; CK(hipMemGetInfo(&fr, &tot));
    printf("hipMemGetInfo free %.1f / total %.1f GiB, MemAvailable %ld GiB\n", fr / 1073741824.0, tot / 1073741824.0,
           memavail_gib());
    unsigned* d_out; CK(hipMalloc(&d_out, 4));

    // 1. device memory
    size_t dbytes = 4ull << 30;
    void* d; CK(hipMalloc(&d, dbytes)); CK(hipMemset(d, 1, dbytes));
    printf("[1] hipMalloc 4 GiB read: %.1f GB/s\n", bw(d, dbytes, d_out));
    CK(hipFree(d));

    // 2. registered host arena
    if (memavail_gib() < (long)arena_gib + 16) { printf("[2] skip: MemAvailable < arena + 16 GiB\n"); }
    else {
        size_t ab = arena_gib << 30;
        auto t0 = std::chrono::steady_clock::now();
        void* h = mmap(nullptr, ab, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE, -1, 0);
        if (h == MAP_FAILED) { printf("[2] mmap failed\n"); return 1; }
        memset(h, 1, ab);
        double tpop = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        t0 = std::chrono::steady_clock::now();
        hipError_t e = hipHostRegister(h, ab, hipHostRegisterPortable | hipHostRegisterMapped);
        double treg = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        printf("[2] mmap+touch %zu GiB %.1fs, hipHostRegister(Portable|Mapped) -> %s in %.1fs, MemAvailable %ld GiB\n",
               arena_gib, tpop, hipGetErrorString(e), treg, memavail_gib());
        if (e == hipSuccess) {
            void* dp = nullptr; CK(hipHostGetDevicePointer(&dp, h, 0));
            CK(hipMemGetInfo(&fr, &tot));
            printf("    device ptr %s host ptr, hipMemGetInfo free now %.1f GiB\n", dp == h ? "==" : "!=", fr / 1073741824.0);
            size_t win = std::min(ab, (size_t)4 << 30);
            printf("    zero-copy GPU read, first 4 GiB: %.1f GB/s; last 4 GiB: %.1f GB/s\n", bw(dp, win, d_out),
                   bw((char*)dp + ab - win, win, d_out));
            void* d2; CK(hipMalloc(&d2, 1ull << 30));
            t0 = std::chrono::steady_clock::now();
            for (int r = 0; r < 4; r++) CK(hipMemcpyAsync(d2, (char*)h + ((size_t)r << 30), 1ull << 30, hipMemcpyHostToDevice, 0));
            CK(hipDeviceSynchronize());
            double tc = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            printf("    registered -> hipMalloc copy: %.1f GB/s\n", 4.0 * 1.073741824 / tc);
            CK(hipFree(d2));
            CK(hipHostUnregister(h));
        }
        munmap(h, ab);
    }

    // 3. async H2D from pageable memory, source overwritten right after the call returns
    {
        const size_t sizes[] = {64, 4096, 65536, 1 << 20, 16 << 20};
        hipStream_t s; CK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));
        unsigned long long* d_sum; CK(hipMalloc(&d_sum, 8));
        for (size_t sz : sizes) {
            int bad = 0;
            for (int it = 0; it < 200; it++) {
                std::vector<unsigned> v(sz / 4, 1u);
                unsigned* dst; CK(hipMallocAsync((void**)&dst, sz, s));
                // keep the stream busy so the copy is queued, not executed immediately
                rd<<<40 * 32, 256, 0, s>>>((const uint4*)dst, sz / 16, d_out);
                CK(hipMemcpyAsync(dst, v.data(), sz, hipMemcpyHostToDevice, s));
                std::fill(v.begin(), v.end(), 7u);       // reuse right away, as ds41_moe's host vectors do
                CK(hipMemsetAsync(d_sum, 0, 8, s));
                sum_words<<<64, 256, 0, s>>>(dst, sz / 4, d_sum);
                unsigned long long hs = 0;
                CK(hipMemcpyAsync(&hs, d_sum, 8, hipMemcpyDeviceToHost, s));
                CK(hipStreamSynchronize(s));
                if (hs != sz / 4) bad++;
                CK(hipFreeAsync(dst, s));
            }
            printf("[3] pageable async H2D %8zu B: %d/200 copies saw the overwritten source\n", sz, bad);
        }
        CK(hipStreamSynchronize(s));
    }
    printf("done\n");
    return 0;
}
