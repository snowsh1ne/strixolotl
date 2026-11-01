# Xolotl -> Strix Halo port plan

> **2026-10-18 update:** decisions taken and the code prepared up to the compile. What happened since: [ROADMAP.md](ROADMAP.md).
> - First model: DeepSeek-V4.1-Flash (1017 when published).
> - ROCm 7.14.0 wheels, a toolbox + compose, models in `~/models/xolotl`, the port in `xolotl/`.
>
> Sections 3-5 below are the original analysis. ROADMAP.md "What the port changed" records what was done and
> "Roadmap" what is next.

Target: Ryzen AI Max+ 395, Radeon 8060S (`gfx1151`, RDNA3.5, wave32), 128 GB unified LPDDR5X (~256 GB/s
theoretical, ~226-240 GB/s measured by strixite / halogen). Source: [Indras-Mirror/Xolotl](https://github.com/Indras-Mirror/Xolotl) (
Strata fork merged at 0.1.44.1). Nothing here had been built or run at the time.

## 1. Backend decision: HIP (ROCm), not Vulkan

"RADV or Vulkan" is really **HIP/ROCm vs Vulkan**. RADV (Mesa) is the Vulkan driver, and the only one left:
AMD discontinued AMDVLK in 2025-09. Remove any stale AMDVLK ICD on the target, because it silently overrides RADV.

Port to **HIP**, using ROCm TheRock for gfx1151. That is 7.14.1, as `docs/STRIX_HALO.md` uses; strixite runs 10.1
nightlies. Reasons:

- **Strata core already runs on gfx1151 through HIP.** `docs/STRIX_HALO.md` documents the build, gfx1151 defaults,
  WMMA prefill kernels, a hipBLASLt table and numbers: Qwen3.8-Flash-Next ~1,300 tok/s prompt, 50-60 tok/s decode.
  Xolotl inherits all of it.
- **Xolotl is CUDA C++, and core already has a HIP path for that.** `cmake/hip_backend.cmake` sets `LANGUAGE HIP` on
  the `.cu` files and force-includes `include/strata/hip_compat/cuda_runtime.h`. Almost every CUDA construct in the
  Xolotl code is already covered by that shim.
  - A Vulkan port would mean rewriting every kernel in GLSL/SPIR-V. The `metal/` port is the precedent: a separate
    project with ~18k lines of rewritten kernels.
  - The Xolotl ggml ops (`DSV41_FAKE_QUANT`, `DSV41_HC_MIXES`, FP4 KV) have no Vulkan implementation.
- **Both reference projects are HIP-only and reach the bandwidth limit.**
  - strixite: hand-written HIP, Qwen3.8-Flash-Next, decode kernels at 84-100% of 226 GB/s, prefill ~1,450 tok/s.
  - halogen: closed-source HIP, Qwen3.8-27B, ~240 GB/s decode, prefill 566 tok/s at 32K.
  - Neither one compares against Vulkan.
- **Vulkan's decode lead is a llama.cpp kernel effect, not a hardware one.** In llama.cpp, RADV beats ROCm by
  +18-30% on MoE decode, but ROCm wins prefill by +6-34%. Profiles put the gap in llama.cpp's HIP Q4_K mat-vec
  kernels. Xolotl's decode runs Strata's own kernels, so that gap does not carry over.

Keep llama.cpp on RADV (Mesa >= 26.x) only as an A/B baseline. One caveat: when the model does **not** fit,
llama.cpp + Vulkan + mmap beat llama.cpp + ROCm by 6.4x on decode (Soothill, 103 GiB Qwen). That comes from
ROCm's poor handling of pageable/mapped memory. It is a design warning for Phase 4: never let GPU kernels touch
pageable mmap'd expert memory.

## 2. Target host setup (from strixite, the STRIX_HALO doc and the toolbox guides)

- **BIOS:** set the UMA carve-out to its minimum (512 MB). The GPU's memory is GTT.
- **Kernel cmdline:** `amdgpu.gttsize=126976 ttm.pages_limit=32505856 amd_iommu=off` (~124 GiB GTT).
  - `amd_iommu=off` gives +1-12%, but it disables the NPU and DMA isolation.
  - The Strata doc uses `ttm.pages_limit=29360128` (112 GiB), which leaves more room for the host. Start there.
- **Kernel:** 6.18.5+ or 7.0. Your user account needs to be in `render` and `video`.
- **Environment:** set no `HSA_OVERRIDE_GFX_VERSION`, since a native gfx1151 build needs none.
  - Set `HSA_DISABLE_COREDUMP_ON_EXCEPTION=1`; otherwise a GPU fault dumps memory into the working directory.
  - Raise `ulimit -l` (memlock) to unlimited.
- **Out-of-memory behaviour:** near-zero MemAvailable hard-hangs the box (strixite #6), and pinned memory can't
  swap. Keep a 4-8 GiB MemAvailable guard.

## 3. What has to change (audit summary)

| Area | State today | Work |
|---|---|---|
| `ds41_generate`, `ds4_generate`, `glm_generate` | Expert-tier GPU libs are gated on `TARGET CUDA::cudart` (`tools/v41/engine/v41_engine.cmake:71`, `tools/ds4/cmake/ds4_moe.cmake:30`). Under HIP they silently build the `*_cpu` variants. | Accept `strata_hip_runtime`. Add `tools/**/*.cu` to the HIP glob. Shim `cuda_profiler_api.h`. |
| Dense half of those engines | Uses ggml through `ggml_backend_init_by_type(GPU)`, so it is backend-agnostic. The tree never builds `ggml-hip`, though; under HIP it runs on ggml-cpu. `DS41_HAVE_CUDA` keys off `TARGET ggml-cuda`. | Add `STRATA_GGML_HIP`, build ggml with `GGML_HIP=ON` + `xolotl-ggml.patch` (its GPU parts sit in `ggml-cuda/`, which ggml-hip compiles). Accept `ggml-hip`. |
| `strata-dense` (Qwen3.8-27B) | CUDA-only (`CMakeLists.txt:643`). `src/kernels/cuda/dense_kernels.cu:624-737` uses `nvcuda::wmma`. | Rewrite the GQA chunk attention with `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32`, following `qsa_prompt_attn.cu`. |
| Core HIP MMQ | `src/prefill/moe_mmq.cu:11` includes the generated `mmq_types.inc`. Only the CUDA branch generates it (`CMakeLists.txt:1522`). The HIP branch lacks it and the new types q2_k / q3_k / mxfp4 / q4_0 / q4_1 / iq1_s. **The merged tree probably does not build on HIP at all.** | Generate it on the HIP branch too, with the same type list. |
| Tiering (`ds41_moe.cpp`, `ds4_moe.cpp`, GLM) | Built for a discrete card: VRAM slots sized from `ggml_backend_dev_memory`, a pinned host arena (`cudaHostRegister`), staging copies arena->device, demote copies device->arena, an O_DIRECT NVMe tier through a pinned bounce, a `pcie_frac` split. On unified memory the VRAM slots and the arena double-count the same DRAM. These engines bypass core's `device_free_bytes()` UMA logic. | A UMA mode (Phase 4). |
| CUDA graphs | `ds41_moe.cpp:2028-2345` captures the tier stream into `tier_g`. The shim maps it, but vincalkr measured `GGML_CUDA_DISABLE_GRAPHS=1` at +6.6% decode on HIP. | Measure; default it off on gfx1151 if it loses. |
| Not a blocker | `moe_gap.cu` uses inline PTX `%globaltimer`, but only in the test tool `ds4_moe_replay`. | Guard it out on HIP. |
| Serving | `serve_ds41.py` / `serve_glm.py` are NVIDIA-neutral. The configs hard-code `/home/mal` paths. `ds41-engine.sh` / `glm-engine.sh` hard-code `MemoryMax=84G`. | Add Strix configs and set the cgroup limit to ~110G. |

No `cp.async`, `mma.sync`, cooperative groups, cuBLAS/CUTLASS, `__nv_fp8`/`bf16`, PDL or cuFile in Xolotl code.
Wave32 matches the code's 32-lane assumptions.

## 4. Model fit (128 GB, ~110-124 GiB GTT)

| Model | Size | Fit |
|---|---|---|
| Qwen3.8-27B IQ3_S + MTP | 13 GB | fully resident |
| Qwen3.8-Flash-Next IQ3_XXS (+MTP) | ~47 + 6 GB | fully resident (already measured by Strata) |
| DeepSeek-V4-Flash Q2 (legacy `ds4_generate`) | 80.8 GiB | fully resident |
| GLM-5.3-Flash Maya-S-v2 IQ2_XXS | ~89 GiB | resident, tight at 512K context |
| DeepSeek-V4.1-Flash GSQ | 152 GiB | **does not fit**: ~40+ GiB on NVMe; Engram (202.8 GB) is always on disk |

## 5. Phases

Each phase ends with a build + `ctest` + token-id gate run **on the target**, then a checkpoint.

**Phase 0: core baseline on gfx1151.**
- Fix the HIP-branch `mmq_types.inc` generation.
- Build `strata` with the `docs/STRIX_HALO.md` recipe.
- Reproduce that doc's Qwen3.8-Flash-Next numbers. They are the regression baseline for every later phase.
- Record a llama.cpp RADV and ROCm baseline on the same models.

**Phase 1: ggml-hip with the Xolotl patch.**
- Add `STRATA_GGML_HIP`; build pinned llama.cpp + `xolotl-ggml.patch` with `GGML_HIP=ON`, `AMDGPU_TARGETS=gfx1151`.
- Make the FQ_OP / HC_MIX checks and `DS41_HAVE_CUDA` backend-neutral.
- Check that the 32-lane rank-argsort and the MXFP4 indexer path (already `!GGML_USE_HIP` for wmma) compile and match.
- Watch for HIP VMM issues (`HipVMM Failure: invalid argument` was seen on DSV4 + llama.cpp on Strix).

**Phase 2: expert engines on HIP, smallest first.**
- Order: `ds4_generate` (V4-Flash, 80.8 GiB, fits; ds4fa reaches 25 t/s decode on Strix as a target), then
  `glm_generate`, then `ds41_generate`.
- Fix the CMake gates, add the profiler shim and a CUDA-graph switch.
- Gate correctness against the CUDA reference token ids, or against the CPU path where no CUDA run exists.

**Phase 3: Qwen3.8-27B dense.** Port `dense_kernels.cu` from wmma to RDNA WMMA and lift the CUDA-only gate.
halogen's ~10.6 t/s plain decode / ~600 t/s prefill is the external yardstick.

**Phase 4: UMA mode for the tier code (the main performance work).**
- **Detection:** `hipDeviceAttributeIntegrated`. Size from core `device_free_bytes()`, not `ggml_backend_dev_memory`.
- **Fully resident models:** one residency tier.
  - Experts live in device-visible memory (`hipMalloc` from GTT, as strixite does, or the existing mapped
    `PinnedArena`), and kernels read them in place.
  - VRAM slot cache off (or a pointer alias); no arena->device staging and no device->arena demotes.
- **DeepSeek-V4.1:**
  - The arena takes everything down to the MemAvailable guard: ~95-100 GiB. vincalkr measured arena 60->75 GiB at
    +40% decode on 128 GB.
  - NVMe misses go O_DIRECT straight into arena slots, with no extra H2D hop.
  - Pre-size any streaming slab to the **largest expert across all layers**; ds4fa hit an MMU fault when one layer's
    expert landed in pageable memory.
- **Staging buffers:** never reuse a pinned staging buffer before its copy event completes (a gfx1151 corruption
  hazard per strixite).
- **Memory guard:** check MemAvailable before each allocation and on a timer.

**Phase 5: gfx1151 tuning.**
- A hipBLASLt table for the DSV4/GLM prefill shapes.
- WMMA prefill expert kernels for the Xolotl quants (GSQ ternary/2/3-bit, IQ2_XXS).
- A `STRATA_GFX1151_DEFAULTS`-style table for the new engines, enabling only bit-identical switches by default
  (AGENTS.md rule).
- Measure `STRATA_SH_STREAM` and spec/MTP/DSpark on each engine.

**Phase 6: setup and serve.** Strix configs in `tools/*/serve/`, AMD telemetry through sysfs, a Strix section in
`docs/STRIX_HALO.md` for the Xolotl engines.

## 6. Risks

- **Correctness gates are CUDA-calibrated.** DSV4.1 bit-exact paths (`DS41_CHECK_ARENA`, hipBLASLt solution
  order) may not reproduce on HIP. Expect to re-baseline with KL instead of byte identity for some paths.
- **DSV4.1 on 128 GB is NVMe-bound.** Decode depends on the arena hit rate and SSD QD1 latency; it won't approach
  the 4090 box's 26.8 t/s unless a smaller quant fits.
- **ROCm on unified memory is fragile under memory pressure:** hangs, sticky `hipErrorIllegalAddress`. Server
  restart on a lost context (strixite exits 3 for its supervisor) is worth copying.
- **The dev box cannot run anything,** so every phase needs a round trip to the target.

## 7. Before starting

- Review **vincalkr/Strata-DSV4.1** (an outside AMD gfx1100 port of the DSV4.1 engine, cited in upstream's
  hand-off notes; now [vincalkr/DS41Engine](https://github.com/vincalkr/DS41Engine)). It may already solve most of Phase 1-2.
- Open questions: model priority; target RAM (128 GB assumed); ROCm version (7.14.1 TheRock as the Strata doc, or
  newer); whether the port lives as a copy of upstream here or as a patch series on top of it.
