# UMA deep dive: where the engine still acts like a discrete GPU (2026-10-28)

**Scope.** This covers `xolotl/tools/v41/engine/` on strixhalo: gfx1151, 123.5 GiB GTT, ROCm 7.14, GSQ-217 on
the 990 Pro, quality mode. Two workloads were looked at: plain decode, and DSpark at depth 4 with
`STRATA_DSPARK_CONF=0.6`.

**What "lossless" means here.** "Bit-identical" means the same kernels on the same bytes in the same order, so the
output does not change. "Exact" means the same model math in FP32, but different rounding than today's verify
pass. Exact changes keep DSpark's guarantee (the trunk's own greedy argmax) but can flip near-ties compared with
today's DSpark build. Each change in the ranking is labelled one way or the other.

**How it was done.** No engine source, script or doc was changed except this file. I made 6 engine runs (A-F), all
under `flock ~/.cache/xolotl/gpu.lock`. I also sampled the other agent's runs passively (read-only sysfs, no ptrace).
Section 6 lists the files.

## 0. Summary

1. **The worst UMA problem isn't a copy. It's KFD queue eviction.** The 80 GiB expert arena is anonymous 4 KiB
   memory registered with `hipHostRegister`. Nothing locks it: no mlock, no pin. So zram swap, kcompactd and
   reclaim can move its pages, and each time they do, amdgpu/KFD **stops every GPU queue of the process** until
   the mapping is rebuilt. The per-process counter `/sys/class/kfd/kfd/proc/<pid>/stats_*/evicted_ms` gives:

   | run | queues evicted |
   |---|---|
   | A (3.5 min) | 82 s |
   | B (3 min) | 67 s |
   | D (rocprof) | 144 s |
   | E | 85 s |
   | the other agent's pf-r2 / pf-r3 | 95 s / 87 s |

   The stalls land at the start of decode, when the 2.5-9 GiB arena extension is built. They cost 8-22 s per decode phase, with the
   GPU idle at 600 MHz. They also land in the DSpark seed passes, the drafter load and the prefill.

   **Turning the extension off (`STRATA_ARENA_EXT_GIB=0`, run E) removed the decode-start stall.** Same 136 tokens;
   decode 56.7 s (A) / 37.3 s (B) -> 17.2 s, i.e. 2.4 / 3.6 -> 7.9 tok/s. Stall-free passes got 6 ms slower
   (+4 % misses).

2. **The verify pass's "dense noise" (110-334 ms) is not dense work.** It is three things:
   - the 8 DSpark seed passes (prompt tail) cost 10-99 s per run and sit inside the `pass_v split` average;
   - the eviction stalls above;
   - whichever timer is open absorbs a stall. Run E's seed-phase stall landed in "logits 695 ms/pass". pf-r2 and
     pf-r3 produced the same tokens with dense/tier 339/233 against 156/363.

3. **In stall-free DSpark passes, ggml's GEMM choice is the biggest waste.** For widths m >= 4, the two
   hyper-connection mixes per layer (F16 [16384 x 24] x F32 [16384 x m], PREC_F32) leave `mul_mat_vec_f`: RDNA3
   F16 is only allowed up to ne11 <= 3 (ggml `mmvf.cu:841`). They go to rocBLAS SGEMM
   `Cijk_..._SB_MT32x32x8` with a 256-thread grid, i.e. **one workgroup**, at 668 us each. That's 80 per pass and
   **53 ms per full pass** (measured, trace D). Passes with k <= 2 pay 0.

4. **The remaining copies are small; the expert kernels run at 140-195 GB/s** (THP arena: only +3 %, measured). Decode expert reads are already
   zero-copy. Single-expert launches (the file-tier misses, after the read wait) run at 36-63 GB/s. `predict_multi`
   is computed and thrown away on every verify layer. Removing it is bit-identical (136/136 tokens) and saves
   3.7 ms/pass.

**Top recommendations.** Estimated unless marked measured (section 5):

| # | change | gain |
|---|---|---|
| R1 | stop the evictions (sysctl profile and/or mlock of arena + extension; build the extension once, not per decode) | measured: 8-22 s per decode phase (DSpark ~50 ms/pass at 390 passes; plain 1.7-7.7 ms/token over 3,000 tokens, 17-77 ms/token over 300) |
| R2 | hc mixes for m = 4-8 through mmvf | ~-50 ms per full verify pass |
| R3 | skip `predict_multi` without a cache | measured -3.7 ms/pass |
| R4 | launch each file miss's kernels as its read lands | ~-2-3 ms/token, -5-10 ms/pass |
| R5 | cache the verify graphs | ~-5 ms/pass |

## 1. Copy / duplicate / sync map on UMA

Notation: P = per plain token, V = per verify pass of m tokens (3.4-3.7 on average). On UMA a small
`hipMemcpyAsync` H2D shows up as a `__amd_rocclr_copyBuffer` blit kernel or a DMA op. It costs ~2-20 us of
latency on the critical path, not bandwidth.

### 1a. Memory placement

| Buffer | Where (file:line) | Size | UMA status |
|---|---|---|---|
| Expert arena | `src/core/pinned.cu:236-260` (4 KiB anonymous mmap: `STRATA_NO_LARGEPAGES=1` is forced by `serve/ds41-engine-strix.sh`), registered at `pinned.cu:410-425` | 80 GiB | Zero-copy (`Arena::zc`). **No mlock or pin on Linux** (the "working-set lock" is Windows-only): it can be swapped (zram) and migrated (compaction). Root of finding 1. |
| Arena extension | `ds41_moe.cpp:2310-2340` (mmap + `cudaHostRegister` + memset, on a side thread) | 0.74-10 GiB | Same as the arena. Released before every multi-token prompt (`:5455-5464`) and rebuilt for the next decode: per-request allocate/pin churn. The build lands on the first decode pass. |
| Dense weights | `ds41_dense.cpp:360-399` (`ggml_backend_alloc_ctx_tensors_from_buft` + `ggml_backend_tensor_set` from the GGUF mmap) | ~9 GB GTT (est.: GGUF minus experts) | A one-time pageable H2D copy. The file pages stay in the page cache (reclaimable), so nothing is held twice. |
| DSpark drafter | `ds41_dspark.cpp:166-212`, same pattern | 7.9 GB (full), 5.1-6.5 GB (requant) | One-time pageable copy. Load took 14-87 s (A-E; 20-122 s tonight) and 150-170 s with on-load requant (CPU at 3,100 %). It takes GTT budget that the extension then can't get: 52 slots instead of 635 with the full drafter. |
| `d_stage` / `h_fstage` | `ds41_moe.cpp:1408` / `:2523, 2855, 3556` | 116 MiB device + 116 MiB pinned | Fallback only (no arena slot), so idle. |
| `d_pf`, `h_dummy` | `ds41_moe.cpp:1421-1427` | 87 MiB | Only with `--pf-b > 0` (the server passes 0). |
| Prompt chunk buffers (`c_stage`, `c_bounce`, `c_hparts`, graph buffer) | `ds41_moe.cpp:4085, 4149, 4215, 4255, 4726` | ~14.5 GiB beyond the arena at batch 8K | A DRAM-to-DRAM DMA of every streamed expert (78.6 GB/s). Prompt zero-copy was tried earlier and reverted: no gain, and not bit-identical. Allocated and freed around every prompt, which adds to the reclaim churn of finding 1. |

### 1b. Per-token / per-pass copies, syncs and wasted work (decode)

| What | Where | Bytes | Count | Note |
|---|---|---|---|---|
| Routing + `fn` readback + **sync** | `ds41_dense.cpp:1593-1596` | (D + 2 NU) x 4 x m | 1/layer | Needed: the host decides each expert's tier. The `fn` copy is unused in plain decode, because the tier reads `x_dev` (`ds41_moe.cpp:2544`). |
| Verify activation round trip | `ds41_moe.cpp:3613-3615` (memcpy `x_host` -> pinned `h_x`, H2D `d_x`) | 16 KB x m | 1/layer (V) | `fn` is already on the device (`o_fn`, via `attn_router(..., fn_dev)`). |
| Grouped-launch metadata H2D | P `ds41_moe.cpp:2679`, V `:3679` | 0.2-1 KB | P 1-2/layer, V 2-3/layer (trace D: 89 copies/pass) | Pinned `h_meta` is device-addressable on the APU. |
| Sum weights H2D | P `:3153`, V `:3870` | 24 B / 24 m B | 1/layer | Same. |
| Tier end **sync** | P `:3157`, V `:3876` | - | 1/layer | The tier stream and ggml's stream are ordered through the host. |
| Graph-compute **sync** | `ds41_dense.cpp:516` (`ggml_backend_graph_compute` = async + synchronize) | - | begin, finish (1/layer), predict_multi (1/layer V), logits | `finish(l)` -> `attn_router(l+1)` doesn't need a host round trip. |
| Verify graphs rebuilt every call | `ds41_dense.cpp:1580-1588` (attn), `:1683-1686` (finish), `:1750-1755` (logits) | - | 81/pass | n = 1 graphs are cached per (layer, cap) (`:1576`); `predict_multi` per (layer, width) (`:1655`). Trace D: 129 graph-boundary gaps/pass x 70 us. |
| `predict_multi` (D3 prefetch) | `ds41_generate.cpp:766-784` | 64 B x m | 1 graph + sync/layer (V) | **Thrown away**: `prefetch_multi` returns at once without a VRAM cache (`ds41_moe.cpp:1881`). |
| Logits D2H | `ds41_dense.cpp:1756` | 0.52 MB x m | 1/pass | Plus a host argmax. Small. |
| DSpark taps / step | `ds41_dense.cpp:1765`; `ds41_dspark.cpp:418-443, 466-473` | ~0.3 MB | 3 + 8 tiny syncs/pass | Taps 0.3 ms per pass. The draft step is 14.1 ms per pass, mostly the F16 head (1.32 GB, read once per 5-row block). |
| Engram rows | `ds41_dense.cpp:1357-1367` | 2 x m rows | 1/pass | Synchronous preads (~0.5 ms per token). |

## 2. Diagnosis: why the verify pass's dense time is so noisy

### 2a. The averages mix in one-time costs

`pass_v split` divides by every pass, including the 8 seed passes (prompt tail, 16 tokens each).
`verify ms/pass` covers only the decode passes. The difference between the two totals is the seed time:

| run | decode passes | seed total | per seed pass | dense (all) | dense, decode passes (lower bound) | verify ms/pass |
|---|---|---|---|---|---|---|
| full-c06 | 390 | 15.2 s | 1.90 s | 119.6 | 83.2 | 293.9 |
| full-c06b | 390 | 38.0 s | 4.75 s | 166.7 | 72.7 | 306.6 |
| full-c07 | 396 | **99.2 s** | **12.40 s** | 334.1 | 90.3 | 292.9 |
| full-d2 | 560 | 71.4 s | 8.93 s | 171.6 | 46.5 | 270.9 |
| full-d3c06 | 471 | 25.8 s | 3.22 s | 255.0 | 204.6 | 403.0 |
| iq3-c06 | 357 | 10.1 s | 1.26 s | 110.4 | 84.7 | 295.9 |
| q2k-c06 | 408 | 10.1 s | 1.26 s | 163.2 | 141.7 | 327.1 |

c07's "dense 334 ms" is 99 s of seed passes spread over 404 passes; its decode passes cost the same as c06's.
`verify ms/pass` still holds the decode-start stall (2b): up to ~22 s / 390 passes = ~57 ms/pass.

**Don't compare tonight's DSpark A/Bs on these averages.** Use the per-pass `STRATA_DSPARK_CONFLOG` and drop the
first decode passes, or compare `evicted_ms`.

### 2b. The stall is a KFD queue eviction

Per-pass verify times (`STRATA_DSPARK_CONFLOG`, agent4k, full drafter, 40 decode passes, the same 136 tokens in all
four runs):

| run | config | passes > 1 s (ms) | stall-free mean / median |
|---|---|---|---|
| A | baseline | 22,133 (pass 2), 4,013, 1,976, 1,024, 13,202 (pass 21) | 382 / 339 |
| B | `STRATA_D3_PF=0` | 22,065 (pass 1), 1,362 | 349 / 329 |
| D | rocprofv3 | 9,629, 1,409, 4,291, 1,947, 1,890, 1,001, 1,098, 3,090 | 354 / 338 |
| E | `STRATA_ARENA_EXT_GIB=0` | 1,717 | 362 / 358 |
| pf-r1 (other agent, warm prompt, iq3 drafter) | | 22,638 (pass 1) = 38 % of its verify time | median 258 |

What happens during a stall:

- **The GPU is idle.** Busy 0 % and sclk 600 MHz in the 10 Hz samples (`A.gpu` t = 146-165 s and 181-191 s).
  Package power drops to 45-65 W.
- **The decode thread spins in `libhsa-runtime64`** (eu-stack, 4 samples inside A's 22 s pass). It waits on a
  signal; it isn't computing.
- **In a kernel trace, a stall shows up as one kernel lasting seconds.** Trace D, pass 0: one ggml kernel ran
  9.3 s. Pass 34: one tier kernel ran 2.86 s. The queue was preempted mid-kernel.
- **The process's `evicted_ms` jumps.** B logged 18,952 ms in the second that ends the first decode pass. Every
  burst comes right after page migration or swap-out on the box (0.25 s samples, `B.ev`):

| t (s) | eviction logged (ms) | pages migrated | kcompactd scanned | swapped out | phase |
|---|---|---|---|---|---|
| 30 | 1,157 | 93,126 | 870,154 | 0 | arena fill |
| 101 | 2,618 | 102,911 | 4,096,000 | 2 | drafter load |
| 108 | 1,724 | 70,717 | 4,365,251 | 0 | drafter load |
| 111-127 | 2,434 + 14,492 + 1,188 + 747 | up to 101,788/s | | 125-150 K/s | seed passes. The engine's **VmSwap rose 0 -> 1.14 GB** |
| 145 -> 165 | 498, then **18,952** | 102,869 | | 133,058 | extension built (MemAvailable 19.1 -> 15.8 GB), first decode pass |

Over B's ~3 min, the box migrated 1.95 M pages and swapped out 7.1 M pages (27 GB) to zram. Live settings on
strixhalo: `vm.compaction_proactiveness=20`, `vm.watermark_boost_factor=15000`,
`vm.compact_unevictable_allowed=1`, zram swap with swappiness 10, THP `madvise`.

**Run E confirms the trigger.** E is A's config (without `DS41_PROF_X`) plus `STRATA_ARENA_EXT_GIB=0`. Result: no decode-start stall
(first passes 370 / 373 ms against 878 / 22,133 in A and 22,065 in B), the same 136 tokens, and decode 17.2 s instead of 56.7 s (A) and 37.3 s (B). The eviction moved
to the seed phase (33 s logged at t = 164, swap-out 80-140 K pages/s), where it now inflates the prompt time
(152 s for 4,096 tokens). E's stall-free passes are 6.0 ms slower than A's (median, paired): 3,037 misses against 2,917 without the 173
extension slots.

**Mechanism (inferred: no root, so no kernel tracing).** When a GPU-mapped page moves on the CPU side, amdgpu's
MMU notifier evicts all of the process's user queues. "Moves" means an arena page (anonymous, on the LRU) is
swapped out or migrated, or the TTM shrinker swaps a GTT buffer. The restore worker then re-validates the
registration; for a userptr BO that is the whole 80 GiB range. The queues resume afterwards. Two signs point at
the arena:

- The engine's own anonymous pages were swapped (VmSwap up to 1.14 GB), and the arena is 80 of its ~97 GB RSS.
- The kernel log has `amdgpu: SVM mapping failed, exceeds resident system memory limit` (21:33-21:43, the earlier
  ARENA_EXT over-budget runs).

Which of the two paths dominates (userptr/SVM invalidation vs TTM swap) needs `trace-cmd` / bpftrace as root on
`amdgpu_amdkfd_evict_userptr`, `svm_range_evict` and `ttm_bo_swapout`.

**Plain decode has the same stall.** Every warm bench carries a one-time excess in its first 64 decode steps:

| run | excess in the first 64 steps |
|---|---|
| 990-warm1 | ~5 s |
| 990-warm2 | ~23 s (attn_router 269.8 ms vs 59.3 steady, tier 201 vs 53) |
| plain1 | ~8.6 s |
| tie2 | ~22.7 s |
| run C (agent4k, plain, under rocprof) | only 0.74 s logged in decode (10.2 s over the run). No drafter, MemAvailable 25 GiB when the 10 GiB extension was built |

How large the stall gets depends on how tight memory is when the extension is built: the DSpark runs (drafter on GTT, MemAvailable 18-19 GiB) are
hit hardest. `verify_setup` (the other agent, 23:33) decoded 32 tokens in 34.6 s. In `--serve` the extension is released before
every multi-token prompt and rebuilt for the decode, so every request goes through this window.

### 2c. Stall-free verify pass, by GPU time (trace D, 13 typical passes, mostly k = 4)

| | ms/pass |
|---|---|
| expert kernels (`native_gu_kernel<11>` 62.4 + `native_down_kernel<11>` 35.0) | 97.3 |
| **rocBLAS SGEMM for the hc mixes (80 x 668 us; 0 when k <= 2)** | **53.4** |
| `mul_mat_vec_q` (Q8_0 dense, weights read once for m columns) | 40.2 |
| `mul_mat_f` (F16 via WMMA: Engram kv, indexer) | 17.0 |
| nvfp4 KV get_rows 6.0, flash-attn 4.0, mmvf 4.0, the rest (~1,900 small kernels) ~14 | ~28 |
| GPU kernel total (incl. the draft step) | 241.8 |
| GPU idle: file-read waits (30 layers x 3.4 ms, agent4k: 73 misses/pass) | ~102 |
| GPU idle: graph boundaries (129 x 70 us) + dependent-kernel gaps (4,180 x 2.8 us) + tier<->dense round trips (79 x ~60 us) | ~29 |

Plain decode is stable at 58-59 ms/token of dense. Trace C: 50.8 ms GPU busy, of which `mul_mat_vec_q` 33.3,
`mul_mat_vec_f` 6.9, and ~3,100 small kernels incl. 646 copyBuffer blits/token.

## 3. Batched expert kernel efficiency

**The kernel.** GSQ-217's experts are 99.3 % (Q3_K, Q3_K). Of the 15.2 MB mean blob, 2/3 is gate/up and 1/3 is
down. On HIP gfx1151, `STRATA_EXP_LAYOUTS` is 0 (`src/kernels/cuda/iq_kernels.cu:3026`), and Q3_K has no `kSplit`
multi-entry kernel (`:706-878`, IQ types only). So every launch is `native_gu_kernel<11>` /
`native_down_kernel<11>`: one warp per row, re-reading and re-dequantizing the row for every entry (token) of the
group (`:1334-1365`). In verify passes that is ~1.3 entries per group, so the re-dequant is a small cost (~10 %).

| case (rocprofv3 durations) | gate/up | down | effective read rate |
|---|---|---|---|
| plain decode, arena site (C, 4.2 experts/launch) | 257 us | 154 us | gu 165 GB/s, down 137 GB/s, all 154 GB/s (23.65 ms/token for 240 x 15.2 MB) |
| verify, arena site, 19-28 experts/launch (D) | 1,238-1,464 us | 677-796 us | gu 155-193, down 142-178 GB/s |
| verify, whole pass (D, 13 passes) | 62.4 ms | 35.0 ms | ~140 GB/s (900 groups x 15.2 MB / 97 ms) |
| **file misses, 1-2 experts/launch after the read wait (D)** | 279-323 us | 83-322 us | **36-63 GB/s** |
| reference: same kernels from `hipMalloc` staging (2026-10-21 trace, GSQ-200) | 192 us | 95 us | 188 GB/s |
| reference: streaming probe on the registered 4 KiB arena or hipMalloc (the 2026-10-18 bandwidth probe) | | | 240 GB/s |

**What limits it.**

1. **Small launches have no latency hiding.** A 1-expert launch has ~720 blocks of one-row warps. That's why the
   file-miss launches run at 36-63 GB/s. They sit on the critical path after the read wait in every layer with a
   miss: ~0.3-0.6 ms per layer.
2. **Page size is not the limit (measured).** Run F backed the arena with THP (AnonHugePages 83.8 GB) and decoded
   the same 104 tokens as C. Expert kernels: 22.88 against 23.65 ms/token (gate/up 249 vs 257 us, down 148 vs
   154 us), only -3 %. So the ~20 % gap to the old device-memory staging trace has another cause: GSQ-200's
   different class mix, or the snooped mapping of registered system memory against GTT. That is unresolved;
   measuring it needs a `hipMalloc`-copy A/B of one expert in `native_expert_bench`, not an engine run.
3. **The arena and file sites launch per class, three kernels each.** Big launches are already efficient: 155-195
   GB/s gate/up.

## 4. Other discrete-GPU leftovers worth noting (no ranking)

- **`--uma-reserve-gib 24`** keeps 24 GiB idle during decode for the next prefill's peak. The arena extension was
  the attempt to use it, and it is now the stall trigger. Building the arena at its decode size once, and shrinking
  the prompt batch instead of releasing pinned memory per request, would remove the churn. That trades some
  prefill speed; a measurement call.
- **The prompt path still double-buffers through device halves** (`c_stage`). It is a PCIe design. Zero-copy was
  tried earlier: no gain, and not bit-identical.
- **The verify-path expert cache (`vram_lru`, admission filter, demotions), the `d_pf` prefetch and `pcie_frac`
  dithering** are all dead code on UMA with `--uma-cache-gib 0`. Only `predict_multi` (R3) still costs time.

## 5. Ranked lossless changes

### 5a. Time budget

**Plain token, 990 warm bench (~120 ms, 8.3 tok/s):**

| part | ms/token | measured split |
|---|---|---|
| attn_router | 59 | GPU ~51: mmvq 33, mmvf 7, ~3,100 small kernels ~11 |
| tier | 53 | file wait ~32, expert kernels ~23, launch/sync ~2 |
| finish | 1.6 | |
| begin | 0.6 | |
| logits | ~5.5 | F16 head |
| decode-start stall | 5-23 s once per decode phase | |

R1 + R4 + R6 + R7 + R9 come to roughly -5 to -8 ms/token plus the amortized stall. Estimate: ~9 tok/s on long
benches, and much more on short answers, where the stall is a large share.

**DSpark pass, warm prompt (2.9 tokens/pass):**

- Today, measured (c06): 9.3 tok/s, i.e. verify 294 ms/pass, which includes ~50 ms/pass of amortized stall, plus
  draft 14 ms/pass.
- Without the stall: ~11 tok/s (rough estimate).
- With R2 + R3 + R4 + R5 + R7 + R8 + R9 on top, about -55 ms/pass: ~14 tok/s (rough estimate, assuming acceptance
  and misses stay as they are).

### 5b. Ranking

All rows are estimates unless marked **measured**. P = ms per plain token (~120 ms/token on the 990 warm bench).
V = ms per DSpark verify pass (stall-free ~260-350 ms; warm prompt ~258 median).

> **Decision (2026-10-28):** zram stays on, because the box needs swap as its OOM fallback for compiles. Since the arena is mlocked,
> zram can't take its pages. `scripts/strix_mem_profile` sets only the three sysctls.

| # | Change | P | V | Effort | Bit-identity |
|---|---|---|---|---|---|
| R1 | **Stop the queue evictions.** (a) System profile while Xolotl runs (root, user decision; no code): `vm.compaction_proactiveness=0`, `vm.watermark_boost_factor=0`, `vm.compact_unevictable_allowed=0`, `swapoff /dev/zram0` or `vm.swappiness=0`. (b) mlock the arena and the extension: memlock is unlimited in the toolbox (patch sketch below). (c) Don't build the extension at decode start: either `STRATA_ARENA_EXT_GIB=0` (measured below), or allocate it once with the arena and keep it across prompts. (d) Alternative to (a)+(b): a hugetlb arena (`vm.nr_hugepages` ~41,000 reserved right after halogen stops; the existing `MAP_HUGETLB` path in `pinned.cu:241`): never swapped, not compacted, 2 MiB GPU PTEs. | **measured** excess 5-23 s in the first 64 steps of 4 warm benches (C: only 0.7 s logged): 1.7-7.7 ms/token over 3,000 tokens, 17-77 ms/token over 300 tokens; per request in serve | **measured** 8-22 s per decode phase (A, B, D, pf-r1, pf-r3) ~ -20 to -55 ms/pass over 390 passes. Seed / drafter-load stalls 10-99 s per run (TTFT). E (c only): decode 56.7 / 37.3 -> 17.2 s for the same 136 tokens; stall-free passes +6 ms (fewer slots) | (a) none, (b) ~10 lines, (c) env / ~30 lines, (d) root + start-script | yes (memory placement only; B and E: same 136 tokens) |
| R2 | **hc mixes for verify widths 4-8 through `mul_mat_vec_f`.** Engine option: in `hc_mixes` (`ds41_dense.cpp:436-441`), split the `ggml_mul_mat(fn, flat)` into column chunks of <= 3 and concat. ggml option: a `strix-NN` patch that lets F16 mmvf take ne11 <= 8 on integrated GPUs (`mmvf.cu:841`). | 0 | ~-50 per pass with m >= 4. **Measured**: 80 x 668 us = 53.4 ms of SGEMM, against ~2 ms (est.) for mmvf. ~-40 ms average at c06's 73 % full passes | small | exact. mmvf's per-column math doesn't depend on `ncols_dst`, so the mixes should match plain decode's n = 1 bits (check with `STRATA_DSPARK_DUMP` against a plain dump) |
| R3 | **Skip `predict_multi` when the tier can't prefetch.** `pf_now &= tier.can_prefetch()` (cache open and `d_pf` allocated), or `STRATA_D3_PF=0` in the UMA wrapper today. | 0 | **measured** -3.7 (paired median, B vs A) | trivial | yes (**measured**, 136/136 tokens) |
| R4 | **Launch each file miss's kernels as soon as its own read lands**: per-blob completion counters in `ReadPool` instead of one `wait()` (`ds41_moe.cpp:3019`, `:3793`). Overlaps the 36-63 GB/s small launches with the remaining reads. | ~-2-3 | ~-5-10 | medium | yes. Groups are independent ("results do not depend on gridDim.y"); every entry writes its own `d_parts` row |
| R5 | **Cache n > 1 attn / finish graphs per (layer, n, cap)**, like `predict_multi` and the n = 1 graphs. | 0 | ~-4 to -8 (of 9 ms graph-boundary gaps) | small-medium | yes (same graph, same kernels) |
| R6 | **THP arena** (drop the forced `STRATA_NO_LARGEPAGES=1`). **Measured** (F vs C): expert kernels -0.77 ms/token (-3 %). But THP compaction during the arena fill caused 15 s of evictions at load. Only worth it as hugetlb (pre-reserved, R1(d)). | **measured** -0.8 | ~-3 (est., same -3 %) | config | yes (**measured**, 104/104 tokens) |
| R7 | **Fewer host round trips.** `finish(l)` via `graph_compute_async` (no sync before `attn_router(l+1)`); tier -> dense ordered by an event instead of `sum sync`. | ~-1 to -2 | ~-2 to -4 | small | yes (ordering only) |
| R8 | **Verify tier reads `fn` on the device** (`run_multi_to_dev(..., x_dev)`): no memcpy + H2D per layer. | 0 | ~-1 | small | yes (same bytes) |
| R9 | **Kernels read `h_meta` / the sum weights through the pinned buffer's device pointer**, instead of an H2D blit per launch (89 copies/pass). | ~-0.5 to -1 | ~-2 to -3 | small | yes, if `h_meta` is allocated coherent (`hipHostMallocCoherent`). Plain `hipHostMalloc` is non-coherent unless `HIP_HOST_COHERENT=1`, and a stale GPU-cached line would be a correctness bug |
| R10 | **Q3_K multi-entry gate/up/down** (`kSplit<11>` with `GRP_NC`, as the IQ types have). | 0 | ~-5 to -10 (1.3 entries/group) | high (kernel work) | yes, if the per-entry reduction order is kept (`iq_multi_parity`) |
| R11 | **Give the GTT back to the arena.** Drafter expert keep (`STRATA_DSPARK_KEEP=...:64`, which the serve wrapper already sets but `dspark_ab.sh` doesn't), or a smaller drafter: more extension slots. iq3 drafter: 635 slots vs 52, misses/token 16.5 vs 19.9. Only after R1, or the extension keeps stalling. | 0 | **measured** file wait -8.6 ms/token (iq3 vs full, from logs) | config | exact (drafts change, the output stays the trunk's) |

### Patch sketches (not applied)

R1(b), mlock the arena (after the `cudaHostRegister` and the fill), `src/core/pinned.cu` around `:425`:

```cpp
#if defined(__linux__)
    // strix-port: GPU-mapped pages must not be swapped or reclaimed - every such page change evicts all KFD queues
    // of the process until the whole registration is re-validated (measured: 8-22 s decode stalls)
    if (std::getenv("STRATA_ARENA_MLOCK") == nullptr || std::atoi(std::getenv("STRATA_ARENA_MLOCK")) != 0)
        note += mlock(base, (size_t) bytes) == 0 ? "; mlocked" : std::string("; mlock FAILED: ") + std::strerror(errno);
#endif
```

The same for the extension: `mlock(p, bytes)` after the memset in `arena_ext_build` (`ds41_moe.cpp:2336`).
`munmap` unlocks. mlock stops swap and reclaim, but compaction can still migrate locked pages unless
`vm.compact_unevictable_allowed=0`. That is why (a) and (b) go together, or (d) hugetlb.

R2, engine side (`ds41_dense.cpp:438-439`):

```cpp
        ggml_tensor * mixes;
        if (nt > 3 && nt <= 8) {   // strix-port: RDNA3 mmvf takes F16 only to ne11 <= 3; wider went to a 1-workgroup SGEMM
            mixes = nullptr;
            for (int64_t c0 = 0; c0 < nt; c0 += 3) {
                const int64_t w = std::min<int64_t>(3, nt - c0);
                ggml_tensor * part = ggml_mul_mat(c, fn, ggml_view_2d(c, flat, HC * D, w, flat->nb[1], (size_t) c0 * flat->nb[1]));
                ggml_mul_mat_set_prec(part, GGML_PREC_F32);
                mixes = mixes ? ggml_concat(c, mixes, part, 1) : part;
            }
        } else {
            mixes = ggml_mul_mat(c, fn, flat);
            ggml_mul_mat_set_prec(mixes, GGML_PREC_F32);
        }
```

R3, `ds41_generate.cpp:766`:

```cpp
        const bool pf_now = rdev1 && !no_tier_batch && d3_pf > 0 && m <= Ds41MoeTier::kMaxMulti && tier.can_prefetch();
        // Ds41MoeTier::can_prefetch(): gpu && cache && !no_cache && d_pf (prefetch_multi's own early-return conditions)
```

## 6. Measured vs estimated

**Measured tonight.**

- **Runs**, logs in `~/.cache/xolotl/runs/uma/`:

  | run | config |
  |---|---|
  | A | DSpark, agent4k, `DS41_PROF_X=2`, conf log |
  | B | A + `STRATA_D3_PF=0` |
  | C | plain decode + rocprofv3 + roctx |
  | D | A + rocprofv3 |
  | E | A without `DS41_PROF_X` + `STRATA_ARENA_EXT_GIB=0` |
  | F | C + THP arena (`STRATA_ARENA_THP=1`, `STRATA_NO_LARGEPAGES` unset) |

  A, B, D and E produced the same 136 tokens; C and F the same 104 (plain greedy).
- **Samplers**: `*.ev` (0.25 s: `evicted_ms`, MemAvailable, VmSwap, GTT, vmstat), `*.gpu` (0.1 s: busy %, sclk,
  power, CPU MHz), `A.stacks` (eu-stack), `*.conf` (per-pass verify ms), `trace-{C,D,F}/` (rocprofv3 CSV).
  Analysis scripts are in `py/`.
- **Read-only samples of the other agent's runs** pf-r1/r2/r3: `other-pf-r1.gpu`, `other2.ev/.gpu`. Their conf logs
  are in `runs/prefetch/`.
- **Numbers**:
  - eviction totals and timelines, the stall passes, the seed-time decomposition (from tonight's logs);
  - the SGEMM cost per pass and its width dependence;
  - the expert kernel durations / rates;
  - the gap classes in trace D;
  - `predict_multi` cost (-3.7 ms/pass) and bit-identity;
  - the extension-off effect (E).

**Estimated, or inferred.**

- The eviction mechanism: which kernel path (userptr/SVM vs TTM), and that mlock + sysctl + hugetlb remove it. This
  needs one run after the change, with `evicted_ms` checked.
- Every ms figure in section 5 not marked measured.
- That R2's mmvf columns are bit-equal to n = 1 (argued from the kernel code, not tested).
- Why the zero-copy expert kernels run ~20 % slower per expert than the 2026-10-21 staging trace (not the page size: F).

**Caveats.**

- rocprofv3 inflates wall time and host gaps, and probably the eviction totals too (D: 144 s). Kernel durations
  are reliable; host-gap numbers are upper bounds.
- The 4K prompt (agent4k) has more file misses per pass (73) than the warm prompt (~56), so its passes are slower
  than the warm-bench ones.
