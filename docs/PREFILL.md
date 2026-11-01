# Prefill / time to first token on Strix Halo (2026-10-28)

Scope: the prompt path of `ds41_generate --serve` (DeepSeek-V4.1-Flash GSQ-217, 990 Pro, served config
`--long --dspark`), with emphasis on agentic turns (+0.8-7K new tokens on a 5-25K conversation). §0-§7 are the
2026-10-28 analysis (a report, no engine change). The 2026-10-29 night built #1-#4 and #7: the next section. Tools are
in `scripts/dev/prefill/`, and run logs are in `~/.cache/xolotl/runs/prefill/`.

## 2026-10-29: built and measured (read this first; §0-§7 below are the 10-28 analysis)

**Status 2026-10-30: FROZEN.** Defaults: fused key gather,
driver-free settle, predicted reads (start script; `--no-pred-reads` = off). Optional: `--fast-prefill`. Not set:
`STRATA_SERVE_DSPARK_EVICT=8193`. The "Open / next" list at the end of this section is parked, not planned.

All numbers are measured on the test box on 2026-10-29 22:00-00:15. Same binary for the A/B pairs, the served `--long --dspark`
config through `scripts/dev/prefill/turns.py` unless a row says plain. Each change has a switch, so the old path still
runs from the new binary; the old binary is `build-strix/ds41_generate.pre-prefill`. Runs are in
`~/.cache/xolotl/runs/prefill/{n29*,f29*,p29*}`, the A/B dumps in `runs/ab/nll-{mmq,mq3,mqv,fkv,v2r}-*`, the report is
`report.py`, and the 17-row compare is `ab_cmp.py v2 CFG` (driver: `ab_mmq.sh CFG`).

### What changed

| # | Change | In the binary | Bit-identical? | Measured effect |
|---|---|---|---|---|
| 2 | **Fused sparse-attention key gather**: `strix-03-dsv41-gather-kv.patch` adds `GGML_OP_DSV41_GATHER_KV`, one HIP kernel that writes each query's padded F16 keys (window rows + selected NVFP4 rows + zero pad); `ds41_dense.cpp` uses it | **on** (`STRATA_FUSED_KV=0` = old chain) | **yes**: two 8K `--ppl` NLL dumps byte-identical to the 10-22 v2 dumps; serve logit dumps identical; verify_setup 32/32 | dense half −35-40%: 8K one-shot 158.6 → 138.5 s; serve dense cold 15K 49.9 → 29.0 s, +6555 30.3 → 19.8 s, +844 5.9 → 3.9 s |
| 4/7 | **Driver-free settle** (`ds41_generate.cpp`): before a batch, wait for what it needs at full width, polling every 20 ms (was: wait only below 9 GiB, 250 ms steps) | **on** (`STRATA_SERVE_DRV_SETTLE=0` = old) | yes (it can only widen a capped batch) | driver wait 250 → 20 ms per batch turn |
| 3 | **Predicted file reads during the dense half** (`ds41_moe.cpp` `PredStage`): while layer L's dense half runs, read the file-tier experts the last batch routed at L (most tokens first) into a pinned buffer; L's passes take the ones they route again from there and read only the rest on demand. Since 23:45 the buffer is given back for a prompt whose full-width batch would not fit next to it (`Ds41MoeTier::pred_plan`) | engine off; **`start_xolotl_dsv41` on by default since 2026-10-30** (`--no-pred-reads` = off; user: bit-identical changes become defaults with an off flag) | **yes**: serve logit dumps (last row + 32 tail rows, 5 turns) byte-identical with #2+#3+#4 vs all off | read waits +844 13.2 → 7.9 s, +2519 5.2 → 0.9 s, +6555 7.5 → 1.1 s. 77-98% of the blobs read in time are used. Cost: 1.5 GiB pinned. The decode arena extension shrinks when MemAvailable bounds it. Plain, 600 greedy tokens at 15K, same tokens: 706 → 564 slots, file misses 2721 → 2779 (+2.1%), 10.75 → 10.58 tok/s (−1.6%, one run). At 2K depth there is no difference (387 tokens, 6845 vs 6702 misses) |
| 7 | `STRATA_SERVE_DSPARK_EVICT=8193` (env, `--dspark` only) | not set (4096) | yes (the drafter just stays loaded) | no drafter free/reopen (1.6-1.9 s) on ≥ 4096-token turns. After a free/reopen the driver showed ~2.2 GiB less free (10.6-10.8 vs 12.9-13.2 GiB), so with 8193 a 6.5K batch still ran at full width |
| 1 | **MMQ id-buffer slack** (`c_dst` and `m_iota` + 128 ints) | always (no effect without `--chunk-mmq`) | – | `--chunk-mmq --prompt-batch 8192` no longer faults (37 8K one-shot rows, 3 serve runs) |
| 1 | **MMQ expert kernels for prompt batches** with the MMVQ path's activation values (`STRATA_MMQ_ACT_MMVQ=1`, `moe_mmq.cu`) | **off** (`start_xolotl_dsv41 --fast-prefill` = `--chunk-mmq` + that env) | **no** | expert kernel 44-49 → 5.5-7.7 µs per token-expert pair (14.8 on +844 turns); quality A/B below |
| – | diagnostics: `STRATA_MMQ_Q3K_ONLY=1` (MMQ only for the pure Q3_K classes) | off | – | – |

### TTFT, 8-turn spec (`spec_turns.json`, `--dspark`, client-side seconds)

| turn | all off | lossless set (#2 #4 #3 1.5 GiB, evict 8193) | lossless + MMQ (`--fast-prefill`) | user 10-28 (old) |
|---|---|---|---|---|
| cold 15K (2 batches) | 166.0 | 146.6 | **67.3** | 165-174 |
| +844 @15K | 25.5 | 20.8 | 20.6 | 32.0 (@23K) |
| +2519 @16K | 51.8 | 44.9 | 30.3 | 49.4 (@8K) |
| +6555 @18K | 96.9 | 84.1 | 38.6 | 92.8 (@11K) |
| +844 @25K | 24.5 | 19.4 | 19.1 | – |
| cold 5.6K | 84.5 | 73.4 | 33.3 | 81.5 |
| +844 @5.6K | 25.0 | 19.7 | 24.3 (5.4 s KFD evictions) | – |
| +2519 @6.5K | 50.8 | 43.9 | 26.8 | – |
| **sum** | **524.8** | **452.7 (−14%)** | **260.2 (−50%)** | |

**Plain mode** (no DSpark, the server's default; `PF_NO_DSPARK=1`, spec_short; lossless = #2 #4 #3 1.5 GiB with the
give-back rule):

| turn | all off | lossless | |
|---|---|---|---|
| cold 5.6K | 82.8 | 72.6 | |
| +844 @5.6K | 22.7 | 19.1 | |
| +2519 @6.5K | 49.9 | 44.7 | |
| +6555 @9K | 91.7 | 81.2 | before the give-back rule: 86.8 s, the batch was capped at 5730 of 6555 (2 sweeps) |
| +844 @15.6K | 27.8 | 23.7 | |

KFD evictions still add noise: one +6555 turn took 112 s, with 31.8 s of `evicted_ms` and dense 40.4 s instead of ~17
(n29new2). Its logits were still byte-identical.

With MMQ the expert phase is bound by the reads again: read waits 9-23 s per turn. A bigger buffer helps there:
`STRATA_PF_PRED_GIB=3` took +6555 from 38.3 to 34.7 s. With MMVQ, 1.5 GiB already hides almost all read waits on turns ≥ 2.5K.
On +844 turns the dense window limits the read-ahead, not the buffer: ~40 of 105 slots per layer are read in time.
The 10-28 floor estimate (+844 ~14-16 s) is not reached: the on-demand reader still runs at 4.95 GB/s (#6 open).

### MMQ quality: the 17-row same-text A/B (vs the served MMVQ path `v2`, teacher T)

| config | ppl vs v2 (mean / median) | 95% CI over rows | rows worse | top-1 = T | gap to T | argmax = v2 | median per-token \|ΔNLL\| |
|---|---|---|---|---|---|---|---|
| v2 (MMVQ, served) | – | – | – | 0.8737 | +1.61% | 1 | 0 |
| **MMQ + MMVQ activation values (`--fast-prefill`)** | **+0.12% / −0.01%** | **[−0.89%, +1.26%]** | **8/17** | **0.8732** | **+1.73%** | 0.9045 | 0.033 |
| MMQ, ggml's own quantizer | +0.97% / +0.48% | [+0.17%, +2.15%] | 12/17 | 0.8730 | +2.59% | 0.9032 | 0.034 |
| MMQ (ggml quantizer) on the Q3_K classes only, 3 worst rows | +4.26% on those rows | | 3/3 | | | | |
| Q6_K dense requant (10-23, reference for the noise) | −0.47% / −0.32% | [−1.33%, +0.40%] | 8/17 | 0.8719 | +1.13% | 0.9003 | 0.036 |

- ggml's MMQ quantizer rounds `x * (127 / amax)` and keeps the block scale in fp32. The MMVQ path rounds `x / (amax / 127)`
  and multiplies with the fp16 scale of `block_q8_1`. `--fast-prefill` uses the MMVQ values (`STRATA_MMQ_ACT_MMVQ=1`).
  Single rows still swing ±5% (DSv4.1 amplifies any numeric change), as they do for the Q6_K requant.
- **Revised 2026-10-30 (literature check, no new runs):** the +0.97% of ggml's quantizer is probably NOT a precision
  effect. ggml's values are the slightly more precise ones (fp32 scale). Upstream measured q8_1-per-32 activation
  quantization at ~0.01-0.1% ppl vs FP16 (llama.cpp PR #4801), so +1% from rounding alone would be ~10x anything
  reported. Most likely it is model-amplified noise (CI low end +0.17%); a bug in that path is not ruled out. Not
  investigated further.
- Known effects of the method: accumulation-order changes barely move ppl but do change exact greedy strings
  (Thinking Machines, "Defeating Nondeterminism in LLM Inference", 2025); MoE routing and sparse-attention top-k are
  documented as sensitive to tiny numeric changes. Not covered by the 17-row test: contexts beyond 8K, exact
  tool-call/JSON strings, KLD against MMVQ.
- Not bit-identical: MMQ accumulates in a different order. On these 17 rows it is not distinguishable from the served
  path. **Decision (user, 2026-10-30): `--fast-prefill` stays an optional flag, off by default.**

### Open / next (parked 2026-10-30, not planned)

- The prediction buffer as part of the decode arena extension, so decode keeps its full 10 GiB. That means two
  registrations and handing slots over at prompt start. The rebalance reader's slot ownership has to be checked first.
- #6, a faster on-demand reader (4.95 GB/s; nvbench 5.3-5.4, prefetch reader 6.0-6.2). It is now the floor of +844 turns.
- #5 persistent prefix cache (side requests / resumed conversations), #7a keep the extension through small turns: not
  started.
- MMQ at 4K-8K needs more file sweeps than 8192 (not relevant at the served `--prompt-batch 8192`). `STRATA_MMQ_TC_GIB`
  and the stage size are untuned for MMQ.

## 0. Summary

**Where the time goes.** Every prompt batch runs layer by layer: the dense half for every token of the layer, then
the routed experts of that layer. File-tier reads for a layer start only when its expert phase starts, so the drive
sits idle while the dense half runs.

- **Small turns (+0.8-1K) are bound by the drive.** A batch of 844 tokens still routes 10.5-10.8K of the 15,360
  experts. Of those, 5.3-5.8K are in the file tier: 81-88 GB, read at 4.9 GB/s, so 17-18 s. The expert kernels (~10 s)
  overlap with those reads. The dense half (4.8-7 s in calm runs, 12.7 s in the live session (2026-10-28 07:05)) runs with the drive
  idle. This floor hardly moves with turn size: +844 tokens read 5.5K experts and +2.5K read 7.5K. That is why a +844
  turn gets only 26-38 tok/s.
- **Medium and large turns (+2.5-7K, and cold prompts) are bound by the expert kernel.** The prompt runs the decode
  kernel (`--no-chunk-mmq`: one warp per row, which re-decodes the Q3_K row for every token). That costs **46-50 µs
  per token-expert pair**, the same in every run, about 1.4 TFLOP/s. A 6.5K turn spends 56 s in it.
- **The dense half is mostly data movement, not GEMMs.** In a rocprof trace of a 2K prompt, 49% of the dense GPU time
  goes to copying the sparse-attention keys of every query. The chain is get_rows dequant NVFP4 → concat → pad →
  F32→F16. The Q8_0 GEMMs take 12%.

**Ranked lossless changes** (what each saves on a typical turn; §4 has the details):

| # | Change | +844 | +2.5K | +6.5K | cold 15K | Effort | Bit-identical? |
|---|---|---|---|---|---|---|---|
| 1 | **MMQ expert kernels for prompt batches** (fix the 8192 fault: id-buffer slack) | ~0 | **−11 s** (measured, at 4096) | **−20 s** (measured at 4096), ~−38 s at 8192 (est.) | ~−80 s (est.) | flags now at 4096; ~5 lines + test at 8192 | no (fp order; same int8 activations); needs the 17-row A/B |
| 2 | **Fused sparse-attention K gather** (NVFP4 + window rows → padded F16 in one kernel) | −2 s | −6 s | −11 s | −20 s | medium (one HIP kernel + graph change) | yes (same FA input, by construction) |
| 3 | **File reads off the dense critical path** (layer-ahead whole-layer prefetch for big batches; progressive / predicted reads for small ones) | −3…−6 s | −5…−10 s | −10…−15 s (measured −14.6 at 4096, MMQ) | −15…−25 s | medium | yes (same bytes) |
| 4 | **Batch width after the release** (wait for the driver's free figure to settle) | 0 | 0 | 0 | cold 5-8K: −10…−15 s (measured 2 sweeps vs 1); cold 34K: −30…−60 s | ~15 lines | yes, or batch-shape noise |
| 5 | **Persistent / multi-slot prefix cache** (park to RAM + disk) | – | – | – | repeated / resumed 34K: 398 s → ~2-5 s | medium | yes (exact state restore) |
| 6 | Faster file reads (file-offset order, no wave barrier; optional 2nd copy on a second drive) | −2…−4 s | −2…−4 s | −3…−5 s | −5…−8 s | small-medium | yes |
| 7 | Keep the arena extension through small turns; keep the DSpark weights when the batch fits; poll the driver at 20 ms | −0.5…−1.5 s | −0.5…−1 s | −1…−2 s | −1 s | env + ~30 lines | yes |

**Estimated TTFT per turn** (§5, estimates built from measured parts):

| turn | user today | after 1+4+7 | + 2 | + 3+6 |
|---|---|---|---|---|
| +844 on 5-25K | 32 s | ~23-25 s | ~21 s | **~14-17 s** |
| +2.5K | 49 s | ~41 s | ~34 s | **~20-26 s** |
| +6.5K | 93 s | ~55-57 s | ~45 s | **~30-38 s** |
| cold 15K | 165-174 s | ~90 s | ~70 s | **~50-60 s** |
| cold 34K (side request) | 398 s | ~225 s | ~180 s | ~110 s; **~2-5 s on a prefix-cache hit (#5)** |

The floor for small turns is the drive. A +844 turn has to read ~84 GB of routed file-tier experts, which takes
~14-16 s at 5.4-6.2 GB/s. Getting below that losslessly needs more of the experts in RAM.

## 1. Method, runs, caveats

- Driver: `scripts/dev/prefill/turns.py NAME SPEC.json [engine args]`. It starts the engine exactly as
  `start_xolotl_dsv41 --model 1017 --no-auth --long --dspark` does (same args and env, no HTTP front end) under
  `flock ~/.cache/xolotl/gpu.lock`, and sends GEN requests over the line protocol. `extend: true` means the previous
  prompt, plus its generated tokens, plus new real tokens (agent/code rows and the tool-result tail of `chain6.i32`).
  It measures client-side TTFT. A 1 s sampler logs the 990's MB/s, KFD `evicted_ms` and whether `hf_upload` is
  running. `report.py` prints the per-turn table: prompt phases, `MIMO_CHUNK_PROF` and `STRATA_DENSE_PROF` splits,
  µs per token-expert pair, ms per dense column-layer, and eviction seconds. `routes.py` analyses
  `STRATA_ROUTE_LOG`. `ktsum.py` sums a rocprofv3 kernel trace. `oneshot_prof.sh` is a one-shot run under rocprofv3:
  serve leaves by `_Exit`, which skips rocprof's finalisation, so serve runs cannot be traced.
- Engine runs, 8 in total, none of them rebuilt:

  | run | config |
  |---|---|
  | base1 | served config + profilers, 8 turns |
  | fixcols | `STRATA_ELASTIC_COLS=0` |
  | mmq4k | `--chunk-mmq --prompt-batch 4096` (+ fixcols) |
  | decA / decB | elastic vs fixed columns, 387-token decode |
  | mmqpf | mmq4k + `STRATA_CHUNK_PF_MIN=2048` (whole-layer prefetch) |
  | prof | serve under rocprof; no trace, because of `_Exit` |
  | trace2k | one-shot 2K under rocprofv3 |

  The server was stopped throughout and is stopped now.
- **Perturbation.**
    - An HF upload ran concurrently from base1, request 2, through fixcols, mmq4k, decA and part of decB. It read a copy
    on a second drive, not the 990: the 990's read
    MB/s and the reader GB/s were unchanged (4.9 GB/s in every run).
  - KFD queue evictions vary a lot from run to run. base1 logged 7-30 s of `evicted_ms` per turn and kcompactd was
    active; decA/decB logged 0.03-0.6 s. base1 requests 2 and 3 (+2519@16K dense 41.6 s, +6555@18K 124.9 s) are
    perturbed and are not used as baselines.
  - Treat dense seconds as noisy. Kernel µs/pair (GPU events) and reader GB/s were stable across all runs.
- The live session (engine log from 07:05) has no eviction data. Its small-turn dense times (12.7-15.3 s for
  0.8-1K tokens) are about 2x the calm runs at similar depth (4.8-8.6 s). The most likely cause is the same stall
  source, but that is not proven.
- Measured and estimated numbers are kept separate (§7).

## 2. Where prefill time goes

### 2a. The live session (measured, engine log 2026-10-28 07:05, MMVQ, batch 8192)

| turn | ctx before | TTFT (prompt s) | dense s | experts s | file reads | token-expert pairs | µs/pair (experts/pairs) |
|---|---|---|---|---|---|---|---|
| cold 34,372 (cancelled) | 0 | 397.7 | 115.0 | 280.2 | 46,918 | 4,624,320 | 61 |
| cold 5,764 | 0 | 81.5 | 23.7 | 56.1 | 8,019 | 1,014,720 | 55 |
| +2,263 | 5.9K | 43.7 | 13.4 | 29.7 | 5,433 | 543,120 | 55 |
| +2,519 | 8.4K | 49.4 | 15.1 | 33.8 | 6,638 | 604,560 | 56 |
| +6,556 | 11.0K | 92.8 | 29.1 | 61.9 | 7,855 | 1,143,360 | 54 |
| +3,524 | 17.8K | 70.5 | 24.7 | 45.2 | 7,045 | 784,320 | 58 |
| +1,049 | 21.6K | 34.5 | 15.3 | 17.9 | 5,237 | 251,760 | 71 |
| +844 | 22.9K | 32.0 | 12.7 | 17.8 | 4,780 | 202,560 | 88 |

The other parts of a turn are small: logits 0.01 s, DSpark seed 0.02 s, DSpark reopen 0.9-1.2 s (only on ≥4096-token
turns, `STRATA_SERVE_DSPARK_EVICT=4096`), and the driver wait after the extension release 0.25 s. **The cold 34K ran
at batch 4,820, not 8,192**: "VRAM 9.77 GiB free → prompt batch 4820". The driver's free figure lagged the DSpark and
extension release (§4, #4), so it took 8 sweeps of the file tier instead of 5.

### 2b. Breakdown with the profilers (measured)

`MIMO_CHUNK_PROF` and `STRATA_DENSE_PROF`, served config (MMVQ, batch 8192). Calm runs, except where an eviction
figure is shown:

| run / turn | TTFT | dense (GPU) | experts | expert kernels (GPU) | read waits | file GB | reader GB/s | µs/pair (kernel) | evicted |
|---|---|---|---|---|---|---|---|---|---|
| base1 cold 15K (2 batches) | 174.3 | 55.3 (53.1) | 113.7 | 102.5 | 10.5 | 187.3 | 1.84 (paced by kernels) | 49.1 | 8.6 s |
| decA +844@2K | 23.6 | 4.8 | 17.9 | – | – | ~78 | – | – | 0.03 s |
| fixcols +844@5.6K | 24.5 | 5.3 (4.7) | 18.1 | 10.2 | 12.4 | 83.9 | 4.92 | 50.5 | 1.7 s |
| fixcols +844@15.6K | 27.2 | 7.0 (5.9) | 19.2 | 10.2 | 12.9 | 87.7 | 4.93 | 50.5 | 3.2 s |
| base1 +844@25K | 28.2 | 8.6 (6.3, +2.3 alloc) | 18.6 | 9.8 | 11.3 | 81.2 | 4.89 | 48.1 | 4.0 s |
| fixcols +2519@6.5K | 52.8 | 15.4 (14.2) | 36.5 | 27.8 | 5.9 | 114.4 | 3.38 (paced) | 45.9 | 1.9 s |
| fixcols +6555@9K | 95.7 | 28.2 (27.3) | 64.8 | 55.9 | 7.6 | 125.0 | 2.10 (paced) | 48.9 | 2.0 s |
| fixcols cold 5.6K | 85.2 | 23.8 | 59.8 | 47.4 | 7.5 | 125.5 | 2.27 (paced) | 45.5 | 0.9 s |
| base1 cold 5.6K, **capped at 4693** | 99.7 | 29.5 | 65.8 | 48.0 | 17.8 | 201.9 (2 sweeps) | 3.33 | 46.0 | 7.4 s |

How a layer of a small turn runs (+844): the dense half takes ~130-220 ms with the drive idle. Then the expert phase
streams that layer's routed file-tier experts, ~138 × 15.2 MB = 2.1 GB, in ~0.43 s at 4.9 GB/s, while the arena
experts' DMAs and the grouped kernels (~250 ms) run underneath. ~0.6-0.7 s × 40 layers comes to ~25 s.

**Why +844 is only 26-38 tok/s.** Almost all of it is fixed per-turn cost:

1. The routed file-tier reads. 844 tokens × 6 × 40 = 202,560 picks land on 10.5-10.8K distinct experts, and 5.3-5.8K
   of those are in the file tier (55-60% of it). That is 81-88 GB, 17-18 s of reader wall time. 22% of these reads
   serve a single token and 47% serve fewer than 4 (`routes.py`). A +2.5K turn reads only 37% more (7.5K) and a +6.5K
   turn 50% more (8.2K).
2. The dense half with the drive idle: 4.8-7 s in the calm runs, 12.7-15.3 s in the live session.
3. ~1 s of overheads: driver wait 0.25 s, and graph alloc 0.5-2.3 s in `STRATA_DENSE_PROF`.

The arena extension is released before every batch, so 223-540 conversation-hot experts that were in RAM go back to
the file tier for that turn. It is rebuilt at decode start (1.5-5.9 s "on the side"), which is where the
registration evictions come from. The DSpark tail is not a cost any more: its seed rows come off the batch (0.02 s).

### 2c. Dense half, by kernel (measured: rocprofv3, one-shot 2,048-token prompt, MMVQ, `trace2k`)

Dense GPU time is 10.2 s, and the engine's split says dense 11.1 s.

| kernel | s | share | what it is |
|---|---|---|---|
| `k_get_rows_kq<dequantize_nvfp4>` | 2.35 | 23% | gathers each query's K=512 selected compressed rows from the NVFP4 cache into F32 `[512, 512, n]` |
| `flash_attn_tile<512,512,1,8>` (+ combine 0.20) | 1.53 | 15% | the attention itself |
| `pad_f32` | 1.00 | 10% | pads the key count 640 → 768 (`M % 256`) |
| `concat_cont` | 0.86 | 8% | window rows ++ compressed rows (`Kall`) |
| `cpy_scalar_contiguous<float,half>` | 0.77 | 8% | ggml's FA converts the F32 K/V to F16 |
| `mul_mat_q` + `quantize_mmq_q8_1` | 1.25 | 12% | **all the Q8_0 dense GEMMs** |
| `Cijk_…SB` (rocBLAS SGEMM) | 1.03 | 10% | F16/BF16-weight mul_mats at F32 precision (fidelity switch `STRATA_DBG_PREC32`; keep) |
| hc pre/post, rms_norm, rope, fake-quant, indexer, copies | ~1.4 | 14% | |

**49% of the dense half (5.0 s of 10.2) materialises a per-query copy of the attention keys four times over**: gather
F32, concat, pad, cast to F16. That works out to ~0.8 MB of F16 keys per token per layer. The Q8_0 GEMMs are only 12%.
Dense time per column-layer is 0.14-0.16 ms on cold/shallow batches. On small deep turns it is 0.21-0.31 ms: the
332-column tail sub-batch, the gallocr growth (`alloc` 0.5-2.3 s per turn), and the ratio-1 layers' `next_pow2`
`pass_cap` (16384 → 32768 past 16K depth) all add to it. +844 measured dense 4.8 s @2K, 5.3 @5.6K, 7.0 @15.6K and
8.6 @25K (6.3 GPU + 2.3 alloc).

### 2d. Expert kernel (measured)

- `strata::kernels::native_gu/down_amd_kernel<11>` takes 26.4 s for the 2K prompt (99.8% of the expert GPU time).
- MMVQ costs 46-50 µs per token-expert pair in every run, whatever the batch width: 70.8 MFLOP per pair, about
  1.4 TFLOP/s. The kernel re-reads and re-dequantises each Q3_K row for every entry of its group (UMA_DEEP_DIVE §3).
- **MMQ (`--chunk-mmq`, ggml `mul_mat_q` with WMMA on RDNA3.5) costs 7.5-8.0 µs/pair on 2.5K+ batches, and 14.5 µs/pair
  on 844 tokens: a 3.4-6.1x reduction.**

## 3. How close each case is to its floor

The floors use the 990's ceiling on this read pattern:

- `nvbench`, 16 random blobs per batch, 32-64 threads: **5.3-5.4 GB/s**.
- The whole-layer prefetch reader, 16 threads in file order: **6.0-6.2 GB/s** (measured in mmqpf).
- The engine's on-demand reader: 4.9 GB/s.

The other inputs are MMQ kernels at 8 µs/pair (14.5 for small batches), and dense as measured, or with the fused
gather (#2, est. ×0.6).

| case | file GB | read floor (5.4-6.2 GB/s) | MMQ kernels | dense (now → #2) | floor ≈ max(dense + kernels, reads) | measured now (calm) |
|---|---|---|---|---|---|---|
| +844 @5-25K | 81-88 | 13-16 s | 2.9-3.0 | 5-7 → 3-4 | **~14-16 s (drive)** | 23.6-28.2 |
| +2,519 @6.5K | 114.6 | 18.5-21 s | 4.5 | 15.4 → 9.3 | **~19-21 s (drive)** | 52.8 |
| +6,555 @9K (1 batch) | 125 | 20-23 s | 9.0 | 28.2 → 17 | **~26-28 s (GPU)** | 95.7 |
| cold 15K (2 batches) | 187 | 30-35 s | 16.5 | ~50 → 30 | **~47-50 s (GPU)** | 174 |
| cold 34K (5 batches at 8192) | ~530 (est.) | 86-98 s | 36.5 | 115 → 70 | **~105 s (GPU ≈ drive)** | 398 |

Today small turns sit 1.6-1.8x above their floor and big turns 3-3.6x above theirs. The gap on big turns is the
kernel. On small turns it is the drive sitting idle during the dense half, plus the reader's 4.9 vs 5.4-6.2 GB/s.

## 4. Lossless changes, ranked

### #1 MMQ expert kernels for prompt batches (largest gain on ≥2K turns and cold prompts)

- **Measured.** mmq4k (`--chunk-mmq --prompt-batch 4096`) against fixcols (same everything else, batch 8192):

  | turn | MMVQ @8192 TTFT | MMQ @4096 TTFT | change |
  |---|---|---|---|
  | cold 5.6K | 85.2 s | 70.2 s | −15 s, even with 2 sweeps (13.6K vs 8.3K reads) |
  | +844 | 24.5 s | 27.2 s | ±2 s noise; read-bound either way |
  | +2519 | 52.8 s | 42.1 s | −10.8 s |
  | +6555 | 95.7 s | 75.6 s | −20 s, with 2 sweeps |
  | +844@15.6K | 27.2 s | 27.2 s | 0 |

  With MMQ, the expert phase becomes ≈ the reader wall time.
- **At 8192, once the fault is fixed** (est.): +6555 ≈ 28 dense + ~26 reads + ~2 = **~56 s**, and cold 15K ≈ 50 +
  38 + 3 = **~90 s**. Batch 4096 is usable today, but it doubles the file sweeps of prompts in the 4-8K range.
- **Why MMQ faults at 8192 (hypothesis; not tested, since that needs a build).** `moe_mmq.cu`'s own comment says the
  MMQ kernels read up to `J_best − 1` (≤127) ids past the end of the id buffer, and `prefill.cpp` allocates
  `T*K + 128` for that reason. `gpu_run_chunk` allocates `c_dst` with exactly `n*K` ints and `m_iota` with exactly
  `cap` ints, so the down product's last sub-run reads past the end of `c_dst`. At n=8192 that buffer is 196,608 B, a
  multiple of 64 KiB, so plausibly nothing mapped follows it. At 4096 it is 98,304 B, which rounds up and leaves
  slack. The fix is to pad both buffers. Test with the 8192 one-shot that faulted (`mul_mat_q<Q3_K,128>`).
- **Bit-identity: no.** The int8 activations are the same q8_1 values, but the float accumulation order differs.
  - 3 of the 5 greedy 8-token continuations were identical.
  - +2519 differed at its first token (671 vs 17). That is a near-tie candidate, but it was not checked.
  - +844@15.6K differed after that, as expected: its context contains the other tokens.
  - Upstream measured equal ppl on CUDA (2.3790 vs 2.3799).
  - **Gate before defaulting:** the 17-row same-text A/B (ppl gap, top-1) and verify_setup.
- The lab log's "MMVQ is also faster here: 8K prefill 63.5 vs 58 tok/s" (2026-10-20, the first drive) compared MMQ at batch
  4096 with MMVQ at 8192, on a 2.9 GB/s drive. The MMQ side paid a second file sweep (41.7 s of file waits). Its
  kernel time was already ~2x better (17.8 vs 31.5 µs/pair at the time).

### #2 Fused sparse-attention K gather (largest gain on the dense half)

- Replace `get_rows(comp, sel)` (NVFP4 → F32) → `concat(Kw, Kc)` → `pad` → (FA's internal F32→F16) in `build_attn`
  (`ds41_dense.cpp:704-734`) with one custom op. It writes `Kall` as F16 `[DH, M_pad, n]` directly from `L.raw` /
  this pass's window rows and the NVFP4 `comp` rows at `sel`, with zero pad rows. ggml's FA then gets F16 K/V and
  skips its conversion.
- **Bit-identical by construction:** the same values, the same F32→F16 rounding, the same FA kernel and inputs.
  Verify with `STRATA_SERVE_DUMP` and a logit md5.
- **Estimate.** 5.0 s of the 2K prompt's 10.2 s goes to ~0.4-0.6 s: one F16 write of 0.8 MB per token-layer at
  ~200 GB/s. That is about **−40% of dense**: +844 −2 s, +2.5K −6 s, +6.5K −11 s, cold 15K −20 s, cold 34K −45 s.
- The next step, not costed here, is an FA kernel that gathers K by index from the cache, with no materialisation
  at all.

### #3 File reads off the dense critical path

1. **Big batches (≥ ~2K rows: they route 75-85% of a layer's file tier).** Use the existing whole-layer prefetch,
   `STRATA_CHUNK_PF_MIN`: layer L+1's whole file tier is read into a pinned slab while layer L computes.
   - **Measured** (mmqpf, MMQ@4096): +6555 TTFT 75.6 → **61.0 s**. Read waits fell from 36.6 to 6.8 s, and the slab
     reader ran at **6.0-6.2 GB/s** against 4.9 on demand.
   - The cost: two 3.88 GiB `cudaMallocHost` slabs, held for the life of the engine, starved everything else. Small
     turns then hit the 20 s driver-free timeout and batches were capped at ~2,300 (+844 went to 53.8 s), so the run
     is no verdict on small turns.
   - **Make it UMA-safe:** carve the two slabs out of the arena extension's pinned pages. Those pages are released
     before every batch anyway and are already pinned and mlocked. Then nothing new gets pinned and the
     release/rebuild churn goes too. Only engage it for ≥ ~2048-row batches. Bit-identical: same bytes.
   - Estimated saving: big turns −10…−15 s with MMQ (with MMVQ the reads already hide behind the kernels, so −5…−8 s).
2. **Small turns.** A whole layer would be ~1.75x the bytes needed, so read routed experts early instead:
   - **Router-progressive:** start layer L's file reads right after each sub-batch's `attn_router_span` (the routing
     of the first 512 tokens covers most of the 844-token batch's set), not when `run_chunk_dev` starts. The reads
     then overlap the rest of that layer's dense half.
   - **Predicted:** during the dense half of layer L, read the file-tier experts the previous turn routed at layer L,
     most-used first. Measured with `routes.py` on the fixcols route log: the previous turn's routed set covers
     **94-97%** of a +844 turn's file reads, and **69-86%** for bigger turns. Precision vs the previous routed set is
     0.38-0.55; that is a lower bound, since its arena members would not be read.
   - Cap the predicted reads to the dense window, and read the exact remainder on demand.
   - A cross-layer predictor (layer L+1's router applied to layer L's output) would allow full-layer lookahead, but
     its hit rate is unmeasured.
   - Estimated saving: −3…−6 s on a +844 turn, which hides most of the 5-7 s dense half. Bit-identical: the reads
     only change timing.

### #4 Batch width after the release (cold prompts, side requests)

- The cap is computed from the driver's free figure right after `DSpark weights freed` + `arena ext: released`. That
  figure still lags at ≥ 9 GiB, so the 20 s wait loop does not trigger:
  - base1 cold 5.6K: "VRAM 9.65 GiB free → prompt batch 4693".
  - the live session's cold 34K: 9.77 GiB → 4820.
- **Measured:** the same cold 5.6K prompt ran 99.7 s capped (2 sweeps, 13.3K reads) and 85.2 s uncapped (1 sweep,
  8.3K). Part of that gap is eviction noise.
- Fix (`ds41_generate.cpp:1549-1560`): after a release, poll every 20 ms until two readings agree within
  0.1 GiB, or the free figure reaches what the batch needs at full width, with a 1-2 s timeout.
- Estimated saving: cold 5-8K prompts −10…−15 s; a cold 34K needs 5 batches instead of 8, −30…−60 s.
- Bit-identical except for batch-shape float noise: the per-token kernels are batch-independent, but the batch split
  changes which tokens share a pass.

### #5 Persistent, multi-slot prefix cache (side requests and resumed sessions)

- The 34K request that was cancelled after 398 s was an earlier conversation (tools 6, thinking on, not a sub-agent
  title call). `park` already saves a conversation's restorable state (`Ds41Dense::park`: snapshot rings + `comp` /
  `idxk` rows) to host memory, but there is only one slot and it is dropped as soon as a request reaches past it.
- Proposal:
  - N slots in RAM (LRU), each keyed by its token prefix. Store hist, the state rows, snap_logits and the DSpark seed.
  - Spill the slots to `~/.cache/xolotl/prefix/<hash>.bin` on the 990.
  - On a request, restore the longest stored prefix that the prompt extends (the same rule as SNAP_BACK), then prefill
    only the rest.
- Size, estimated: the compressed KV of the 4 kv sources (~1 row per position, NVFP4 512-d ≈ 288 B) plus the 8 index
  sources' MXFP4 keys (~68 B) plus the rings ≈ **50-100 MB per 34K conversation**. That writes in well under a second
  and loads in ~0.1 s.
- Estimated saving: a resumed or repeated 34K becomes a ~2-5 s TTFT instead of 398 s. Bit-identical: the same state
  bytes, as unpark already does.
- Effort: medium (~200-300 lines). Measure the park size first: log `parked` bytes.

### #6 Faster file reads

- **Measured headroom.** The on-demand reader runs at 4.9 GB/s. Random 16-blob batches reach 5.3-5.4 GB/s in
  `nvbench`, and the prefetch reader in expert-id (file) order reaches 6.0-6.2 GB/s.
- Fix: in `gpu_run_chunk`, issue a layer's routed file reads sorted by file offset, all classes at once. Replace the
  2-half wave barrier (`cudaStreamSynchronize(s_cp)` before every 140-blob wave) with a slot ring that frees a bounce
  slot as soon as its DMA is done.
- `STRATA_READ_SORTED` segfaulted on 2026-10-15 and was reverted. Re-implement it, don't re-enable it.
- Optional: a second drive holding a byte-identical copy of the files. A second reader pool on it (PCIe 4.0 x1, ≤ ~1.9 GB/s, with known slow phases) that pulls from the same per-layer queue could add ~25-30% of bandwidth.
  It needs work stealing so that a slow phase only costs its share, and it competes with the OS and HF upload traffic.
- Estimated saving: 10-20% of read time on read-bound turns (+844 −2…−3 s), up to −30% with the second drive.
  Bit-identical.

### #7 Small per-turn costs (env or few lines, all bit-identical)

- **Keep the arena extension through small turns.** `rebalance_begin_prompt(keep_ext)` already keeps it for
  by-step turns. Extend that to batches ≤ ~2048 rows whose `need(B)` fits next to it, about 4.3 GiB + 0.95 MiB/col.
  - The extension holds 223-540 conversation-hot experts in use.
  - Estimated saving: −0.5…−1.5 s of reads per small turn, and no rebuild with its registration evictions at decode
    start.
- **DSpark weights:** `ds41-engine-strix.sh` sets `STRATA_SERVE_DSPARK_EVICT=4096`, so every ≥4096-token turn frees
  the drafter and reopens it afterwards.
  - Measured reopen: 0.9-1.7 s, and 3.2-8.1 s in the perturbed runs.
  - With the drafter resident, a 6.5K batch needs ~11.4 GiB of the ~14.5-15 GiB free, so `STRATA_SERVE_DSPARK_EVICT=8193`
    should keep the full width. Check for a "prompt batch N (of 8192)" line on the first 6-8K turn.
- **Driver poll:** the wait loop sleeps a whole 250 ms before its first re-check. Every batch turn logged "after
  waiting 250 ms". Poll at 20 ms instead: −0.2 s per turn.

### Examined and not recommended (or no gain shown)

- **`STRATA_ELASTIC_COLS=0` (fixed state columns).** The same tokens came out.
  - In base1 vs fixcols it looked like −5 s per turn, but calm A/B runs show no difference: decA vs decB +844 TTFT
    was 23.55 vs 23.74 s, and decode 9.70 vs 9.46 tok/s.
  - The apparent gain was the eviction noise in base1.
  - It would only matter for the graph-allocator churn (`set_columns` frees both gallocrs every turn), which is
    better fixed by keeping the pass allocator's buffer.
  - Not recommended without a churn-specific A/B.
- **by_step threshold** (`STRATA_SERVE_STEP_MAX=96`, `BATCH_MIN=128`): the pass path costs ~decode speed per token,
  so it is only right below ~100-200 tokens. It does not apply to 0.8-7K turns.
- **Smaller batches or sub-batching short turns:** every extra batch is another sweep of 5-8K file reads. Short turns
  already read only their routed experts.
- **The `c_stage` double buffer.** The arena experts' DRAM→DRAM copies run as `__amd_rocclr_copyBuffer` blits:
  0.27 s per 2K prompt in the trace, overlapped. That is low priority until the expert phase is kernel-bound again.
  Earlier prompt zero-copy was "not bit-identical"; the cause is unknown and worth a look if it is revisited.
- **Page cache instead of O_DIRECT:** there is no spare RAM (an 80-86 GiB arena plus the 24-30 GiB reserve), and the
  HF upload churns the cache.
- **Ordering by arena residency:** already done (resident → arena DMAs first → file reads in the background).

## 5. What a user would see (estimates)

The model is TTFT ≈ dense + max(expert kernels, reads not hidden) + ~1-2 s of overheads, built from the measured
parts in §2-§3. Ranges cover the dense noise (calm vs the live session (2026-10-28 07:05)) and 70-100% read overlap.

| turn | user (measured) | calm baseline (measured) | 1+4+7 | +2 (fused gather) | +3+6 (overlap, faster reads) |
|---|---|---|---|---|---|
| +844 @5-25K | 32.0 s | 23.6-28.2 | 23-25 | ~21 | **14-17** |
| +1,049 @21.6K | 34.5 s | – | 24-26 | ~22 | **15-18** |
| +2,263 / +2,519 | 43.7 / 49.4 s | 52.8 (+2519) | ~38-41 | ~33-35 | **20-26** |
| +3,524 @17.8K | 70.5 s | – | ~47 | ~40 | **26-32** |
| +6,556 @11K | 92.8 s | 95.7 (+6555@9K) | ~55-57 | ~45 | **30-38** |
| cold 15K | 165-174 s | 174 (base1) | ~90 | ~70 | **50-60** |
| cold 34K | 398 s | – | ~225 | ~180 | ~110, or **2-5 s on a prefix-cache hit** |

#1 must pass the quality gate. #2, #3, #4, #5 and #6 each need a build plus a bit-identity check.

## 6. Patch sketches (not applied)

**#1 MMQ id-buffer slack** (`xolotl/tools/v41/engine/ds41_moe.cpp`, `gpu_run_chunk`):

```diff
-        ck(cudaMalloc(&gp.c_dst, sizeof(int32_t) * (size_t) ne), "c_dst");
+        // MMQ reads up to J_best - 1 (<= 127) ids past the end of its id buffer (moe_mmq.cu; prefill.cpp pads +128)
+        ck(cudaMalloc(&gp.c_dst, sizeof(int32_t) * ((size_t) ne + 128)), "c_dst");
+        ck(cudaMemset(gp.c_dst, 0, sizeof(int32_t) * ((size_t) ne + 128)), "c_dst zero");
...
-            ck(cudaMalloc(&gp.m_iota, sizeof(int32_t) * (size_t) cap), "m_iota");
-            mmq::iota((int32_t*) gp.m_iota, cap, gp.s);
+            ck(cudaMalloc(&gp.m_iota, sizeof(int32_t) * ((size_t) cap + 128)), "m_iota");
+            mmq::iota((int32_t*) gp.m_iota, cap + 128, gp.s);
```

Then `--chunk-mmq --prompt-batch 8192` on the warm 14.8K one-shot: no fault, plus the 17-row A/B.
`start_xolotl_dsv41` drops `--no-chunk-mmq`.

**#4 Driver free settle** (`ds41_generate.cpp`, before the `pb_cap` computation):

```diff
-                if (uma && frb < (size_t) 9 << 30) {
+                // strix-port: right after a release (extension / DSpark) the driver's free figure keeps rising for
+                // ~0.25-1 s even above 9 GiB (cold 5.6K: 9.65 GiB -> batch 4693; the live session's 34K: 9.77 -> 4820)
+                const size_t want_b = (size_t) ((4.3 + a.vram_margin_gib) * 1073741824.0 + 0.95 * 1048576.0 * std::min(dh, PB));
+                if (uma && frb < want_b) {
                     const double tw0 = now_ms();
                     const size_t fr0 = frb;
-                    while (frb < (size_t) 9 << 30 && now_ms() - tw0 < 20000) {
-                        std::this_thread::sleep_for(std::chrono::milliseconds(250));
+                    size_t last = 0;
+                    while (frb < want_b && now_ms() - tw0 < (frb < ((size_t) 9 << 30) ? 20000 : 1500)) {
+                        std::this_thread::sleep_for(std::chrono::milliseconds(20));
+                        last = frb;
                         dev_free(frb, totb);
+                        if (frb >= ((size_t) 9 << 30) && frb < last + (100u << 20) && now_ms() - tw0 > 300) break;   // settled
                     }
```

**#7 DSpark eviction threshold** (env only, `scripts/start_xolotl_dsv41` ENV_JSON for `--dspark`):
`STRATA_SERVE_DSPARK_EVICT: "8193"`.

**#3a Prefetch slabs from the extension's pages** (`ds41_moe.cpp`):
- In `rebalance_begin_prompt`, on a batch of ≥ `STRATA_CHUNK_PF_MIN` rows, do not `arena_ext_release`. Instead, mark
  the extension slots as staging (`ChunkStage::slab[0/1] = ext_base` and `ext_base + ext_bytes/2`, `cap` =
  `ext_bytes/2`), and invalidate their `slot_of` entries, as the release does.
- `chunk_stage_alloc` returns early when the slabs are borrowed. `release_chunk` hands them back as an empty
  extension, which the next decode refills with no new `cudaHostRegister`.
- A 10 GiB extension gives two 5 GiB slabs, enough for the largest routed layer's file tier (3.88 GiB measured).

**#3b Router-progressive reads** (new tier API; `ds41_generate.cpp` layer loop):

```cpp
// after each attn_router_span(l, s, sn):
tier.read_ahead(l, sn, dense.batch_ids() + (size_t) s * top_k);   // queue this sub-batch's not-yet-queued
                                                                  // file-tier experts of layer l (reader pool,
                                                                  // per-layer staging ring of pinned slots)
// gpu_run_chunk: a file-tier expert already in the ring -> DMA from there (skip its read; wait on its ready flag)
```

The ring can live in the same borrowed extension pages. The predicted variant also queues, before
`attn_router_span(l, 0, …)`, the previous turn's file-tier experts of layer l by decreasing token count, until the
dense window closes.

**#2 Fused gather** (`ds41_dense.cpp` `build_attn` + a small ggml op in a `strix-NN-*.patch`; the op contract):

```
dsv41_gather_kv_f16(raw_window_rows  [DH, RAW+n] f32,  widx [SWA, n] i32,
                    comp (NVFP4)      [DH, rows],       sel  [K, n]   i32,
                    M_pad)  ->  Kall  [DH, M_pad, n] f16   // rows 0..SWA-1 window, SWA..SWA+K-1 compressed, rest 0
```

The mask concat stays as is. The FA call takes the F16 `Kall` for both K and V, as now. Check: `STRATA_SERVE_DUMP`
logits md5 before and after.

## 7. Measured vs estimated

**Measured** (logs in `~/.cache/xolotl/runs/prefill/`; the live session (2026-10-28 07:05) in `~/.local/state/xolotl/logs/ds41-engine.log`):
- Every number in §2a-§2d.
- The MMVQ µs/pair (46-50) and MMQ µs/pair (7.5-8.0, and 14.5 at 844 rows).
- Reader 4.9 GB/s; prefetch reader 6.0-6.2 GB/s; `nvbench` 5.3-5.4 GB/s.
- The mmq4k, mmqpf and fixcols / decA / decB TTFTs.
- Greedy-token comparisons:
  - fixcols = base1: 3/3 identical.
  - decA = decB: 2/2, incl. 387 decode tokens.
  - MMQ vs MMVQ: 3/5 identical, one first-token difference.
- The capped batch 4693 and its 2 sweeps.
- The route-log recall of 0.94-0.97.
- The trace2k kernel table.

**Estimated:**
- The MMQ@8192 TTFTs (needs the fault fix).
- The fused-gather cost (~0.4-0.6 s per 2K prompt).
- The read-overlap gains: no build was made, and mmqpf measured only the big-batch half, with a confound.
- The second-drive gain.
- The prefix-state size (50-100 MB per 34K).
- The 34K projections: no 34K run was made.
- §5 as a whole.
- The MMQ fault cause: a hypothesis from code reading.
- Why the live session had 2x the dense time of the calm runs: likely KFD stalls, not reproduced in the calm runs.

**Perturbed and not used as baselines:** base1 +2519@16K (30 s evicted) and +6555@18K (23 s evicted). The small turns
in mmqpf were starved by the 7.8 GB of slabs.
