# `--vision`: image input for DeepSeek-V4.1-Flash-1017 as a launch option (plan)

Status 2026-11-06: research and design only, no repo code changed. **Section 0a has CPU-encoder measurements from
the same day (server stopped); they replace several (est.) figures below.** For the design part, the running server (port 8150) was left
alone; only its log, `/proc/meminfo`, the safetensors/GGUF headers and the code were read. "(est.)" = not measured.
This updates `vision_todo.md` (2026-10-25) against the current code. It does not repeat that file's architecture
research (sections 2, 4, 5 there still hold). Corrections to that plan are marked **[corr]**.

## 0. Summary

- **Recommendation:** option B (CPU torch encoder = DeepSeek's own `vision.py`, BF16 weights read straight from
  checkpoint shards 1+2) behind `scripts/start_xolotl_dsv41 --vision` (off by default). Do option D (GPU via mtmd)
  later, only if encode time turns out to matter. The LLM prefill of the span (184-1,001 tokens) costs more than the
  encode anyway.
- **Effort (est.):** 5-8 working days to a validated first `--vision`: encoder 0.5-1 d, engine 2.5-4.5 d, server +
  start script 1-1.5 d, plus 1-2 d of machine time for the gates. The teacher top-1 check comes after, in hours of
  machine time.
- **Memory:** the CPU encoder takes ~1.5-1.8 GiB of resident RSS (est.). It loads before the engine, so the
  UMA arena shrinks by that much on its own (79.6 -> ~77.9 GiB, est.). Raise the arena extension's floor 9 -> 12 GiB,
  so that an encode, which runs while the engine is idle but still holds its +10 GiB extension, never pushes the box
  into swap/compaction. Those are what cause the KFD queue-eviction stalls (UMA_DEEP_DIVE 2b). `--uma-reserve-gib`
  stays at 30.
- **Bit-identical:** with `--vision` off nothing changes: no encoder, no env change, and every new engine path stays
  dormant until a request has image positions. With `--vision` on, a text-only request should give identical tokens
  (only expert tiering moves). Both cases are to be proven with verify_setup + serve logit dumps (section 3.4).

## 0a. Measured 2026-11-06: the CPU encoder (option B)

DeepSeek's reference `vision.py` ViT + Aligner, run unchanged on the CPU in the `xolotl-quant` toolbox (torch
2.11.0+rocm7.13, GPU hidden). BF16 weights come straight from shard 1, and the model is built the way `generate.py`
builds it: bf16 Linear, fp32 RMSNorm. The server was stopped and the box idle (MemAvailable 120 GiB). The images are
synthetic PNGs: encode cost depends only on the patch grid, not on the content. Scripts and logs are in
`~/.cache/xolotl/runs/vision/`: `bench_enc.py` (`ref_sweep.jsonl`, `threads.jsonl`), `prof_enc.py` and
`attn_variants.py`/`.log`.

**Load:** 485.25M params. Shard 1 is read in 0.2-1.0 s and the model is built in 1.6-2.4 s. Process RSS is
**2.20 GiB** after load (torch alone 0.53): more than the 1.5-1.8 GiB estimated in section 2.

**Encode, reference path, 16 threads** (2 reps each, within ±3%; 32 threads was 10-20% slower):

| image | patches | span tokens | encode | peak RSS above load |
|---|---|---|---|---|
| ≤ 544x544 | 1,521 | 184 | 6.6-6.9 s | +0.02 GiB |
| 800x600 | 2,494 | 317 | 9.2-9.6 s | +0.39 GiB |
| 1024x768 | 4,070 | 496 | 13.6 s | +2.0 GiB |
| 1280x800 | 5,336 | 642 | 19.4 s | +3.4 GiB |
| 1920x1080 | 8,418 | 968 | 38.6-38.9 s | +9.5 GiB |
| square 1302 | 8,649 | 994 | 40.7-40.8 s | +10.1 GiB |
| 4000x3000 | 8,748 | 1,001 | 40.8-41.2 s | **+10.3 GiB** |

- **The estimate was 3-4x too optimistic** (vision_todo said 2-4 s typical, 6-12 s max). The Linear layers run fast:
  a bf16 `F.linear` microbench reached 3.9 TFLOPS, against 1.9 in fp32. The time goes into attention instead. With
  the reference's **3-D** q/k/v, torch runs `_scaled_dot_product_attention_math` (bmm + fp32 softmax + mask ops),
  which takes 84-90% of the encode at 1024x768 (torch.profiler).
- **Memory:** that math kernel materializes the scores, +10.3 GiB at max size. Encodes run while the engine idles
  with ~12 GiB available, so an unchanged reference path **would push the box into swap**. The peak has to be capped
  (next table).
- Deterministic: two encodes of the same image are byte-identical (all sizes).

**Attention variants inside the same ViT** (only the SDPA call is wrapped; output compared with the reference):

| variant | 1024x768: time / peak | 4000x3000: time / peak | vs reference |
|---|---|---|---|
| reference (3-D math) | 13.6 s / +2.0 GiB | 40.8 s / +10.3 GiB | – |
| **4 heads at a time** (same 3-D math call) | 23.9 s / +0.28 GiB | **50.4 s / +2.3 GiB** | **bit-identical** |
| 1 head at a time | 63.7 s / +0.03 GiB | 91.1 s / +0.39 GiB | bit-identical |
| query blocks of 1024 | 26.6 s / +0.28 GiB | 94.3 s / +1.1 GiB | bit-identical |
| 4-D SDPA (flash kernel) | 63.1 s / +0.03 GiB | 273.3 s / +0.04 GiB | **not** identical (cos min 0.991-0.994) |

- Suggested rule (bit-identical either way): the plain reference call up to ~4,100 patches (peak ≤ +2.0 GiB), 4
  heads at a time above that (peak ≤ +2.3 GiB, +24% time at max). The worst-case resident encoder is then
  2.2 + 2.3 ≈ **4.5 GiB**. Section 2's extension floor of 12 GiB (9 + 3) still covers it.
- torch's 4-D flash path on the CPU is slow here and changes numerics: rejected.

**bf16 vs fp32 compute (decision 4):** the same bf16 weights computed in fp32 took 18.9 s at 1024x768 (bf16: 13.6 s)
and 41.2 s at 1920x1080. Against the bf16 reference rows: mean cosine 0.9992-0.9995, but **min 0.86-0.95** (relative
error median 1.8%, max 38-52% on a few rows). So "fp32 = more accurate" does not mean "closer to the reference".
Keep bf16, which is the reference path. The section 4 V1 gate (per-row cos ≥ 0.999) is only meaningful against the
same bf16 code, and `ds41_vision.py` runs that code, so V1 should come out bit-identical rather than just above 0.999.

**What this changes in the plan:**
- Image latency on the CPU: encode **7-14 s typical, ~50 s max** (with the 4-head cap), plus the span's LLM prefill
  (6-45 s, est.). The encode is no longer small next to the prefill.
- That makes option D (GPU encoder via mtmd), or a faster CPU attention, worth more than section 3.1 assumed. B is
  still the fidelity reference and the quickest way to a working `--vision`.
- Resident RSS is 2.2 GiB, not 1.5-1.8: the arena shrinks 79.6 -> ~77.4 GiB (est. from the measured RSS).

## 1. Current-code check of vision_todo.md section 6

Verified 2026-11-06 in the tree. The checkpoint still has everything: shard 1 = `vision.*` 259 tensors (411,842,560
params) + `aligner.*` 4 (73,410,560), all BF16. Shard 2 holds `image_start/end/newline` (BF16 [5120]). The served GGUF
has `blk.{0..39}.exp_probs_b_vl.bias` [384]. The MTP sidecar `V41-1017-MTP.IQ3_XXS.gguf` has `mtp.{0..2}.ffn.gate.bias_vl`
but no vision tensors (built with `--no-vision`). Token `<｜deepseek_image｜>` = 129264, token type 4 (special) in
`/mnt/fast/V41-1017-GSQ/tokenizer`.

| # | todo item | where it is now | what the new code adds |
|---|---|---|---|
| 1 | `GENI` in the serve loop "~l.1275" | `ds41_generate.cpp:1296-1321`: the loop accepts only `GEN ` (1299, else `ERR unsupported command`). Words without `=` become the csv (1306-1308) | GENI has two bare words (SVE1 path, then csv). The reply must come before the server deletes the file (`server.py:4286`), so the file is read synchronously. Unknown `k=v` keys (`img_keys=`, `ckpt=`) are already ignored |
| 2 | embedding override in `upload_inputs` "~l.1279" | `ds41_dense.cpp:1314-1325` (host_emb from the F16 rows on the mmap) | **Every** prompt path goes through `upload_inputs(tids, n, pos0)` with absolute positions: `begin()` 1412-1416 (steps, `pass_m`, the DSpark tail `pass_v`, SNAP_BACK tail passes, the resume re-feed) and `begin_batch()` 1440-1443 (layer-major batch, full width up to 8192, sub-batches by `col0`). So the hook is a **position-keyed overlay** in Ds41Dense (`pos -> row`), set per request and checked in that one loop. Batch splits, enc-skip and SNAP_BACK cuts inside a span are all fine: the reference's "single chunk" is only how it is implemented (`model.py:1258`) |
| 3 | router "l.762, 922, 956, 965, every place" **[corr]** | The selection that changes numerics is only `ds41_dense.cpp:797` (`s0 = sc + exp_probs_b.bias`, in `build_attn` 551). 957 / 991 / 1000 are next-layer *predictions* for prefetch (n = 1 decode, speed only) | Use an exact select, not `bias + vl*(bias_vl-bias)` (fp rounding): a per-column bias table, or `bias*(1-f) + bias_vl*f` with f in {0,1}. Build it only when the pass has image columns. Such a pass must not use the cached n = 1 graph (`GKey`, 1609-1613): build it uncached. Text graphs stay byte-for-byte the same. The tier (`ds41_moe.cpp`) takes the ids from the dense half: no change. PredStage reads (`STRATA_PF_PRED_GIB`) only guess from the previous batch's routing (speed only) |
| 4 | Engram DEAD + gate | hash: `ds41_engram.cpp:248-268`; gate: `ds41_dense.cpp:563-581` (build_attn) **and 879-890** (`build_attn_kv_only`, the enc-skip phase A) **[corr: two graph sites]** | (a) `set_token` stores DEAD for image positions. (b) `hash()` must block **cumulatively** like `engram.py:173-176` (`blocked |= source == DEAD`, then pad for every older shift). Today it pads only for `pos-k < 0`, which is the same thing for text. (c) gate × 0/1 mask, only in graphs with image columns. (d) **new [corr]:** `hist_` is not in snapshot/restore/park (`ds41_dense.cpp:1814-1874`). Re-set the 3 positions under every resume point (max n-gram 4, config `engram_max_ngram_size`) from the engine's own (id, key) history. That is a no-op for text (same values) |
| 5 | reuse keys | LCP `ds41_generate.cpp:1322-1323`; park `1329-1360`, `park_hist` 1406; DSpark resume 1370-1390 (keyed on `lcp`); snapshot/regenerate 1392-1412, 1747-1763; pushes 1903, 1905, 1929 | Keep a parallel `hkey` array (0 = text, else hash(img_key) mixed with the offset in the span) with every `hist` / `park_hist` write, and compare (id, key) in both LCP loops. DSpark resume, cancel-reuse, snapshot and SNAP_BACK all derive from `lcp`, so they come out right without further changes. **New:** the lookup drafter (`STRATA_SERVE_LOOKUP=14`, the wrapper default; 1854-1866) copies from `hist`, so it must skip `hkey != 0` positions. Otherwise it proposes runs of 129264: they are verified, so this is not wrong, only wasted |
| 6 | DSpark "follow the reference" | `ds41_dspark.hpp` header; `ds41_dspark.cpp:428-431` | **No drafter change needed.** It embeds only [bonus, 4 x noise] (`model.py:1131-1137`; ours 428-431), never prompt ids. Images reach it only through `main_hidden` (HC-mean of the trunk entering layers 37-39). Those rows are already image-conditioned when the trunk is right. That covers both the window seeding (`bcap_vh` off the batch / `pass_v` tail, 1660-1700) and `snap_seed` for regenerate. Its MoE uses the text bias, as `model.py:1128` ("drafts are text: no VL bias"). Verify passes are decode positions: text, vl = 0. Acceptance on image answers may be lower: measure tokens/pass, it never changes the output |
| 7 | text bit-identical | | See 3.4. Every new branch is gated on "this request has image positions" |
| S | server | `to_encoder_messages` `serve_ds41.py:117` (flattens with `_text`, 111, joined with ""); `Vision` `server.py:2275`; `prepare()` 3997-4093; expansion test `t == pad and ids[j-1] == start` at ~4040; GENI 1582; `input_modalities` 5565; `img_keys` 4087 | (a) Only messages that contain image items get DeepSeek's block form (`image_url` -> placeholder, texts joined with "\n\n", `dsv41_encoding.py:319-389`). Messages without images keep `_text()`, so text prompts stay byte-identical. (b) One hook in `prepare()`: `VISION_START = None` means every 129264 is an image (a typed placeholder is already a 400 from `_validate_no_image_sp_tokens`, 351). (c) **Confirmed:** encodes run inside the request FIFO (`with self._turn(cancel)`, 4027-4033; `_turn` = `self.fifo`, 3952-3964), so never during a GEN. The engine still runs its idle arena-rebalance reads (CPU/NVMe) meanwhile |

## 2. Memory on UMA

Measured (engine log 2026-11-06 02:05, `/proc/meminfo` now): MemAvailable after the dense half 109.6 GiB -> arena
79.6 GiB (reserve 30), mlocked. Then DSpark 3.28 GB and pred reads 1.49 GiB pinned. The arena extension `+706 slots
(10.00 GiB)` is taken in decode at MemAvailable 22-23 GiB and released before each prompt batch (12 -> 21-22 GiB).
Idle/decode MemAvailable = **~12 GiB**, Mlocked 89 GiB, swap 7 GiB (stays on). Extension size = min(10,
MemAvailable - `STRATA_ARENA_EXT_FLOOR_GIB` (default 9), driver free - 1.5) (`ds41_moe.cpp:2399-2438`). Engine floor 8
GiB (`ds41_generate.cpp:387`).

| item (option B, CPU, BF16) | size | where |
|---|---|---|
| encoder weights | 485.3M x 2 B = 0.90 GiB | process RSS (pageable, not GTT) |
| python + torch runtime | ~0.5-0.8 GiB (est.) | RSS |
| activations, typical 1024x768 (4,070 patches) | ~0.3-0.6 GiB (est.) | transient |
| attention at max (8,748 patches) **if SDPA falls back to the math kernel** | 16 x 8748^2 x 2 B = 2.45 GB scores (+ softmax copy, ~4.9 GB) per layer | transient. `vision.py:59` passes **3-D** q/k/v. Torch's CPU flash kernel may need 4-D (not verified) -> wrap with `unsqueeze(0)` or chunk queries in our process |
| `.sve` cache, 64 images x up to 20.5 MB | up to 1.3 GB | `Vision.work_dir` = `tempfile` -> **/tmp is tmpfs here (61.7 G)**, i.e. RAM |
| per-request combined file | ~10-20 MB / image | /dev/shm, deleted after the run (`server.py:111-134, 4286`) |

What `--vision` must change:
- **Resident encoder** (the server's default): it starts and loads before the engine (`server.py:6962-6976`, "loading
  the vision encoder" before "loading the model"). The UMA arena = MemAvailable - reserve shrinks by its RSS without
  any change: **79.6 -> ~77.9 GiB (est.)**. Keep `--uma-reserve-gib 30`.
- **Arena extension floor:** encodes run between requests, with the extension held and ~12 GiB available. A peak of
  3-5 GiB would drive the box toward swap-out/compaction, which UMA_DEEP_DIVE 2b ties to multi-second KFD queue
  evictions. So `--vision` sets `STRATA_ARENA_EXT_FLOOR_GIB=12` (est.: 9 + a 3 GiB encode peak). The extension then
  becomes ~8 GiB (est.). This is tiering only, so it is bit-identical. It is also the only text-speed effect, and only
  with `--vision` on.
- Cap the encode peak in our process: 4-D SDPA or query chunks (< ~1 GiB at max, est.), `--threads 16`.
- `TMPDIR=$XOLOTL_CACHE/vision` for the server, so that the `.sve` cache lives on disk and not in tmpfs RAM.
- Encoder env: `HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1` (plus the existing `CUDA_VISIBLE_DEVICES=-1`), so
  the encoder never opens a KFD context. `vision_env()` otherwise hands it the engine's env (`server.py:2934-2946`).
- Lazy alternative (`"lazy": true`, `server.py:2320, 7180`): the arena stays 79.6. The encoder (~1.5-1.8 GiB plus
  peak) then appears on top of the ~12 GiB idle headroom, on a first-image start of torch import + 0.9 GiB read (~3-6
  s, est.). It would need reserve 32 or floor 13-14. Its `make_room` sends `VRAM` to an engine that answers
  `ERR unsupported command`: caught, logged, harmless (`server.py:3614-3633, 1162-1181`).
- Option D for comparison: weights ~0.9 GiB F16 + compute buffer + its own HIP context, ~2-3 GiB of **GTT**
  (est.). That shrinks driver free, so `pb_cap` and the extension's driver margin see it: reserve 30 -> 33 (est.).

## 3. First launch option: path B

### 3.1 Why B now (and not D)
B *is* the reference code (lowest fidelity risk), and most of it exists (`ds41_vision.py`, 355 lines, `ENC` protocol +
SVE1). Encode at 2-4 s typical / 6-12 s max (est., vision_todo 4) is small next to the span's prefill: a batch turn
costs 24 s for 412 new tokens and 47 s for 2,451 tokens (measured, same log). So an image adds ~5-20 s (est.) on top
of the turn. D needs 2-3 d more (V4.1 layout patch, 1017 mmproj converter, stb vs PIL resize). It can replace the
config's `exe` later without touching the engine or the server.

**Runtime blocker [corr]:** the serving toolbox `xolotl-strix` has no torch and no Pillow
(`toolbox/requirements-serve.txt`), and the server spawns the encoder inside its own container.
`xolotl-quant` (15.3 GB, installed) is the same image + torch 2.11.0+rocm7.13 + Pillow + safetensors
(QUICKSTART 80). Plan: `--vision` requires running the start script inside `xolotl-quant`. It checks
`$(xolotl_python) -c 'import torch, PIL'` and dies with that hint. The engine binary from `build-strix` should run
there unchanged (same base layer, est.: verify once).

### 3.2 Weights: BF16 from the checkpoint (recommended)
Read `vision.*`/`aligner.*` from `model-00001-of-00048.safetensors` (970 MB, nothing else in it) and `image_*` from
shard 2 by header offset. That is a ~40-line safetensors reader in `ds41_vision.py`, and the reference dir defaults
to `<ckpt>/inference`. Do not regenerate a sidecar: `v41_sidecar.py` writes vision as **Q8_0** (lossy, against
quality-first), and the served MTP sidecar was made `--no-vision` on purpose. Compute dtype **bf16**, as the
reference (`generate.py:118` sets bf16, norms/RoPE in fp32 inside `vision.py`). The emitted rows are then exactly the
reference's bf16 values (as float32 in SVE1). New paths-file key: `XOLOTL_DSV41_VISION=<dir with config.json,
shards 1+2, inference/>`, default `$XOLOTL_MODELS/DeepSeek-V4.1-Flash-1017` when present. Downloaded-layout users
(HF release) have no shards 1/2: decision 5.

### 3.3 Steps (files)
1. `xolotl/tools/v41/vision/ds41_vision.py`: `--mmproj` also takes a checkpoint dir (safetensors loader), default
   dtype bf16, reference dir = `<ckpt>/inference`, 4-D/chunked SDPA wrapper, HIP/ROCR env. New
   `ds41-vision.sh` wrapper (env, `python3 -B`). Port `test_ds41_vision.py` to the BF16 checkpoint.
2. Engine (`ds41_generate.cpp`, `ds41_dense.{hpp,cpp}`, `ds41_engram.{hpp,cpp}`): `GENI` + SVE1 reader (n_embd
   5120, total n = count of 129264, rows assigned in order). `Ds41Dense::set_images(spans)` overlay + per-pass "has
   image columns". Exact `bias_vl` select (797) on uncached graphs. Engram DEAD + cumulative block + gate mask
   (563-581, 879-890) + `hist_` re-sync at resume. `hkey` beside `hist`/`park_hist`. Lookup skip. One-shot
   `--ids-file + --image-embd F` and an opt-in router/Engram-hash dump at image positions for the gates. Bias_vl is
   already on the device (the dense loader skips only routed experts and `token_embd`, `ds41_dense.cpp:272`).
3. `xolotl/serve/server.py`: `VISION_START = None` mode in the expansion loop (~4038-4052), and an image-source policy
   switch (decision 3).
4. `xolotl/tools/v41/serve/serve_ds41.py`: image items -> DeepSeek blocks only for messages that have images;
   `S.IMAGE_PAD = "<｜deepseek_image｜>"`, `S.VISION_START = None`. Tests in `test_serve_ds41.py` (MockEngine):
   text-only render unchanged with vision configured, one image -> n x 129264, the GENI line, `img_keys`.
5. `scripts/start_xolotl_dsv41 --vision` (`XOLOTL_VISION=1` flips the default; `--no-vision`). It writes
   `"vision": {"exe": ".../ds41-vision.sh", "mmproj": "$XOLOTL_DSV41_VISION", "model": "$XOLOTL_DSV41_GGUF",
   "threads": 16, "max_tokens": 1024}`. It adds `STRATA_ARENA_EXT_FLOOR_GIB=12` to the env JSON and exports
   `TMPDIR=$XOLOTL_CACHE/vision`. It does **not** change `--uma-reserve-gib` (resident). It checks torch/PIL, the
   weights, and that the engine knows GENI (e.g. `--help` mentions `--image-embd`). The help text is updated. Without
   the flag, the config and env are byte-identical to today's.
6. Docs: README/QUICKSTART line, ROADMAP entry. `dv -s` at each checkpoint.

### 3.4 Text bit-identical: guarantee + tests (future runs)
By construction: no image -> overlay empty, no vl graph, no DEAD, `hkey` all 0 (same LCP), the lookup sees no keyed
position. The engine runs the same graphs, the same host loops, the same hashes. Server: same rendered ids.

| run | what | expect | time (est.) |
|---|---|---|---|
| T1 | `scripts/verify_setup` with the new binary (needs the server stopped, MemAvailable >= 110) | PASS token for token vs `verify_setup.ref` | 10-15 min |
| T2 | serve logit dumps (`STRATA_SERVE_DUMP_TAIL`) + token md5: old vs new binary, `--vision` off, 3 prompts (chat, agent 8K, code), `--no-dspark` and DSpark (temp 0, fixed seed) | byte-identical | 45-60 min |
| T3 | same as T2 with `--vision` on, text-only requests (arena ~1.7 smaller, extension ~8 GiB) | byte-identical (tiering only) | 45 min |
| T4 | `test_serve_ds41.py` render golden, vision configured vs not (no GPU) | identical ids | 5 min |
| T5 | one timing sanity, `--vision` off, same turn as a recent log | within noise (code path unchanged) | 15 min |

## 4. Image validation (future runs, not run)

| run | method | gate | time (est.) |
|---|---|---|---|
| V1 encoder | `test_ds41_vision.py` on 1017: `ds41_vision` vs the reference `ViT`/`Aligner` on the same BF16 tensors; 3 synthetic sizes + 4 real (photo, screenshot, chart, max 4000x3000); also encode twice -> identical bytes; record RSS peak at max | per-row cosine >= 0.999 over the whole span incl. delimiters; peak < 3 GiB | 30-45 min CPU (server stopped or idle) |
| V2 engine wiring | one-shot `--image-embd`: router top-6 at image positions, layers 0-3, vs torch `Gate` with `bias_vl`; Engram hashes at the 3 text positions after a span vs `engram.py`; gate 0 at image positions | >= 99% identical top-6 sets; hashes equal | 2-3 h incl. setup |
| V3 reuse | 2-turn same image (reuses past the span), same-size different image (LCP stops at the span start), cancel mid-answer + continue (DSpark resume), side request + unpark | log `reuse lcp=...` as expected; no stale span | 1 h |
| V4 teacher | layer-streamed BF16/FP8 reference (`calib_layers.sh` path) + image embeddings + `bias_vl` + Engram mask; ~32 image prompts (caption, OCR, chart, counting, UI); engine top-1 / KL on forced answers | within ~1 pt of the GGUF's text top-1 agreement | ~11 min/layer -> ~7-8 h per set + 1 h engine |
| V5 spot checks | captions/VQA by eye; grounding `<｜box｜>` prompt; DSpark tokens/pass on image answers | sane; acceptance recorded | 30 min |
| V6 expert tiers | log routed experts at image tokens vs `alloc_1017.json` tiers | decide requant or not | 1 h + analysis |

## 5. Risks and open decisions

Open decisions:
1. **Encoder runtime:** serve from `xolotl-quant` under `--vision` (no download, recommended), a separate CPU-torch
   venv on local disk (download), or go straight to D (no torch, GPU, +2-3 d)?
2. **Resident or lazy encoder:** resident = arena ~1.7 GiB smaller the whole time `--vision` is on (est.). Lazy =
   arena unchanged, ~3-6 s first-image start, tighter idle headroom (reserve 32 or floor 13-14). Either way:
   extension floor 12?
3. **Image sources under `--no-auth` on 0.0.0.0** (the default): `Vision.load` reads any local file path and fetches
   http(s) URLs (`server.py:2399-2419, 2420-2435`). `_no_local_images` blocks only foreign *browser* origins
   (5935-5945). So any LAN client could have image files on this box described back to it, or make the server fetch
   LAN URLs. Proposal: `--vision` accepts only `data:` (+ http(s) behind a flag), with local paths off unless asked.
4. Encoder dtype: bf16 (= reference) or f32 (2x slower, est.)?
5. Weights for HF-release users (no checkpoint): publish a BF16 vision safetensors (~0.97 GB) in the release repo,
   or document "download shards 1+2"?
6. Image token cap 1024 (up to ~1,000 prefill tokens per image) or lower; max images per request; whether old
   images keep their full span in long conversations.
7. If V6 shows VL experts in low tiers: requantize with image tokens in the corpus, or accept?

Risks:
- torch CPU SDPA math fallback at 8.7K patches (2.5-5 GB per layer) -> swap/KFD stalls: must be capped (step 1).
- ROCm 7.14 `LD_LIBRARY_PATH` from `xolotl_rocm_env` leaking into torch's ROCm 7.13 libs at import (est. fine,
  since the quant toolbox runs torch with that layer; check once).
- Engram `hist_` not snapshotted: there is already an edge case for text when a side request writes up to the main
  conversation's tail. The re-sync in section 1, row 4 (d) fixes it for images. For text it is value-identical except in that edge
  case.
- DSpark acceptance on image answers unknown (lower acceptance = speed only).
- The FIFO releases between encode and GENI (another request may run in between): harmless, as today for Qwen.
- Not verified: torch CPU SDPA kernel choice for 3-D inputs; encoder RSS; encode times; the arena/extension
  figures under `--vision` (all est. above).

Sources read: `vision_todo.md`; `scripts/start_xolotl_dsv41`; `xolotl/tools/v41/serve/{ds41-engine-strix.sh,
serve_ds41.py}`; `xolotl/serve/{server.py,frontend.py}`; `xolotl/tools/v41/{dsv41_encoding.py,v41_sidecar.py,
vision/ds41_vision.py}`; `xolotl/tools/v41/engine/{ds41_generate.cpp,ds41_dense.cpp,ds41_engram.{hpp,cpp},
ds41_dspark.{hpp,cpp},ds41_moe.cpp}`; checkpoint `inference/{model.py,engram.py,vision.py,image_processor.py,
generate.py}`, `config.json`; shard 1/2 headers; GGUF headers (served + MTP); `~/.local/state/xolotl/{strata-ds41-strix.json,
logs/ds41-engine.log}`; `toolbox/requirements-*.txt`, QUICKSTART; docs/UMA_DEEP_DIVE.md 2b, docs/CANCEL_REUSE.md.
