# Vision input for DeepSeek-V4.1-Flash-1017: plan

Status: plan only, nothing built or run. Researched 2026-10-25 on the test box while the 1M-context benchmark was running
(read-only: code, configs, safetensors headers, `podman image inspect`, `/sys`, `lsmod`, `journalctl -k`, web pages).
"(est.)" marks an estimate that has not been measured.

## 1. Summary and recommendation

- The checkpoint has a complete vision path: a 32-layer ViT (411.8M params) plus a 2-layer aligner (73.4M), 485.3M
  params in all, BF16. DeepSeek's reference code is small and clear (`inference/vision.py`, `image_processor.py`).
- **Most of the encoder side already exists in this repo.** `xolotl/tools/v41/vision/ds41_vision.py` is a torch CPU
  encoder that is a drop-in for the server's `strata-vision` process (same argv, same `ENC` protocol, same SVE1 output
  file), with a gate test (`test_ds41_vision.py`, per-row cosine >= 0.999 against the reference). It was written for
  the old V4.1 sidecar (Q8_0, on the other machine) and has never been connected to anything.
- **The missing part is the engine.** `ds41_generate.cpp` has no `GENI` command (embeddings in), no VL routing bias, no
  Engram mask for image tokens, and its prompt-reuse logic compares token ids only, so two different images would match.
  The server (`xolotl/serve/server.py`) already does the rest for Qwen: image fetching, WebP conversion, hashing, the
  encode cache, `img_keys`, and the `GENI` line. The V4.1 shim (`serve_ds41.py`) only flattens content to text.
- **The NPU is not worth doing now.** (a) The NPU is off on this box: the kernel has `amd_iommu=off`, and `amdxdna`
  refuses to probe without an IOMMU (`journalctl -k`: "aie2_init: Running without IOMMU not supported"; no
  `/dev/accel`). (b) No Linux NPU runtime takes a custom ViT on Fedora without a lot of work (section 5). (c) The
  encoder is not the bottleneck: the LLM then prefills 184-1,001 image tokens at 22-85 tok/s (6-45 s), against a
  0.3-12 s encode on any device (est., section 4).
- **Recommendation:** first get correct images end to end with the CPU encoder (option B: existing code, BF16 weights
  loaded straight from the checkpoint). Then move the encoder to the GPU through llama.cpp's `deepseek4v` mtmd graph,
  which is already in our pinned llama.cpp and needs a V4.1 layout patch (option D). Keep the NPU (option A) as a
  separate experiment, only if turning the IOMMU back on is acceptable.

## 2. Architecture facts (from the checkpoint's reference code and headers)

Sources: `~/models/xolotl/DeepSeek-V4.1-Flash-1017/inference/{vision.py,image_processor.py,model.py,engram.py,
generate.py,config.json}`, `config.json` `vision_config`, the shard 1/2/3 safetensors headers, and `tokenizer.json`.

**Preprocessing** (`image_processor.py`): RGB; normalization `(x/255 - 0.5)/0.5`; patch 14.
- Upscale if `w*h < min_pixels = 295,936` (= 544x544; the earlier note said "29", which was wrong). Then round each
  side up to a multiple of 14 and `ImageOps.pad` (aspect kept, gray 127 padding; PIL bicubic when it resizes).
- Token budget `vision_max_n_token = 1024` for the whole span. If it is over budget, `solve_resize_ratio` shrinks the
  image keeping its aspect. `max_wh_ratio = null`. No tiling or crops: one native-resolution grid per image.
- LLM grid: `n_llm = ceil(n_vit / 3)` per side. Span = `[START] + ([IMAGE]*w + [NEWLINE])*h + [END]`, so
  `n_tokens = h*(w+1) + 2`.

**ViT** (`vision.py`, "full bidirectional attention over one image with 2D RoPE"):
- patch_embed: Linear 588 -> 1024 with bias (no conv, no CLS, no learned position embedding).
- 32 pre-norm blocks: RMSNorm (eps 1e-6) -> attention -> RMSNorm -> MLP.
  - Attention: fused wqkv 1024 -> 3072 (with bias), 16 heads x 64, full attention over all patches (no window), wo with bias.
  - MLP: SwiGLU, w1 1024 -> 2x2816 (no bias), w2 2816 -> 1024 (no bias).
- Final RMSNorm.
- 2D RoPE: rope_dim 32 per head, theta 10000. 16 frequencies on the row index and 16 on the column index, applied in
  NeoX style (rotate halves) over the full 64-dim head. Computed in fp32.

**Aligner:** the patch grid is zero-padded to a multiple of 3, then `F.unfold` 3x3 (9216 = 9x1024), Linear 9216 -> 5120,
erf-GELU, Linear 5120 -> 5120 (both with bias).

**Splicing** (`model.py` `merge_image_embeddings`):
- Every position of the span carries id 129264 (`<｜deepseek_image｜>`; checked in `tokenizer.json`).
- The embeddings at those positions are overwritten. IMAGE rows take the aligner rows in reading order. START, END and
  NEWLINE take the learned `image_start` / `image_end` / `image_newline` (BF16 [5120], **in shard 2**, not shard 1).
- The text model uses plain 1D positions (no M-RoPE).
- The whole span (delimiters too) counts as "image" for two more things:
  - **MoE routing** uses `ffn.gate.bias_vl` instead of `ffn.gate.bias` for selection (all 40 layers + the 3 MTP layers).
    It is already in our GGUF as `exp_probs_b_vl.bias` (`v41_gguf.py:253`) but unused. It matters: on layers 0/10/20/39
    the per-expert pattern of `bias_vl` correlates -0.30 / -0.15 / 0.00 / -0.36 with the text bias (read from the
    shard headers' 1.5 KB tensors). Image tokens pick different experts.
  - **Engram:** image positions are stored as DEAD in the n-gram history, so no n-gram spans an image (text right after
    an image hashes with pad), and the Engram gate is 0 at image positions.
- The DSpark/MTP `forward_spec` in the reference ignores images (it embeds the raw ids and uses no VL mask).

**Parameters** (shard 1 header, all BF16): vision 411,842,560 (259 tensors) + aligner 73,410,560 (4 tensors) +
3 x 5,120 delimiters = **485.3M**, 0.97 GB in BF16.

**Image cost** (my script; weights FLOP = 2·411.8M·N, attention = 4·N²·1024·32, aligner = 2·73.4M·cells):

| image (px) | ViT grid = patches | LLM grid | span tokens | TFLOP (weights + attn + aligner) |
|---|---|---|---|---|
| any image <= 544x544 (upscaled) | 39x39 = 1,521 | 13x13 | 184 | 1.25 + 0.30 + 0.02 = 1.6 |
| 800x600 | 43x58 = 2,494 | 15x20 | 317 | 2.9 |
| 1024x768 | 55x74 = 4,070 | 19x25 | 496 | 3.35 + 2.17 + 0.07 = 5.6 |
| 1280x800 | 58x92 = 5,336 | 20x31 | 642 | 8.2 |
| 1920x1080 | 69x122 = 8,418 | 23x41 | 968 | 16.4 |
| square >= 1302 px | 93x93 = 8,649 | 31x31 | 994 | 7.12 + 9.80 + 0.14 = 17.1 |
| 4000x3000 | 81x108 = 8,748 | 27x36 | 1,001 | 17.4 |

This corrects the earlier estimate (~9,200 patches, ~18 TFLOP): real images top out at about 8,750 patches and
~17.4 TFLOP. At the top end attention is ~57% of the work, because attention cost grows with the square of the
patch count.

## 3. What exists already (repo)

- `xolotl/tools/v41/vision/ds41_vision.py` + `test_ds41_vision.py`: the encoder (reference ViT/Aligner classes,
  DeepSeek's `load_image`) and its gate. It is argv/protocol-compatible with `xolotl/tools/vision/strata_vision.cpp`.
  It writes the SVE1 span with the delimiter rows included, as flat reading order (`nx = n, ny = 1`).
- `xolotl/tools/v41/v41_sidecar.py`: writes `vision.* / aligner.* / image_*` (Q8_0 + F32) and the MTP heads into
  `V41-MTP-Vision*.gguf`. The 1017 quant plan (`xolotl/docs/ds4/V41_1017_QUANT_PLAN.md` step 8) says to regenerate it
  for 1017 and not to reuse V4.1's.
- `xolotl/serve/server.py`: class `Vision` (~l.2275, resident encoder, cache keyed by the sha256 of the normalized
  bytes), `prepare()` (~l.3996-4090: fetch, encode inside the request FIFO, expand the placeholder to n tokens, `img_keys`),
  `GENI <max_new> <keys> <embeddings file> <ids>` (~l.1582), and `input_modalities` that turns to text+image when vision
  is configured (~l.5565).
- Upstream notes (`xolotl/RESUME_PROMPT.md` ~l.174, `docs/ds4/V41_RESUME_PROMPT.md` ~l.813/1031): "Vision: exported
  only; no engine code". The plan there: "Q8_0, lives in RAM/NVMe, loaded on demand".
- llama.cpp at our pinned commit `3cf03257` (2026-09-20, local checkout `~/.cache/xolotl/llama.cpp`) has
  `tools/mtmd/models/deepseek4v.cpp` (PR #28133, for **DeepSeek-V4-Flash-Vision-Exp**). It has the same ViT (RMSNorm,
  SwiGLU, 2D vision RoPE, theta 10000) and the same 3x3 unfold + GELU aligner. It differs from V4.1 in the token block:
  - It hard-codes `max_n_token = 384` and `max_wh_ratio = 8` (`clip.cpp` ~l.1599; the converter asserts both).
  - Its layout adds PAD rows and a lead pad to a 4-token boundary, and interleaves row pairs ("N-layout").
  V4.1's layout has none of these.

## 4. Options

Latency and contention assumptions (all est.):
- Typical = 1024x768 (5.6 TFLOP, 496 tokens). Max = 17.1-17.4 TFLOP, ~1,000 tokens.
- LLM prefill of the span at the measured text range of 22-85 tok/s (ROADMAP "Long-context mode" and "Prefill"): 6-23 s
  typical, 12-45 s max, for **every** option.
- The server encodes only while the engine is idle, in the same FIFO (`server.py` ~l.4024), so no option runs at the
  same time as the LLM unless that rule is changed.

| option | effort (est.) | encode typical / max (est.) | memory | fidelity risk | interaction with the LLM |
|---|---|---|---|---|---|
| **B. CPU, torch** (`ds41_vision.py` in the xolotl-quant toolbox, BF16 weights from shards 1+2) | small: 0.5-1 d (load safetensors instead of the sidecar, 1017 paths, BF16 mode) | 2-4 s / 6-12 s at 1.5-3 TFLOPS effective on 16 Zen 5 cores with AVX-512 BF16; FP32 roughly 2x slower | ~1 GB weights + activations. Check torch's CPU SDPA: a dense score matrix would be 16·8649²·2 B = 2.4 GB per layer | lowest: it *is* the reference code | none while serialized. Process RSS counts against MemAvailable (11 GiB free during the 1M run) |
| **D. llama.cpp mtmd `deepseek4v`, patched for V4.1** (`strata_vision` on ggml-hip or ggml-cpu) | medium: 2-3 d (V4.1 mode: limits 1024/none, min_pixels from GGUF, plain layout, converter for the 1017 tensors) | GPU 0.2-0.4 s / 0.6-1.2 s at 15-30 TFLOPS effective of ~59 peak (est.); CPU about like B | BF16/F16 mmproj ~1 GB (Q8_0 ~0.5 GB) + compute buffer (needs flash attention at 8.7K patches) | low-medium: PR reported NMSE 1e-4 on IMAGE rows for V4-Exp. Resize is stb bicubic, not PIL: gate on cosine | GPU memory comes out of the same UMA RAM. Under the FIFO rule no compute contention. Count ~2 GB in `--uma-reserve-gib` |
| **C. own ggml-hip graph inside `ds41_generate`** | large: 4-6 d (port ViT + aligner + 2D RoPE as a sidecar graph, buffers, scheduler) | like D on GPU | like D, but inside the engine's budget | medium (new code) | shares the engine's GPU context. Encodes would block the decode loop unless scheduled between requests. No gain over D |
| **A. NPU (XDNA 2)** | very large, uncertain: 2-4+ weeks, after a host change and a reboot | 0.8-1.4 s / 2.5-4.5 s at 4-7 TFLOPS effective. Best measured BF16 GEMM on XDNA2 is 14.7 TFLOPS (bf16 emulated on bfp16, arXiv 2512.13282), plus attention/softmax/RoPE/norm overheads | weights in host memory (~1 GB), pinned | medium-high: AMD notes BF16/BFP16 accuracy drops in some transformers; INT8 is worse for vision | could run beside the LLM. Halogen's docs say GPU+NPU work at the same time can hang the machine unless the GPU fabric clock is held at max |

The earlier NPU figure (~1-2 s per max image) was too optimistic. ~2.5-4.5 s (est.) is more realistic, and an
idle GPU with option D is faster.

## 5. NPU and halogen findings

**This box:**
- PCI `f5:00.1` [1022:17f0] rev 11 (Strix Halo NPU), `amdxdna` loaded (kernel 7.1.10), but the probe failed with -22
  ("Running without IOMMU not supported"): no `/dev/accel`, no `/sys/class/accel`.
- Kernel cmdline: `amd_iommu=off amdgpu.gttsize=131072 ttm.pages_limit=33554432`. ROADMAP "Host setup", `PORT_PLAN.md`
  and `QUICKSTART.md` already note that `amd_iommu=off` (+1-12% GPU) turns the NPU off.
- No XRT, xrt-npu or Ryzen AI packages are installed (`rpm -qa`).

**Halogen's GPU build is GPU-only:** its images (`ghcr.io/peonist-ai/halogen-flash-server`) carry only ROCm 7.14.0
wheels and `flash_serve`, no XRT layer or NPU variables. Its vision tower (0.84 GiB of weights, 1.97 GiB with
worst-case scratch) runs on the GPU and is charged against the KV pool.

**Halogen upstream** (`docs/NPU.md`, `docs/npu-small-models.md`, `docker-compose.npu.yml`, example tag 0.18.15):
- What runs on the NPU: small models next to the Flash model. `decider-0.8b/4b` (classifier/decisions), `d3-lite/nano`
  (decisions with a picture, 0.2-0.9 s), qwen3 embedding/reranker/guard 0.6B, `qwen3.5-2b` (summaries), flux2-klein-4b,
  ASR and TTS.
- How: its own closed engine (LicenseRef-Peonist-EULA) on the host's XRT + NPU plugin, through `/dev/accel/accel0`.
  Fine-tunes of the first four families are converted to `halogen-npu.hnpw` on first start. Inputs are padded to buckets.
- Host requirements: IOMMU on (`iommu=pt` is fine), memlock unlimited, and the GPU fabric clock held at max (a systemd
  unit), "GPU work and NPU work at the same time can otherwise hang a Strix Halo machine".
- **We cannot reuse it for our ViT:** it is closed and limited to its own model families.

**AMD's Linux NPU stack:**
- Ryzen AI 1.7.1/1.8.0 on Linux: Ubuntu 24.04, STX/KRK. Models: CNN INT8/BF16, NLP encoders (BERT) in BF16, NPU-only
  LLMs, compiled through ONNX Runtime + the VitisAI EP. Fedora has no packages: XRT and the plugin must be built from
  source (dev.to Fedora 44 Strix Halo guide). A ViT with custom 2D RoPE is not a listed model class.
- FastFlowLM / Lemonade support Linux on XDNA2, but only their own model zoo.
- The open route for a custom model is IRON/mlir-aie (BF16 GEMM ~14.7 TFLOPS measured), which means hand-written kernels.
- Static shapes: the V4.1 grid changes with every aspect ratio. On the NPU it would need patch-count buckets, with
  padded keys masked out of the attention (exactness), and the aligner unfold on the CPU.

## 6. Integration design

**Engine (`xolotl/tools/v41/engine/`):**
1. `ds41_generate.cpp` serve loop (~l.1275): accept `GENI <max_new> [keys] [img_keys=k1,k2] <SVE1 file> <ids>`.
   - Read the SVE1 records (int32 header + float32 rows of 5120).
   - Map them in order onto the runs of id 129264 in the prompt. Each run's length must equal the record's `n`,
     otherwise send `ERR`.
   - Build a per-position `vl` flag.
   - Also accept `--ids-file` + `--image-embd` for one-shot runs and tests.
2. `Ds41Dense::upload_inputs` (`ds41_dense.cpp` ~l.1279): overwrite `host_emb` rows at image positions. This happens on
   the host, before `ggml_backend_tensor_set`, so it is a small change. Check the column offsets for layer-major
   batches, `--prefill-chunk` / `--prompt-batch` splits and `--enc-skip`: a span may cross a chunk boundary (the
   reference's "single chunk" rule only comes from how it is implemented).
3. Router (`ds41_dense.cpp` l.762, 922, 956, 965: every place `exp_probs_b.bias` is added, including the next-layer
   prediction paths): `s0 = sc + bias + vl * (bias_vl - bias)` with an input tensor `i_vl [1, n]`. Weights stay the
   unbiased scores. Decode tokens are always text (`vl = 0`).
4. Engram (`ds41_engram.cpp/.hpp`): `record()` stores DEAD for image positions (so the hashes block there, as
   `engram.py` l.160-184), and the gate is masked to 0 at those positions in the graph.
5. Prompt reuse: `hist` LCP (~l.1300), snapshots, regenerate and session save must compare `(id, image key)` and not
   the id alone. For example, keep a parallel key array: position -> 64-bit hash of the image key + offset in the span.
   Otherwise two different pictures with equal spans match.
6. DSpark: follow the reference (no image path; it embeds the raw id). (DSpark is the serving default since 2026-10-31.)
7. Text-only requests must stay bit-identical. With `vl` all zero the graph must not change: gate it on "the batch has
   image rows" if a no-op add changes rounding.

**Server (`xolotl/tools/v41/serve/serve_ds41.py` on `xolotl/serve/server.py`):**
- `to_encoder_messages` (l.117) currently flattens content with `_text()`. Instead:
  - Emit `<｜deepseek_image｜>` per image part in order. `encoding.py`'s `_process_image_blocks` / `_extract_image`
    already handle OpenAI `image_url` + Anthropic `image` blocks, and `_validate_no_image_sp_tokens` rejects the
    placeholder typed as text.
  - Then reuse `prepare()` with a DeepSeek expansion: id 129264, no `<|vision_start|>` before it. This needs a small
    hook in `server.py` ~l.4014-4050.
- Config `"vision": {"exe": "python3 .../ds41_vision.py" or strata-vision, "mmproj": <1017 weights>, "model": ...}`.
  `input_modalities` then reports `["text","image"]`.
- Prompt reuse with images: already keyed by the sha256 of the normalized bytes (`img_keys`, server l.4084). The
  engine needs item 5 above.
- Reading DeepSeek's grid in SVE1: ds41_vision writes `nx = n, ny = 1`. The engine needs no 2D positions (plain 1D
  RoPE), so it only checks `n`.

## 7. Validation

1. **Encoder:** port `test_ds41_vision.py` to 1017. Reference = DeepSeek's `vision.py` + `Aligner` with the BF16
   checkpoint tensors. Gate: per-row cosine >= 0.999 over the full span (delimiters included), for the 3 synthetic
   sizes + 4 real photos/screenshots incl. max size. Use the same gate for the mtmd port (option D) and for any NPU build.
2. **Engine wiring (no teacher):**
   - With a fixed image prompt, dump the router ids at image positions on layers 0-3 and compare with the reference
     `Gate` (torch, the same dense tensors): >= 99% identical top-6 sets.
   - Check Engram hashes at text positions right after a span against `engram.py`.
   - Use `ds41_golden.py` taps where possible.
3. **End-to-end teacher check (the project's method):**
   - Teacher = the layer-streamed BF16/FP8 reference forward, as the calibration teacher pass (`calib_layers.sh`,
     ~11 min/layer, est. hours per prompt set), extended with image embeddings + `bias_vl` + the Engram mask.
   - On ~32 image prompts (caption, OCR of a screenshot, chart/table reading, counting, UI question), force a
     reference answer and measure the engine's **top-1 agreement** and KL on the answer tokens.
   - Gate: within ~1 pt of the text top-1 agreement of the same GGUF.
4. **Spot checks:** captions and VQA by eye. Grounding tokens exist (`<｜box｜>`, `<｜point｜>`, `<｜ref｜>`,
   `<｜polygon｜>`, tokenizer ids 129271-129278): try a "box the X" prompt.
5. **Regression:** text-only md5 of the generated tokens and the API golden tests unchanged. Then a 2-turn conversation
   with the same image (it must reuse) and a different image (it must not reuse).

## 8. Phased TODO

| phase | work | effort (est.) |
|---|---|---|
| 0 | Encoder for 1017: `ds41_vision.py` loads `vision.*/aligner.*/image_*` from shards 1+2 (BF16, no quant), `--dtype bf16`, 1017 reference path; run its gate on CPU (when the box is free) | 0.5-1 d |
| 1 | Engine: `GENI` + SVE1 reader, embedding override, `bias_vl` selection, Engram DEAD/gate mask, image-keyed reuse/snapshots, one-shot `--image-embd` | 3-5 d |
| 2 | Server shim: placeholders from image blocks, DeepSeek expansion hook, vision config, modalities; tests in `test_serve_ds41.py` with MockEngine | 1-2 d |
| 3 | Validation 7.1-7.2 + 7.5; then 7.3 teacher run (mostly machine time) | 2-4 d + teacher hours |
| 4 | GPU encoder: mtmd `deepseek4v` V4.1 mode + 1017 mmproj converter; gate 7.1; measure s/image and memory; switch the config | 2-3 d |
| 5 | Expert check: VL-heavy experts may sit in low tiers, because the 1017 tier map was calibrated on text. Log routed experts for image tokens; compare with `alloc_1017.json`; protect or requant if needed | 1-2 d + requant time |
| 6 (optional) | NPU experiment: host IOMMU on (`iommu=pt`), XRT from source, measure GPU loss, prototype one ViT block (IRON or Ryzen AI ONNX BF16), go/no-go on measured s/block | 2-4+ weeks |

## 9. Open questions

1. Order: CPU first, then GPU via mtmd (recommended), or straight to GPU? Is the NPU still wanted, knowing it needs
   `amd_iommu=off` removed (and a measured GPU loss of up to ~1-12%) and a hand-ported ViT?
2. Image token cap: DeepSeek's 1024 (up to ~45 s prefill per image at the slow end), or a lower default (e.g. 576) as
   a speed knob? Is the lower cap allowed under `--speed` only?
3. Encoder weights: BF16 straight from the checkpoint (0.97 GB; recommended) or a Q8_0 sidecar (~0.5 GB)?
4. If phase 5 shows VL experts in low tiers: requantize (calibrate with image tokens) or accept?
5. Allow http(s) image URLs (the server fetches them) or only inline/base64?
6. Limits: max images per request, and whether old images in a long conversation keep their full spans.

## 10. Sources

**Read locally:**
- Checkpoint: `inference/{vision.py,image_processor.py,model.py,engram.py,generate.py,config.json,README.md}`,
  `encoding/encoding.py`, `config.json`, `model.safetensors.index.json`, `tokenizer.json`, the shard 1/2/3 headers.
- Engine: `xolotl/tools/v41/engine/{ds41_generate.cpp,ds41_dense.cpp,ds41_dense.hpp,ds41_engram.hpp,ds41_dspark.hpp}`,
  `xolotl/tools/v41/v41_gguf.py`.
- Encoder and sidecar: `xolotl/tools/v41/{v41_sidecar.py,vision/ds41_vision.py,vision/test_ds41_vision.py,dsv41_encoding.py}`,
  `xolotl/tools/vision/{strata_vision.cpp,CMakeLists.txt}`.
- Server: `xolotl/tools/v41/serve/serve_ds41.py`, `xolotl/serve/{server.py,images.py}`.
- Docs: `xolotl/docs/{IMAGES.md,DEEPSEEK_V41.md}`, `xolotl/docs/ds4/{V41_RESUME_PROMPT.md,V41_1017_QUANT_PLAN.md}`,
  `xolotl/RESUME_PROMPT.md`, `ROADMAP.md`, `PORT_PLAN.md`, `QUICKSTART.md`.
- llama.cpp: `~/.cache/xolotl/llama.cpp/tools/mtmd/{models/deepseek4v.cpp,clip.cpp}`.
- Halogen: `~/models/Qwen3.8-NEXT_HALOGEN/README.md`, `~/logs/halogen-sidecar-test-20260919-211153.log`, and
  `podman image inspect` / `podman image history` of `ghcr.io/peonist-ai/halogen-flash-server:0.12.1`.
- Host: `lspci`, `lsmod`, `journalctl -k`, `/proc/cmdline`, `rpm -qa`.

**Web:**
- Halogen: https://github.com/peonist-ai/halogen-flash-server (README, `docs/NPU.md`, `docs/npu-small-models.md`,
  `docker-compose.npu.yml`).
- AMD Ryzen AI: https://ryzenai.docs.amd.com/en/latest/linux.html ;
  https://onnxruntime.ai/docs/execution-providers/Vitis-AI-ExecutionProvider.html ;
  https://ryzenai.docs.amd.com/en/1.4/relnotes.html (BF16 accuracy note).
- Fedora + Strix Halo NPU guide: https://dev.to/ankk98/running-llms-on-amd-npu-with-fastflowlm-fedora-guide-1oo5 ;
  FastFlowLM/Lemonade Linux: https://lemonade-server.ai/flm_npu_linux.html.
- amdxdna IOMMU patch: https://lkml.rescloud.iu.edu/2605.2/10109.html.
- XDNA2 GEMM: https://arxiv.org/html/2512.13282v1.
- llama.cpp: https://github.com/ggml-org/llama.cpp/pull/28133 ;
  https://newreleases.io/project/github/ggml-org/llama.cpp/release/v0.4.0.

**Not verified:**
- Whether 1017's vision weights differ from V4.1's.
- The CPU/GPU/NPU throughput figures (all est.).
- Whether torch's CPU SDPA avoids the N² score matrix at 8.7K patches.
- Whether `iommu=pt` keeps the GPU speed of `amd_iommu=off` on this box.
