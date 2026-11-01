# Strixolotl

**Strixolotl (Xolotl on Strix Halo)** runs DeepSeek-V4.1-Flash-1017 on one AMD Strix Halo box (Ryzen AI Max,
Radeon 8060S, gfx1151) with 128 GB of unified memory. It is a port of the [Xolotl](https://github.com/Indras-Mirror/Xolotl)
inference engine from CUDA to HIP/ROCm, plus our own quantization of the model (GSQ-217) and the scripts to build,
download, verify and serve it.

The goal is quality first: no extra quantization at run time, exact expert routing, and numerics kept close to
DeepSeek's reference code. Decode runs at **about 11 tokens per second with DSpark** (the default: the model's own
MTP heads draft tokens that one batched pass verifies) and **about 9 without** (`--no-dspark`), on a Samsung 990 Pro
NVMe. The server starts with the full 1,048,576-token context.

- The model's experts (243 GB) do not fit in 128 GB. About 86-96 GiB of them stay in RAM; the rest are read from the
  NVMe drive when the router picks them. That makes decode drive-bound.
- One request at a time. OpenAI-compatible (`/v1`) and Anthropic-compatible (`/v1/messages`) server, tool calls,
  prompt-cache reuse across agent turns.

## Measured results

All numbers come from one box: Minisforum MS-S1 MAX, Ryzen AI MAX+ 395, 128 GB, Fedora 43, ROCm 7.14.1, model on a
Samsung 990 Pro 1 TB NVMe (PCIe 4.0 x4, ~6 GB/s on expert reads). Details, dates and the roadmap: [ROADMAP.md](ROADMAP.md).

**Quality** (same-text engine A/B, 17 held-out 8K-token rows, teacher-forced against the unquantized FP8 checkpoint,
2026-10-23):

| | ppl gap to the teacher | top-1 agreement with the teacher |
|---|---|---|
| GSQ-200 (first quant, no fidelity switches) | +2.74% | 0.854 |
| **GSQ-217, quality mode (the release)** | **+1.61%** | **0.874** (higher on 16 of 17 rows) |

For scale: two sampled runs of the teacher itself (temperature 1, top_p 0.95) pick the same token at 0.71 of the
positions. Top-1 agreement over many sequences is the signal we trust; single-text perplexity on this model moves by
several percent from small numeric changes.

**Speed** (quality mode, Samsung 990 Pro; decode 2026-10-27/28, prefill 2026-10-29/30, frozen):

| | DSpark (default) | plain decode (`--no-dspark`) |
|---|---|---|
| **Decode** | **~11 tok/s**: 11.0-11.6 tok/s on 700-token answers; 10.5 tok/s over an 8-turn agent session (8-13 per turn, 11.3 on a 1,525-token answer) | **~9 tok/s**: 8.8-9.2 tok/s on the same answers; 8.3 tok/s over 100 live agent requests at 36-190K context |
| Prefill of new tokens | about the same in both. TTFT of an agent turn (default: fused key gather + predicted reads): +0.8K ~19-24 s, +2.5K ~44 s, +6.5K ~81-84 s; cold 5.6K ~73 s, cold 15K ~147 s. Optional `--fast-prefill` (MMQ, not bit-identical, ppl within noise): +2.5K ~27-30 s, +6.5K ~39 s, cold 15K ~67 s | |
| Prompt reuse | every agent turn reads only its new tokens; a cancelled answer resumes where it stopped | |
| Engine load | ~30 s | |

- **Where a plain token goes** (~112 ms): dense half ~58 ms, experts in RAM ~20 ms, expert reads from the SSD
  ~32 ms (12.8 per token at ~2.5 ms each). Decode is drive-bound: see [SSD_SPEED.md](SSD_SPEED.md).
- **DSpark** drafts up to 4 tokens with the checkpoint's own MTP layers (`mtp.0-2`, a 5.9 GB sidecar) and verifies
  them in one batched pass, with exact speculative sampling: the output distribution is the model's at any
  temperature. It is not bit-identical to plain greedy decode (batched arithmetic: ~1.4% of greedy tokens differ, at
  near-ties). DSpark is the default since 2026-10-31; `--no-dspark` gives plain decode.

## Modes

**Quality mode** (the default, done). No additional quantization: the dense half stays the file's Q8_0, every expert
runs in its file format, routing is exact, and four fidelity switches bring the numerics closer to DeepSeek's
`model.py`. The file-tier read path defaults (2026-10-25) and the prefill defaults (fused key gather, predicted reads,
2026-10-29/30) are bit-identical to the reference. `scripts/start_xolotl_dsv41` starts it, with long context and
DSpark on by default.

**Long-context mode** (the default since 2026-10-31; `--no-long` = 262,144 tokens). Quality mode with the context at
the model's limit of 1,048,576. Measured 2026-10-25 on one conversation that grew in stages (filler: llama.cpp source
code), with two hidden pass keys added per stage and every turn asking for all keys so far:

| Prompt | New tokens read | Prefill | Pass keys recalled |
|---|---|---|---|
| 65K | 65K | 77 tok/s (14 min) | 2/2 |
| 130K | 65K | 68 tok/s (16 min) | 4/4 |
| 261K | 131K | 73 tok/s (30 min) | 6/6 |
| 523K | 262K | 58.5 tok/s (75 min) | 8/8 |

- Memory: the larger context costs nothing up front; free memory stayed at ~10 GiB through 523K.
- Prompt reuse: each turn read only its new tokens. Reading a long prompt from scratch is slow: prefill drops with
  depth because every token attends over more context (523K took ~2.3 h in total).
- Decode does not slow down with depth: plain decode measured 8.24 tok/s at 80-190K. Deeper, only the attention
  index scan grows with the context (estimated at a few ms per token against ~112 ms). DSpark at long depth is not
  measured yet.

**Speed mode** (paused). The lossy knobs that stay within our quality budget (dense q6_K, a small file penalty) gave
about +15%, because decode waits on the drive, not on compute. The lossless steps gave far more (a fast NVMe, then
DSpark), so there is no separate speed mode; see [SSD_SPEED.md](SSD_SPEED.md).

## Requirements

- AMD Strix Halo with 128 GB (gfx1151). Kernel cmdline `amdgpu.gttsize=131072 ttm.pages_limit=33554432` (GTT ~123
  GiB), memlock unlimited, BIOS UMA frame buffer at the minimum.
- Linux with podman and toolbox (tested: Fedora 43, kernel 7.1.10). The engine builds and runs in a Fedora 44
  toolbox with ROCm 7.14 (`toolbox/Containerfile`).
- A local NVMe with ~460 GB free for the prequantized model (GGUF 243.1 GB + Engram tables 203.1 GB), or ~1 TB to
  quantize it yourself. ext4, xfs or btrfs (`chattr +m` on the model folders). Drive speed sets decode speed.

## Getting started

[QUICKSTART.md](QUICKSTART.md) has the full recipe:

1. **Download the prequantized model** (recommended): `scripts/download_model --repo Snowsh1ne/DeepSeek-V4.1-Flash-1017-GSQ-217-Strixolotl`
   fetches GSQ-217 from [Hugging Face](https://huggingface.co/Snowsh1ne/DeepSeek-V4.1-Flash-1017-GSQ-217-Strixolotl) in verified parts, plus only the two Engram
   shards of DeepSeek's checkpoint. No quantization.
2. **Or quantize it yourself:** download the 510 GB FP8 checkpoint, calibrate (~12.5 h) and run the GSQ quant pass
   (~8.7 h). GPTQ is not bit-reproducible, so this gives an equivalent file, not the released one.

Then `scripts/verify_setup` checks the box and runs a 32-token generation that must match `scripts/verify_setup.ref`
token for token, and `scripts/start_xolotl_dsv41` serves it on port 8150 (default: the LAN without a key, long context,
DSpark; `--local` or `--api-key KEY` to restrict it, `--fast-prefill` for the faster, not bit-identical prompt path).

## Project layout

| Path | What |
|---|---|
| `xolotl/` | The port: Xolotl's tree with every change marked `strix-port:`. Engine in `xolotl/tools/v41/engine/`, quantization tools in `xolotl/tools/v41/`, server in `xolotl/tools/v41/serve/`. |
| `scripts/` | Host and toolbox scripts: `toolbox_create`, `build_xolotl`, `download_model`, `fetch_1017_fp8`, `start_xolotl_dsv41`, `verify_setup`, `hf_prepare`, `hf_upload`; `scripts/dev/` is the test harness |
| `toolbox/` | Containerfiles (engine, quantization), compose file |
| `hf/MODEL_CARD.md` | The Hugging Face model card; `HF_RELEASE.md` is the upload procedure |
| `QUICKSTART.md`, `ROADMAP.md`, `SSD_SPEED.md`, `PORT_PLAN.md` | Recipe, status / results / roadmap, drive analysis, the original port plan |
| `docs/` | Engineering write-ups: prefill (`PREFILL.md`), unified memory (`UMA_DEEP_DIVE.md`), MTP prefetch, cancel-and-resume |

## Credits

- **[Xolotl](https://github.com/Indras-Mirror/Xolotl)** by Indras Mirror: the DeepSeek-V4.1 engine, the GSQ
  quantization pipeline and the server this project ports. Xolotl is built on
  **[Strata](https://github.com/Niko1221/Strata)** by Niko1221 and [ggml / llama.cpp](https://github.com/ggml-org/llama.cpp).
- **[DeepSeek](https://huggingface.co/deepseek-ai)** for [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017).
- **[antirez/ds4](https://github.com/antirez/ds4)** for the imatrix calibration prompts used in our calibration corpus.
- Reference and inspiration:
  - [strixite](https://github.com/shawnshekari/strixite) (hand-written HIP for Strix Halo; host setup and
    memory-pressure lessons).
  - [halogen](https://github.com/peonist-ai/halogen-flash-server) (HIP inference on Strix Halo).
  - [vincalkr/DS41Engine](https://github.com/vincalkr/DS41Engine) (formerly Strata-DSV4.1: a gfx1100 port of the
    DeepSeek-V4.1 engine; HIP graph measurements).
  - [OpenBMB/llama.cpp-omni](https://github.com/OpenBMB/llama.cpp-omni) (its MoE expert-slot predictor, reviewed for
    the read path).

## Support

If Strixolotl is useful to you: [buymeacoffee.com/snowsh1ne](https://buymeacoffee.com/snowsh1ne).

## License

- The Strixolotl code (scripts, docs, the port's changes): MIT, see [LICENSE](LICENSE).
- `xolotl/`: MIT, "Copyright (c) 2026 Niko1221 and the Strata contributors" (`xolotl/LICENSE`). The port's changes are
  in that tree and keep that notice; upstream is [Indras-Mirror/Xolotl](https://github.com/Indras-Mirror/Xolotl).
- ggml / llama.cpp (cloned at build time, and `xolotl/third_party/ggml`): MIT, "Copyright (c) 2023-2026 The ggml authors".
- The model: DeepSeek-V4.1-Flash-1017 is released by DeepSeek under the MIT License, weights included
  ("Copyright (c) 2023 DeepSeek"). The GSQ-217 quant and the Engram tables it uses are derived from those weights
  and carry that license and notice.
