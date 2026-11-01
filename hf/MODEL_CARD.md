---
license: mit
license_link: LICENSE
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
base_model_relation: quantized
pipeline_tag: text-generation
tags:
- deepseek
- deepseek-v4.1
- moe
- quantized
- gptq
- mixed-precision
- 3-bit
- strix-halo
- gfx1151
- rocm
- xolotl
- strixolotl
---

# DeepSeek-V4.1-Flash-1017 GSQ-217 for Strixolotl

A quantization of [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) for
**Strixolotl** (Xolotl on Strix Halo): a HIP/ROCm port of the [Xolotl](https://github.com/Indras-Mirror/Xolotl)
engine that runs the whole model on one AMD Ryzen AI Max (Strix Halo, gfx1151) box with 128 GB of unified memory.

Strixolotl repo: <https://github.com/snowsh1ne/strixolotl>.

This file is **not** a llama.cpp, Ollama or LM Studio GGUF. Its experts use per-expert quantization types those
runtimes do not load. Use it with Strixolotl.

## What is in this repo

| File | Size | What |
|---|---|---|
| `V41-1017-GSQ-217.gguf.part-01-of-23` ... `part-23-of-23` | 243,075,678,208 bytes together | The model, split into 10 GiB byte ranges. `cat` in order gives the GGUF. These are not llama.cpp `gguf-split` shards. |
| `V41-1017-GSQ-217.gguf.experts.json` | 1.2 MB | Per-expert quant types and offsets (used by the verifier) |
| `rank-217.bin` | 120 KiB | Expert ranking that seeds the in-memory expert arena |
| `tokenizer/` | 4.6 MB | BPE vocab and merges exported from the GGUF |
| `dspark/V41-1017-MTP.IQ3_XXS.gguf` | 5.9 GB | The checkpoint's MTP draft stack (`mtp.0-2`, experts IQ3_XXS, rest Q8_0) for `--dspark` speculative decoding |
| `engram/` | small | Index and notes for the Engram tables (see below) |
| `manifest.json`, `SHA256SUMS` | small | sha256 of every part, of the joined GGUF and of every small file |
| `LICENSE` | | DeepSeek's MIT license, copied from the base model |

**Engram tables are not in this repo.** DeepSeek-V4.1-Flash reads two large Engram n-gram tables at run time
(`layers.1` and `layers.14`, 202.8 GB of FP8 rows). The quant does not change them. They are downloaded from
DeepSeek's own repo, only the two shards that hold them (`model-00047-of-00048.safetensors` and
`model-00048-of-00048.safetensors`, 203.1 GB, revision `1bae0e1088993b9b82de08f59de5be5a34e42170`), and checked
against the sha256 values in `engram/README.md`. The other 46 shards of the 510 GB checkpoint are not needed.

## Quantization

| | |
|---|---|
| Method | GSQ: Xolotl's GPTQ-style pipeline (`tools/v41/`). Every routed expert gets its own format from calibration statistics. Tiers: PTQ1_0 (ternary, 1.75 bpw), Q2_0 (2.25 bpw), Q3_K. |
| This file | 217 GiB of routed experts, 99.4% Q3_K. 217 GiB is the ceiling of these tiers. |
| Dense half | Q8_0 (attention, router, shared expert; quality mode keeps it at Q8_0) |
| Calibration | 48 sequences x 8,192 tokens: agentic_long 18, agent 4, code 8, reason 6, web 2, zh 3, chat 4, multi 3. Sources: real-world agentic coding workflows (multi-turn tool use), antirez/ds4's imatrix prompts, Chinese Wikipedia, pile-10k. Calibration run `calib-1017-v2`, quant run `quant-1017-217`. |
| Teacher | The FP8/FP4 checkpoint `deepseek-ai/DeepSeek-V4.1-Flash-1017`, run layer by layer in torch |

GPTQ here is not bit-reproducible: a re-run of the same recipe gives a different, equivalent file. The checksums
in this repo identify this exact file.

## Measured quality

Same-text engine A/B on 17 held-out 8K-token rows (not used for calibration), teacher-forced, against the teacher's
own next-token distribution. Engine in quality mode, fixed 82 GiB arena, 2026-10-23.

| | ppl gap to the teacher | top-1 agreement with the teacher |
|---|---|---|
| Earlier quant GSQ-200, without the fidelity switches | +2.74% | 0.854 |
| **This quant, GSQ-217, quality mode** | **+1.61%** | **0.874** (higher on 16 of 17 rows) |

For scale: two runs of the unquantized teacher with different sampling seeds (temperature 1, top_p 0.95) pick the
same token at 0.71 of the positions. Top-1 agreement is the more stable signal; single-text perplexity on this model
moves by several percent from small numeric changes.

On the calibration chain's own held-out set (6 sequences x 8K, torch head gate): ppl 6.374 vs the teacher's 6.365
(+0.15%), top-1 0.675 vs 0.676.

## Measured speed

On one box: Minisforum MS-S1 MAX, Ryzen AI MAX+ 395 / Radeon 8060S (gfx1151), 128 GB LPDDR5X, Fedora 43, ROCm 7.14.1,
model on a Samsung 990 Pro 1 TB NVMe (PCIe 4.0 x4, ~6 GB/s on these reads). Quality mode, one request at a time
(2026-10-27/28).

| | |
|---|---|
| **Decode with DSpark** (MTP drafts; the default) | **~11 tok/s**: 11.0-11.6 tok/s on 700-token answers; 10.5 tok/s over an 8-turn agent session (11.3 on a 1,525-token answer) |
| **Decode, plain** (`--no-dspark`) | **~9 tok/s**: 8.8-9.2 tok/s on the same answers; 8.3 tok/s over 100 live agent requests at 36-190K context |
| Prefill of new tokens | TTFT of an agent turn (default: fused key gather + predicted reads): +0.8K ~19-24 s, +2.5K ~44 s, +6.5K ~81-84 s; cold 5.6K ~73 s, cold 15K ~147 s. Optional `--fast-prefill` (MMQ, not bit-identical, ppl within noise): +2.5K ~27-30 s, +6.5K ~39 s, cold 15K ~67 s |
| Prompt reuse | an agent turn that extends the previous conversation reuses its cache (~99% of tokens) |
| Engine load | ~33-50 s |

About 6,000 experts (80-86 GiB) stay in RAM; the others are read from the SSD when the router picks them, so decode
is drive-bound: a plain token is ~112 ms, of which ~32 ms are expert reads. A slower drive means slower decode (see
`SSD_SPEED.md` in the Strixolotl repo). DSpark drafts up to 4 tokens with the checkpoint's own MTP layers and
verifies them in one batched pass with exact speculative sampling: the output distribution is the model's at any
temperature, but it is not bit-identical to plain greedy decode (~1.4% of greedy tokens differ, at near-ties).

## Requirements

- AMD Strix Halo (Ryzen AI Max, gfx1151) with **128 GB** of memory; GTT raised to ~123 GiB on the kernel cmdline
  (`amdgpu.gttsize=131072 ttm.pages_limit=33554432`), memlock unlimited.
- Linux with podman and toolbox. Tested on Fedora 43; the engine builds in a Fedora 44 toolbox with ROCm 7.14.
- A local NVMe with **~466 GB free** (GGUF 243.1 GB + Engram 203.1 GB + DSpark sidecar 5.9 GB + one 10.7 GB part while
  downloading). Decode speed depends on the drive's random-read speed on ~15 MB reads: the numbers above are a
  Samsung 990 Pro (PCIe 4.0 x4, ~6 GB/s on these reads). ext4,
  xfs or btrfs; on btrfs the model folders need `chattr +m` (the download script sets it). Not NFS, SMB or exFAT.

## How to run

```bash
git clone https://github.com/snowsh1ne/strixolotl strixolotl && cd strixolotl
scripts/toolbox_create                                   # the xolotl-strix toolbox (ROCm 7.14)
toolbox run -c xolotl-strix scripts/build_xolotl         # the engine, 20-30 min the first time
toolbox run -c xolotl-strix scripts/download_model --repo Snowsh1ne/DeepSeek-V4.1-Flash-1017-GSQ-217-Strixolotl
scripts/verify_setup                                     # on the host: checks + a 32-token reference generation
toolbox run -c xolotl-strix scripts/start_xolotl_dsv41                  # LAN without a key, 1M ctx, DSpark: ~11 tok/s
toolbox run -c xolotl-strix scripts/start_xolotl_dsv41 --no-dspark      # plain decode, ~9 tok/s
toolbox run -c xolotl-strix scripts/start_xolotl_dsv41 --fast-prefill   # ~2x faster prefill of 2.5K+ tokens, not bit-identical
```

The server speaks the OpenAI API at `http://<host>:8150/v1` and the Anthropic API at `/v1/messages`, model id
`strixolotl-v4.1-flash`. The default server listens on the LAN without a key (never forward a port to it);
`--local` or `--api-key KEY` restrict it. Full steps: `QUICKSTART.md` in the Strixolotl repo.

`download_model` downloads one part at a time, checks its sha256, appends it to the GGUF and deletes it, then checks
the whole file. It is resumable. To join the parts by hand instead: `cat V41-1017-GSQ-217.gguf.part-* >
V41-1017-GSQ-217.gguf`, then `sha256sum -c SHA256SUMS` in a folder that holds the joined file.

## Limitations

- One request at a time; ~11 tok/s decode with DSpark (the default), ~9 with `--no-dspark`. Long reasoning turns (10-35K thinking tokens)
  take a while.
- Measured on one box and one SSD. Quality numbers come from 17 held-out rows and one blind-judged coding prompt;
  they are not a benchmark suite.
- Default context in the start script: the model's limit of 1,048,576 tokens (`--no-long`: 262,144).
  Measured in long-context mode: pass-key recall 8/8 at 523K tokens; prefill 58-77 tok/s and slower with depth, so
  reading a long prompt from scratch takes hours (523K: ~2.3 h); plain decode does not slow down with depth (8.24
  tok/s at 80-190K). DSpark at long depth is not measured yet.
- The vision tower is not available with this file.
- The server's greedy decode is not repeatable across requests (the expert arena changes between requests); one-shot
  engine runs at a fixed arena are.

## License

The weights are derived from DeepSeek-V4.1-Flash-1017, which DeepSeek releases under the MIT License
("This repository and the model weights are licensed under the MIT License"). This quant is distributed under the
same license; `LICENSE` is DeepSeek's license text (Copyright (c) 2023 DeepSeek). The Engram tables are downloaded
from DeepSeek's repo under that license. The Strixolotl engine is a separate project: see its repo for its license.

## Credits

- [DeepSeek](https://huggingface.co/deepseek-ai) for DeepSeek-V4.1-Flash-1017.
- [Xolotl](https://github.com/Indras-Mirror/Xolotl) by Indras Mirror (the DeepSeek-V4.1 engine and the GSQ
  quantization pipeline), built on [Strata](https://github.com/Niko1221/Strata) by Niko1221, and
  [ggml / llama.cpp](https://github.com/ggml-org/llama.cpp).
- [antirez/ds4](https://github.com/antirez/ds4) for the imatrix calibration prompts.
- Reference projects for Strix Halo: [strixite](https://github.com/shawnshekari/strixite),
  [halogen](https://github.com/peonist-ai/halogen-flash-server), [vincalkr/DS41Engine](https://github.com/vincalkr/DS41Engine).

```
@misc{deepseekai2026deepseekv41flash,
      title={DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression},
      author={DeepSeek-AI},
      year={2026},
}
```
