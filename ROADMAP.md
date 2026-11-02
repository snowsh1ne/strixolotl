# Strixolotl: status and roadmap

The public summary of where the project stands, what was measured, and what comes next. The detailed lab log behind
it is not published. Numbers come from one box: Minisforum MS-S1 MAX (Ryzen AI MAX+ 395, Radeon 8060S / gfx1151,
128 GB), Fedora 43, ROCm 7.14.1, model on a Samsung 990 Pro 1 TB NVMe. Measured and estimated numbers are kept apart;
"measured" means a run on that box.

## Where it stands (2026-11-01)

The port, the quantization and the performance work are done and frozen. Released as
[0.1.0](https://github.com/snowsh1ne/strixolotl/releases/tag/0.1.0) on 2026-11-01, together with the public Hugging Face
repo.

| | DSpark (default) | plain decode (`--no-dspark`) |
|---|---|---|
| Decode | ~11 tok/s (11.0-11.6 on 700-token answers; 10.5 over an 8-turn agent session) | ~9 tok/s (8.8-9.2; 8.3 over 100 live agent requests at 36-190K context) |
| Time to first token, agent turn adding 0.8K / 2.5K / 6.5K tokens | 21-28 s / ~44 s / ~83 s | 19-24 s / ~44 s / ~81 s |
| Cold 5.6K / 15K prompt | ~74-77 s / ~147 s | ~73 s / – |
| With `--fast-prefill` (optional, not bit-identical) | +2.5K 27-30 s, +6.5K ~39 s, cold 5.6K ~33 s, cold 15K ~67 s | |
| Quality, 17 held-out 8K rows | same output distribution as plain | ppl gap to the teacher +1.61%, top-1 agreement 0.874 |

Model: GSQ-217, our own quantization of DeepSeek-V4.1-Flash-1017. It is 243.1 GB: 99.4% of the routed experts are
Q3_K (~3.44 bits per weight, from DeepSeek's native MXFP4 at ~4.25), the dense half is Q8_0: a 3-bit-class quant. Download: [Snowsh1ne/DeepSeek-V4.1-Flash-1017-GSQ-217-Strixolotl](https://huggingface.co/Snowsh1ne/DeepSeek-V4.1-Flash-1017-GSQ-217-Strixolotl).

## Defaults and switches

`scripts/start_xolotl_dsv41` with no arguments serves the configuration everything above was measured with:

| Default | Off switch | Notes |
|---|---|---|
| GSQ-217 (`--model 1017`) | `--model v41` | |
| LAN without a key (`--host 0.0.0.0 --no-auth`) | `--local`, `--api-key KEY` | an unauthenticated LAN server: never forward a port to it |
| Long context (`--long`, 1,048,576 tokens) | `--no-long` (262,144) | |
| DSpark speculative decode | `--no-dspark` | same output distribution, not bit-identical to plain greedy (~1.4% of greedy tokens differ, at near-ties) |
| Predicted file reads during prompt batches | `--no-pred-reads` | bit-identical; keeps 1.5 GiB pinned (decode at 15K: ~1-2% slower, one run) |
| Fused attention-key gather, driver-memory settle | `STRATA_FUSED_KV=0`, `STRATA_SERVE_DRV_SETTLE=0` | engine defaults, bit-identical |
| `--fast-prefill` | (off by default) | MMQ expert kernels for prompts; see "Prefill" below |

The project's rule: a change that is bit-identical to the reference becomes a default, with an off switch. A change
that is not bit-identical stays an opt-in flag, unless the quality A/B shows it is neutral and the maintainer accepts
it (DSpark).

## Milestones

| Date | What |
|---|---|
| 2026-10-18 | Port of Xolotl from CUDA to HIP/ROCm (gfx1151): engine and tests build clean; UMA memory planning; toolboxes for the engine and for quantization |
| 2026-10-19/20 | Quantization pipeline on the box; first own quant GSQ-170, then GSQ-200 |
| 2026-10-20/21 | First engine runs; exact expert routing; quality-mode server; first real agentic workload |
| 2026-10-21 | Fidelity study: the engine sits at the noise floor of an exact torch reference; four fidelity switches on |
| 2026-10-21/23 | Calibration corpus v2 (48 sequences x 8K, ranked windows); GSQ-217 becomes the release model |
| 2026-10-24 | Prompt-cache fixes for cancelled requests (snapshot point) |
| 2026-10-25 | File-tier read path; long-context mode validated to 523K tokens |
| 2026-10-27 | Samsung 990 Pro: decode +43%, prefill ~2x, losslessly; DSpark (MTP drafts) measured |
| 2026-10-28 | UMA polish (mlocked arena, fewer queue evictions); DSpark serve config; cancel-and-resume for DSpark; release prepared and uploaded to Hugging Face (private) |
| 2026-10-29/30 | Prefill work: fused attention-key gather, predicted reads, MMQ fault fix and `--fast-prefill`; performance frozen |
| 2026-10-31 | DSpark, long context and LAN serving become the start-script defaults |
| 2026-11-01 | Public release 0.1.0 (GitHub) and the public Hugging Face repo |

## Quality A/B (17 held-out rows)

Same text through the engine and through the teacher (DeepSeek's official checkpoint run in torch: FP8 weights,
natively FP4 experts): 17 held-out sequences of 8K tokens, not used
for calibration. Per-token NLL and argmax vs the teacher, pooled over all positions.

| Configuration | ppl gap to the teacher | top-1 agreement |
|---|---|---|
| GSQ-200, no fidelity switches | +2.74% | 0.854 |
| **GSQ-217, quality mode (the release)** | **+1.61%** | **0.874** |
| GSQ-217 + Q6_K dense requant (speed-mode candidate) | +1.13% | 0.872 |
| GSQ-217 + `--fast-prefill` (MMQ prompt kernels) | +1.73% | 0.873 |

- DeepSeek-V4.1 amplifies small numeric changes: its expert routing and the FP4 sparse-attention indexer flip on tiny
  differences. A single 8K text moves by up to ±5-10% in ppl for any change of op order. Rank changes by top-1
  agreement over many rows, never by one text.
- Two sampled runs of the teacher itself (temperature 1, top_p 0.95) pick the same token at 0.71 of positions.
- Fidelity study (2026-10-21, torch ablation): the engine's distance to an exact torch chain with the same experts is
  below that of a control that only changes FP8 op order. Under that control, the share of tokens with an identical
  top-6 expert set falls from 1.00 at layer 0 to 0.22 at layer 21. The engine is at the noise floor; the four fidelity
  switches (FP8 `o` before `wo_a`, BF16 index scores, F32 accumulation for F16-weight prefill GEMMs, an F16 output
  head) stay on as model-faithful and nearly free.

## Quantization (GSQ)

- Xolotl's GPTQ-style pipeline. Every routed expert gets its own format from calibration statistics (tiers from
  ternary PTQ1_0 to Q3_K); the dense half stays Q8_0.
- The checkpoint already ships its routed experts as MXFP4 (~4.25 bits, quantization-aware trained), so a "Q4" file
  of this model is essentially the original. GSQ-217 goes below that, to ~3.44 bits: compare it with 3-bit-class
  quants. No head-to-head against other GGUFs of this model has been measured; the 17-row A/B also includes engine
  numerics noise, so its top-1 figure is not comparable to other projects' published ones.
- Proxy curve (whole model, error vs all-Q2_0): 170 GiB −33%, 200 −71%, 217 −85%; no knee at 170, the deep layers
  gain most from Q3_K. 217 GiB is the ceiling of what the expert tier can serve on this box.
- Corpus v2: 48 sequences x 8,192 tokens (agentic_long 18, agent 4, code 8, reason 6, web 2, zh 3, chat 4, multi 3),
  from real-world agentic coding workflows and the antirez/ds4 imatrix prompts. Windows were rated by several model
  raters, centered per rater and picked greedily with diversity penalties.
- GPTQ is not bit-reproducible run to run: requantizing gives an equivalent file, not the released one.

## Long-context mode

Quality mode with the context at the model's limit of 1,048,576 tokens (the default since 2026-10-31). One
conversation grew in stages (filler: llama.cpp source), with two hidden pass keys per stage:

| Prompt | New tokens read | Prefill | Pass keys recalled |
|---|---|---|---|
| 65K | 65K | 77 tok/s | 2/2 |
| 130K | 65K | 68 tok/s | 4/4 |
| 261K | 131K | 73 tok/s | 6/6 |
| 523K | 262K | 58.5 tok/s | 8/8 |

- Free memory stayed at ~10 GiB through 523K; the larger context costs nothing up front.
- Prefill slows with depth (the attention span graph grows); decode does not (plain 8.24 tok/s at 80-190K). DSpark at
  long depth and contexts above 523K are not measured.

## File-tier read path (2026-10-25)

The experts do not fit in RAM: ~6,000 of 15,360 stay in a pinned arena, the rest are read from the NVMe when routed.

- Decode is drive-bound. A plain token takes ~112 ms: dense half ~58 ms, experts in RAM ~20 ms, ~12.8 expert reads
  from the SSD at ~2.5 ms each.
- Read-path defaults (all bit-identical to the reference): misses read early by a persistent pool straight into
  their arena slot; arena eviction by decayed use count; +10 GiB of extra arena slots during decode, released before
  a prompt batch. Together −28% misses and +4-7% decode.
- Drive choice matters more than anything in the engine: see [SSD_SPEED.md](SSD_SPEED.md).

## DSpark

DSpark drafts up to 4 tokens with the checkpoint's own MTP layers (`mtp.0-2`, a 5.9 GB IQ3_XXS sidecar) and verifies
them in one batched pass with exact speculative sampling.
- Serve config: depth 4, confidence gate 0.6, the drafter keeps its 64 most-used experts per block.
- Result: +22-29% decode in serve (9.0 → 11.0-11.6 tok/s on the same answers).
- A cancelled answer resumes where it stopped instead of re-reading the generated tokens.
- Not bit-identical to plain greedy decode (batched verify arithmetic); the output distribution is the model's.

## Prefill (2026-10-29/30)

| Change | Default | Effect |
|---|---|---|
| Fused sparse-attention key gather (one HIP kernel writes each query's padded F16 keys; a ggml op added by `strix-03-dsv41-gather-kv.patch`) | on, bit-identical | dense half −35-40% (an 8K prompt 158.6 → 138.5 s) |
| Driver-memory settle (wait for what a batch needs at full width, 20 ms polls) | on, bit-identical | −0.23 s per turn; no capped batches after a memory release |
| Predicted file reads (read the experts the last batch routed while the next layer's dense half runs) | on, bit-identical | read waits +844-token turn 13.2 → 7.9 s, +2.5K 5.2 → 0.9 s |
| `--fast-prefill`: MMQ expert kernels for prompt batches, with the MMVQ path's q8_1 activation values | opt-in, not bit-identical | expert kernel ~6x faster; 8K prompt 138 → 60 s |

TTFT, 8-turn agent spec with DSpark, all off → defaults → + `--fast-prefill` (seconds): cold 15K 166 → 147 → 67;
+844 25.5 → 20.8 → 20.6; +2519 51.8 → 44.9 → 30.3; +6555 96.9 → 84.1 → 38.6.

`--fast-prefill` quality: ppl +0.12% vs the default path (95% CI −0.89..+1.26%, 8 of 17 rows worse), top-1 0.8737
→ 0.8732. With ggml's own MMQ activation quantizer the same test read +0.97% (CI +0.17..+2.15%). Upstream measures
this kind of activation quantization at ~0.01-0.1% ppl, so that is most likely amplified noise; the flag uses the
MMVQ-matched values anyway. Not covered by the test: contexts beyond 8K and exact tool-call strings.

## Key findings

- **Decode is bound by the drive.** Expert reads take a third of each plain token; a faster NVMe was the biggest
  speed-up of the project, and it was lossless.
- **KFD queue evictions** stall the whole process. They come from GPU registration of big host ranges, and from
  GPU-mapped pages that move (swap-out, compaction). Fix: the arena and its extension are mlocked.
- **The driver's free-memory figure lags** ~250 ms after unpinning; MemAvailable does not. The GTT budget (~123 GiB)
  is the hard limit on UMA: arena, extension, drafter and dense half must fit together.
- **Small agent turns are drive-bound too.** 844 new tokens still route ~10.6K of 15,360 experts; ~5.5K of them come
  from the NVMe.
- **Throughput figures in tok/s are noisy** on this box (queue evictions); file-tier miss counts and stage timers are the
  reliable comparisons.

## Host setup

- BIOS: UMA frame buffer at the minimum (512 MB); the GPU allocates from GTT.
- Kernel command line used here: `amdgpu.gttsize=131072 ttm.pages_limit=33554432` (~123 GiB GTT), memlock unlimited.
  `amd_iommu=off` is set on this box; it disables the NPU and DMA isolation.
- Swap / zram can stay on: the arena is mlocked.
- Models on a local NVMe (ext4, xfs or btrfs with `chattr +m`), never on NFS. Full recipe: [QUICKSTART.md](QUICKSTART.md).

## What the port changed

Every change in `xolotl/` is marked `strix-port:` in the code. The main ones:
- HIP build wiring: ggml-hip for the dense half, the MMQ prefill types on HIP, and the HIP expert tier. Before this, a
  HIP build silently fell back to CPU experts.
- iGPU detection: ggml types the 8060S as an IGPU.
- UMA memory planning: device budgets from min(driver free, MemAvailable − floor), and the arena sized from
  MemAvailable.
- Zero-copy arena reads by the GPU kernels.
- Pageable-copy safety on HIP.
- llama.cpp patches `strix-01..03` (a HIP shuffle width, deterministic top-k ties, the fused key gather).

## Roadmap

**Next**
- Nothing scheduled. Bug reports and measurements from other boxes and drives are welcome (issues).

**Parked (not planned; measured or estimated where noted)**
- Prefill: prediction buffer inside the decode arena extension (would remove its decode cost); a faster on-demand
  reader (4.95 GB/s today, ~5.4-6.2 reachable on this drive); a persistent prefix cache for side requests (a resumed
  34K conversation: est. 398 s → a few s); keeping the arena extension through small turns.
- Speed mode (lossy presets): paused. The in-budget knobs gave too little; the drive is the limit.
- Vision input: researched (a CPU encoder first), not built.
- Other models: GLM-5.3-Flash (320B / 18B active) would fit at ~3 bpw fully in RAM; not started.
- A second NVMe for expert reads: analysed in [SSD_SPEED.md](SSD_SPEED.md), not built.
