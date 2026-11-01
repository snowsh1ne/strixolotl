# Quickstart: fresh Strix Halo box -> Strixolotl serving DeepSeek-V4.1-Flash-1017 (GSQ-217)

Strixolotl (Xolotl on Strix Halo) is the HIP/ROCm port of the Xolotl/Strata DeepSeek-V4.1 engine. This file takes a
fresh 128 GB Strix Halo box to a running server with our own GSQ quant of DeepSeek-V4.1-Flash-1017 (GSQ-217: 217 GiB
of experts, run `quant-1017-217`) in quality mode. Measurements, milestones and the reasons behind each choice are in
[ROADMAP.md](ROADMAP.md); the overview is [README.md](README.md). This file only gives the recipe.

Two ways to get the model:
- **Path A (recommended): download the prequantized GSQ-217** from Hugging Face (section 4.1). ~447 GB download, no
  quantization, no full DeepSeek checkpoint, only the `xolotl-strix` toolbox.
- **Path B: quantize it yourself** (sections 4.2 and 5). ~510 GB checkpoint download, ~22 h of calibration and
  quantization, both toolboxes. GPTQ is not bit-reproducible, so you get an equivalent file, not this one.

When you're done, `scripts/verify_setup` checks the whole setup (section 8).

Ground rules:
- The repo can live on NFS. Builds, clones, temp files and models must be on the local NVMe (`~/.cache/xolotl`,
  `~/models/xolotl`), never on the share.
- Run the host check first (below: a gfx1151 KFD node listed, GTT ~100+ GiB). Build and run only on the Strix Halo.

Paths below: `REPO=~/projects/Xolotl_StrixHalo` (wherever the share is mounted), `M=~/models/xolotl`,
`G=$M/V41-1017-GSQ`.

---

## 1. Host prerequisites (once, outside any container)

| Item | Value on strixhalo (2026-10) |
|---|---|
| Box | Ryzen AI MAX+ 395 / Radeon 8060S (gfx1151, 40 CUs), 128 GB unified memory |
| OS | Fedora 43, kernel 7.1.10-100.fc43 (6.18.5+ or 7.x works) |
| Packages | `podman` 5.8.2, `toolbox` 0.3 (`podman-compose` only for the compose path) |
| BIOS | UMA frame buffer / dedicated VRAM at the minimum (512 MB); the GPU allocates from GTT |
| Kernel cmdline | `amdgpu.gttsize=131072 ttm.pages_limit=33554432 amd_iommu=off` -> GTT 123 GiB |
| memlock | unlimited: `/etc/security/limits.d/99-memlock-unlimited.conf` with `* soft memlock unlimited` and `* hard memlock unlimited`, then log in again. The engine pins an ~80-87 GiB arena. |
| GPU access | `/dev/kfd` and `/dev/dri` read/write for your user (`render` and `video` groups, or the seat ACL) |
| Disk, path A | ~460 GB free on a local NVMe: GGUF 243.1 GB + Engram shards 203.1 GB + one 10.7 GB part while downloading, images ~9 GB |
| Disk, path B | ~1 TB: HF checkpoint 510 GB (476 GiB), GGUF 243.1 GB, calibration work dir ~47 GiB (calib-1017-v2: 40 GB after the run), images ~24 GB. strixhalo started with 1161 GB free on `/home` (btrfs, `compress=zstd:1`). |
| NVMe speed | Decode is bound by expert reads from the drive: see [SSD_SPEED.md](SSD_SPEED.md) before buying or choosing a drive |

```bash
# kernel cmdline (then reboot); amd_iommu=off is optional (+1-12% elsewhere; turns off the NPU and DMA isolation)
sudo grubby --update-kernel=ALL --args="amdgpu.gttsize=131072 ttm.pages_limit=33554432 amd_iommu=off"
# after the reboot: the host check (gfx1151 node listed, GTT ~123 GiB, MemAvailable ~120 GiB idle)
grep -l 'gfx_target_version 110501' /sys/class/kfd/kfd/topology/nodes/*/properties
for f in /sys/class/drm/card*/device/mem_info_gtt_total; do echo "GTT $(( $(cat $f) >> 30 )) GiB"; done
ulimit -l          # unlimited
```

Notes:
- `amdgpu.gttsize=126976 ttm.pages_limit=32505856` (~124 GiB, the strixite guides' values) also works; the box runs
  the values in the table above.
- Don't set `HSA_OVERRIDE_GFX_VERSION`: the build is native gfx1151. `common.sh` unsets it.
- On btrfs, the model directories must be `chattr +m` (no compression), because btrfs serves O_DIRECT reads of
  compressed extents as buffered reads. `scripts/download_model` sets it itself (path A). For path B run `chattr +m`
  on the host before the GGUF is written (step 5.4); the toolbox has no `chattr` binary. Models on exFAT, NFS or SMB
  are refused.
- ROCm RPM layer cache: `toolbox/Containerfile`'s first two `RUN` layers are byte-identical to
  `~/src/toolbox-qwen4exp-opt/Dockerfile.rocm-7.14`. On strixhalo podman therefore reuses the cached 8.4 GB
  `amdrocm-core-devel7.14-gfx1151` layer, and the image builds in ~30 s. On a fresh box that layer downloads from
  `repo.amd.com`. Don't reformat those two layers.

## 2. Toolboxes

Two images, both built from the repo's `toolbox/` folder. Path A needs only `xolotl-strix`.

```bash
cd $REPO
scripts/toolbox_create            # image localhost/xolotl-strix:rocm7.14            -> toolbox xolotl-strix (engine build + run, hf CLI)
scripts/toolbox_create --quant    # path B only: localhost/xolotl-quant:rocm7.14-torch2.11 -> toolbox xolotl-quant (quantization)
toolbox run -c xolotl-quant python3 -I /opt/xolotl/torch-check.py     # path B: ALL PASS expected
```

`--rebuild` rebuilds the image and recreates the toolbox (with `--quant`, only the torch layer is rebuilt).
toolbox 0.3 takes no extra podman arguments. A toolbox already has the host's `/dev` and ulimits, so memlock comes
from the host.

| Toolbox | Contents (pinned where the files pin them) |
|---|---|
| `xolotl-strix` (8.8 GB) | `toolbox/Containerfile`: Fedora 44, `amdrocm-core-devel7.14-gfx1151` from `https://repo.amd.com/rocm/packages-multi-arch/rhel10/x86_64` (resolves to ROCm **7.14.1**, HIP 7.14.60850, hipBLASLt 100401, tree `/opt/rocm/core-7.14`), gcc/cmake/ninja, Python 3.14 venv `/opt/xolotl/venv` with `requirements-serve.txt` (numpy 2.5.3 etc., pinned) + `huggingface_hub[cli,hf_xet]`. `rocm-discover.py` writes `/opt/xolotl/rocm.env`. |
| `xolotl-quant` (15.3 GB) | `toolbox/Containerfile.quant`: the image above + **torch 2.11.0+rocm7.13.0** from `https://repo.amd.com/rocm/whl/gfx1151/`. torch loads its own ROCm 7.13 pip libraries, not `/opt/rocm/core`. It also gets `requirements-quant.txt`, which is not pinned. Installed today: safetensors 0.8.0, tokenizers 0.23.3, transformers 5.20.1, datasets 5.1.0, pyarrow 26.0.0. |

Reproducibility note: the Containerfiles are the declarative spec. Three things in them float: the ROCm package
(any 7.14.x), `requirements-quant.txt`, and `huggingface_hub`. Pinning ROCm to `-7.14.1` would stop the layer
sharing with the other toolboxes. For a bit-exact rebuild, pin the versions listed above.

## 3. Build the engine (in `xolotl-strix`)

```bash
toolbox enter xolotl-strix
cd $REPO
scripts/build_xolotl          # ds41_generate; 20-30 min the first time; log ~/.cache/xolotl/build-strix/build.log
```

- The script clones llama.cpp into `~/.cache/xolotl/llama.cpp` and pins it to `3cf03257`. It then applies
  `xolotl/tools/xolotl/ggml/xolotl-ggml.patch` (upstream's, keep it untouched) and every `strix-*.patch` in name
  order:
  - `strix-01-hip-shfl-width.patch`: HIP 4-argument shuffle.
  - `strix-02-topk-deterministic-ties.patch`: top-k ties go to the lowest index, which makes runs deterministic.
- **It force-checks-out llama.cpp on every build.** `xolotl-ggml.patch` no longer reverse-applies once strix-02 is
  on top, so the tree is reset and re-patched each time. Any ggml change you make in the checkout is lost. Make it a
  new `strix-NN-*.patch` (`git diff > ...` in the checkout).
- Output: `~/.cache/xolotl/build-strix/ds41_generate`, with the ggml-hip dense half and the HIP expert tier
  (`ds41_moe_cuda`).
- Optional:
  - `scripts/build_xolotl --tests`, then `ctest --test-dir ~/.cache/xolotl/build-strix -j8`. Expect 112/116 pass;
    the rest need model files.
  - ASan build: run in xolotl-strix, then run the engine with `ASAN_OPTIONS=protect_shadow_gap=0:detect_leaks=0`.
    ```bash
    XOLOTL_BUILD_DIR=~/.cache/xolotl/build-asan scripts/build_xolotl -- -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DCMAKE_C_COMPILER=/opt/rocm/core-7.14/llvm/bin/clang -DCMAKE_CXX_COMPILER=/opt/rocm/core-7.14/llvm/bin/clang++ \
      "-DCMAKE_CXX_FLAGS=-fsanitize=address -fno-omit-frame-pointer" "-DCMAKE_EXE_LINKER_FLAGS=-fsanitize=address"
    ```

## 4. Get the model

### 4.1 Path A: download the prequantized GSQ-217 (recommended)

```bash
toolbox enter xolotl-strix
cd $REPO
scripts/download_model --repo Snowsh1ne/DeepSeek-V4.1-Flash-1017-GSQ-217-Strixolotl --dry-run    # plan, disk check; small manifest only
scripts/download_model --repo Snowsh1ne/DeepSeek-V4.1-Flash-1017-GSQ-217-Strixolotl              # resumable; re-run after any interruption
```

What it does (details in the script header):
1. The GGUF (243,075,678,208 bytes) comes as 23 byte-range parts of 10 GiB. Each part is downloaded, checked against
   its sha256 from `manifest.json`, appended at its offset to `$G/V41-1017-GSQ-217.gguf` and deleted. At the end the
   whole file is checked (~2-4 min). Peak disk: the GGUF + one part.
2. `rank-217.bin`, `V41-1017-GSQ-217.gguf.experts.json` and `tokenizer/` go into `$G`, sha256-checked.
3. The Engram tables: the engine reads `layers.{1,14}.engram.embed.{weight,scale}` (202.8 GB of FP8 rows) from
   shards 47 and 48 of `deepseek-ai/DeepSeek-V4.1-Flash-1017`. The script downloads only those two shards (2 x 101.5
   GB) at the pinned revision `1bae0e10...`, checks their sha256, and writes a minimal
   `model.safetensors.index.json` next to them in `$M/V41-1017-Engram`. If you already have the full DeepSeek
   checkpoint, pass `--engram-dir <its dir>` (add `--verify-engram` to hash it) and nothing is downloaded for Engram.
4. `$M/V41-1017.paths` (an existing one is kept as `.bak-<time>`).

On btrfs it sets `chattr +m` on `$G` and `$M/V41-1017-Engram` before writing (no root needed). Then go to section 6.

### 4.2 Path B: download the FP8 checkpoint (host or a toolbox)

```bash
cd $REPO
scripts/fetch_1017_fp8 --all [--until 07:45] [--jobs 2]    # deepseek-ai/DeepSeek-V4.1-Flash-1017 -> $M/DeepSeek-V4.1-Flash-1017
```

- Contents: all 48 shards + config, `inference/`, `encoding/`, tokenizer. That's 510 GB (476 GiB on disk).
- Each file is sha256-checked and gets a `<file>.ok` marker. The calibration waits on those markers.
- Resumable. `--until HH:MM` stops cleanly at a deadline (exit 3); run it again to continue. The log is
  `$M/DeepSeek-V4.1-Flash-1017/fetch.log`.
- To split the download: `--no-engram` (the default) gets everything except shards 47/48 (~313 GB) and is enough for
  calibration layer 0. `--engram` gets shards 47/48 (2 x ~101.5 GB), which layers 1/14 and the engine need.
- **Keep this directory.** In path B the engine reads the Engram tables straight from it (shards 47/48 via
  `model.safetensors.index.json`). It's also the quantization source.

## 5. Quantize (path B only: GSQ-217, in `xolotl-quant`)

Every step below runs inside `toolbox enter xolotl-quant`, after `cd $REPO && . scripts/v41_1017.env`. The env file
sets:
- `V41_SRC`, `V41_REF` (the checkpoint and its `inference/`)
- `V41_ENCODING` (the 1017's own `encoding/encoding.py`)
- `V41_WORK` (default `$M/v41-work/calib-1017`; GSQ-217 was made in `$M/v41-work/calib-1017-v2`: export
  `V41_WORK=$M/v41-work/calib-1017-v2` before sourcing)
- `V41_ANTIREZ`, `V41_SESSION_ROOTS`, `V41_CODE_GLOBS`, `V41_DOC_GLOBS`, `V41_GGUF_PY`, `V41_ACT_ORDER=1`,
  `DS4_LOCK`, `PYTHONPYCACHEPREFIX`, `HF_DATASETS_OFFLINE=1`

Long steps: start them detached so they survive a logout, e.g.
`setsid nohup toolbox run -c xolotl-quant bash -c '. scripts/v41_1017.env && ...' > log 2>&1 < /dev/null &`.
The whole GSQ-217 chain (calibration -> quant pass) is in `$M/v41-work/calib-1017-v2/run_chain.sh`.

> GPTQ is **not bit-reproducible** from run to run: the expert token counts differ by layer 0, and nearly every
> code differs from layer 1 on. A redo gives a different (equivalent) GGUF. Always record which run produced a GGUF.
> The served one is `quant-1017-217`; `quant-1017-200` (GSQ-200) is kept for comparison.

### 5.0 Calibration inputs (once, ~1 GB)

```bash
R=~/.cache/xolotl/calib-refs
# antirez/ds4 imatrix prompts, pinned commit
D=$R/antirez-ds4-imatrix/dataset; mkdir -p $D
B=https://raw.githubusercontent.com/antirez/ds4/fc80bd695da76ee14bcb86820bf900a5cdfda806/gguf-tools/imatrix
for f in prompts.jsonl manifest.json README.md; do curl -sSfL -o $D/$f $B/dataset/$f; done
# Chinese Wikipedia, one shard (zh pool)
mkdir -p $R/wikipedia-zh && curl -sSfL -o $R/wikipedia-zh/train-00002-of-00006.parquet \
  https://huggingface.co/datasets/wikimedia/wikipedia/resolve/main/20231101.zh/train-00002-of-00006.parquet
# pile-10k into the HF datasets cache (prep reads it offline afterwards)
HF_DATASETS_OFFLINE=0 python3 -c 'import datasets; datasets.load_dataset("NeelNanda/pile-10k", split="train")'
# agentic coding sessions (Claude Code-format JSONL logs) for the agentic pools: list the folders that hold them
export V41_SESSION_ROOTS=~/.claude/projects   # ":"-separated; add more folders as needed
```

`V41_GGUF_PY` points at `~/.cache/xolotl/llama.cpp/gguf-py`, so build the engine (step 3) first.

### 5.1 Corpus v2 (~1 min)

```bash
cd ~ && python3 $REPO/xolotl/tools/v41/build_corpus.py \
  --zh-parquet ~/.cache/xolotl/calib-refs/wikipedia-zh/train-00002-of-00006.parquet \
  --windows 4 --spread --tool-chars 2000 --tool-arg-chars 4000 --sessions 400 \
  --select $V41_WORK/corpus-select.json --exclude-session <live session id>
cd $REPO
```

- The agentic pools come from real-world agentic coding sessions (multi-turn tool use) in Claude Code's JSONL format,
  read from the folders in `V41_SESSION_ROOTS`. GSQ-217 used 236 windows from 82 sessions, 1.79M tokens.
- `build_corpus.py` strips harness noise (system reminders, task notifications, slash-command blocks) and packs the
  windows in a deliberate order. `corpus-select.json` is the swarm ranking (ROADMAP "Quantization (GSQ)"); without it,
  the default order is used.
- Output in `$V41_WORK/corpus/`: `sessions.jsonl`, `zh_wiki.jsonl` (400 articles), `antirez_*.jsonl`.
- The corpus depends on which sessions are available when you run it, so it isn't byte-reproducible.
- Corpus v1 (GSQ-200): the defaults `--max-tokens 8191 --sessions 40 --windows 1` in `calib-1017`.

### 5.2 Prep (~2 min)

```bash
cd ~ && python3 $REPO/xolotl/tools/v41/v41_calib.py prep --n 48 --seq 8192 --corpus $V41_WORK/corpus \
  --mix agentic_long=18,agent=4,code=8,reason=6,web=2,zh=3,chat=4,multi=3
cd $REPO
```

Output: `$V41_WORK/{tokens.npy,meta.json,hashes.npy}`, 48 x 8192 tokens, 11 of them held out (eval). (GSQ-200: `--n
32 --mix agentic_long=8,agent=5,code=6,reason=3,web=2,zh=2,chat=4,multi=2`, 6 held out.)

### 5.3 Init + calibration, layers 0-39 (~12.5 h at 48 sequences)

```bash
NEED_GIB=60 MAX_GIB=80 bash xolotl/tools/v41/calib_layers.sh 0 39
```

- **Init:** runs first on a fresh `V41_WORK` (embedding -> `h/L00.bin`).
- **Per layer:** attention, MoE and the full 3-tier screening (ptq1 / q2_0 / q3_k GPTQ). Each layer runs under
  memguard behind `DS4_LOCK` and needs MemAvailable >= 60 GiB, with a cap of 80 GiB.
- **Time:** ~19 min per layer at 48 sequences (~15 min at 32). The Engram layers 1 and 14 take longer, because of
  random reads of the 101 GB tables. A layer waits for its shard's `.ok` marker.
- **Resume and stop:** resumable (a layer is done once `stats/LNN.json` exists). `touch $V41_WORK/STOP` stops
  between layers. A CUDA OOM retries the layer with half the rows.
- **Output:** `$V41_WORK/stats/L00..L39.json`, `h/` (15 GiB per layer file at 48 sequences), `calib.log`, `logs/LNN.log`.
- **Teacher head gate:** runs automatically after layer 39 (`v41_calib.py head` -> `stats/head_L40.json`). Expected
  for calib-1017-v2: ALL/eval ppl **6.365**, top-1 0.676 (calib-1017: 7.041 / 0.659; different held-out text).

### 5.4-5.7 One command for head gate -> alloc -> layout -> quant pass

```bash
# on the host first (the toolbox has no chattr):
mkdir -p ~/models/xolotl/V41-1017-GSQ && chattr +m ~/models/xolotl/V41-1017-GSQ
# then:
setsid nohup toolbox run -c xolotl-quant env V41_WORK=$M/v41-work/calib-1017-v2 BUDGET_GIB=217 bash scripts/v41_1017_night.sh \
  > $M/v41-work/calib-1017-v2/logs/night-217.out 2>&1 < /dev/null &
```

- `scripts/v41_1017_night.sh` waits for calibration, then runs the four steps below.
- It stops at the first red gate. Its log is `$V41_WORK/night.log`. Stop it with `touch $V41_WORK/night.STOP` and
  `touch <qwork>/STOP`.
- `BUDGET_GIB` defaults to 200 (GSQ-200); set 217 for the served file. 217 GiB is the ceiling of the current tiers.
- The steps by hand (each one is skipped if its output already exists):

5.4 Teacher head gate. Already done if 5.3 ran to layer 39. Gate: ppl finite, ppl < 30, top-1 > 0.35.
```bash
MEMGUARD_ALLOW_COMFY=0 bash xolotl/tools/ds4/memguard.sh 80 60 -- python3 xolotl/tools/v41/v41_calib.py head   # -> $V41_WORK/stats/head_L40.json
```

5.5 Tier map, ~40 s. Optional: first log the proxy curve over budgets.
```bash
for B in 142.38 155 170 185 200 217; do echo "== $B"; python3 xolotl/tools/v41/v41_alloc.py --work $V41_WORK --layers 0-39 --per-proj --budget-gib $B | sed -n 2p; done
python3 xolotl/tools/v41/v41_alloc.py --work $V41_WORK --layers 0-39 --per-proj --budget-gib 217 --out $V41_WORK/alloc_217.json
```
Proxy error vs all-Q2_0 at equal bytes (calib-1017, corpus v1):

| Budget (GiB) | 142 | 155 | 170 | 185 | 200 | **217** |
|---|---|---|---|---|---|---|
| Proxy error | -19% | -25% | -33% | -52% | -71% | **-85%** |

There's no knee at 170; the deep layers gain most from Q3_K above it. The 217 map is 99.4% q3_k.

5.6 GGUF layout, ~1.5 min. This writes a sparse 243 GB file: dense tensors filled, expert holes left for 5.7. It
needs the file's size + 8 GiB free (~235 GiB). `--no-engram`: the Engram stays in the HF dir.
```bash
cd ~ && python3 $REPO/xolotl/tools/v41/v41_gguf.py --out $G/V41-1017-GSQ-217.gguf --rtn-map $V41_WORK/alloc_217.json \
  --experts-in-place --no-engram --threads 8 && cd $REPO        # also writes $G/V41-1017-GSQ-217.gguf.experts.json
```

5.7 GSQ quant pass, ~8.7 h. It writes the experts into the GGUF in place and runs its own head gate.
```bash
NEED_GIB=60 MAX_GIB=80 bash xolotl/tools/v41/quant_layers.sh $M/v41-work/quant-1017-217 0 39 -- \
  --map $V41_WORK/alloc_217.json --gguf $G/V41-1017-GSQ-217.gguf
```
Expected: all `rc=0`, then `quant-1017-217/stats/head_L40.json` with ALL/eval ppl **6.374**, top-1 0.675 (+0.15% vs
the teacher on calib-1017-v2's held-out set).

### 5.8 Verify the GGUF (~1.5 min)

```bash
cd ~ && python3 $REPO/xolotl/tools/v41/v41_gguf.py --verify $G/V41-1017-GSQ-217.gguf \
  --rtn-map $V41_WORK/alloc_217.json --experts-in-place --no-engram && cd $REPO
# -> verify: <N> tensors checked, 0 problems   (1295 for GSQ-200; GSQ-217's count is not in the logs)
```

### 5.9 rank.bin and tokenizer (seconds)

```bash
python3 xolotl/tools/v41/engine/v41_rank.py --stats $V41_WORK/stats --map $V41_WORK/alloc_217.json --out $G/rank-217.bin
python3 xolotl/tools/v41/serve/export_tokenizer.py $V41_SRC $G/tokenizer
```

### 5.10 The paths file (by hand in path B)

`~/models/xolotl/V41-1017.paths`:
```bash
# written by hand: our own GSQ-217 quant (run quant-1017-217); Engram read from the HF checkpoint
XOLOTL_DSV41_NAME=V41-1017
XOLOTL_DSV41_REPO=local
XOLOTL_DSV41_GGUF="$HOME/models/xolotl/V41-1017-GSQ/V41-1017-GSQ-217.gguf"
XOLOTL_DSV41_RANK="$HOME/models/xolotl/V41-1017-GSQ/rank-217.bin"
XOLOTL_DSV41_TOKENIZER="$HOME/models/xolotl/V41-1017-GSQ/tokenizer"
XOLOTL_DSV41_ENGRAM="$HOME/models/xolotl/DeepSeek-V4.1-Flash-1017"
XOLOTL_DSV41_DSPARK="/mnt/fast/V41-1017-MTP/V41-1017-MTP.IQ3_XXS.gguf"   # only for --dspark
```
`--dspark` needs the MTP sidecar. Path A: `download_model` fetches `dspark/V41-1017-MTP.IQ3_XXS.gguf` and writes this
line. Path B: build it from shards 44-46 of the checkpoint (CPU only, a few minutes):
```bash
toolbox run -c xolotl-quant env V41_SRC=$M/DeepSeek-V4.1-Flash-1017 V41_GGUF_PY=$HOME/.cache/xolotl/llama.cpp/gguf-py \
  python3 xolotl/tools/v41/v41_sidecar.py --out /mnt/fast/V41-1017-MTP/V41-1017-MTP.gguf --no-vision   # 7.9 GB, MXFP4 experts
toolbox run -c xolotl-strix ~/.cache/xolotl/build-strix/v41_sidecar_requant \
  --in /mnt/fast/V41-1017-MTP/V41-1017-MTP.gguf --out /mnt/fast/V41-1017-MTP/V41-1017-MTP.IQ3_XXS.gguf --type iq3_xxs   # 5.9 GB
```

Copies of the live paths files are in `scripts/paths/` (`V41-1017.paths` = GSQ-217, the served model;
`V41-1017-200.paths` = GSQ-200). The start script reads `~/models/xolotl/V41-1017.paths`, not the repo copy.

## 6. Run the server (in `xolotl-strix`)

```bash
toolbox enter xolotl-strix
cd $REPO
scripts/start_xolotl_dsv41                  # defaults: --model 1017, LAN without a key (--host 0.0.0.0 --no-auth; no port
                                            # forwarding to this box), --long (1,048,576 ctx), --dspark, predicted reads
scripts/start_xolotl_dsv41 --fast-prefill   # + MMQ prompt kernels: ~2x faster prefill of 2.5K+ tokens, not bit-identical
# plain decode: --no-dspark; 262,144 ctx: --no-long; local only: --local; with a key: --api-key KEY
# --dry-run writes the config and prints the commands without starting anything
```

- **Mode:** quality mode is the default and the only finished mode. Decode on a Samsung 990 Pro: ~11 tok/s with
  DSpark (the default since 2026-10-31: the checkpoint's MTP heads draft, one batched pass verifies; same output
  distribution, not bit-identical to plain greedy), ~9 tok/s plain (`--no-dspark`). Speed mode is paused: decode is bound by
  the drive, not by compute ([SSD_SPEED.md](SSD_SPEED.md)).
- **API:** `http://<host>:8150/v1` (OpenAI) and `/v1/messages` (Anthropic), model id `strixolotl-v4.1-flash`.
- **Logs:** the engine log is streamed to the terminal and kept in `~/.local/state/xolotl/logs/ds41-engine.log`.
  The config is `~/.local/state/xolotl/strata-ds41-strix.json`. Ctrl-C stops both.
- **Tool calls:** `--tool-parse auto` returns DSML tool calls as `tool_calls` only for requests that declare `tools`
  (and a `tool_choice` other than `none`). Any other request gets the DSML verbatim in `content`, for clients that
  declare no tools. The default `always` parses them for every request. A request's own `"tool_parse"` wins.
- **Smoke test:**
  ```bash
  curl -s http://127.0.0.1:8150/v1/chat/completions -H 'content-type: application/json' \
    -d '{"messages":[{"role":"user","content":"Write a Python function that reverses a list."}],"max_tokens":256}'
  ```

What quality mode sets (all in the start script; every setting is lossless or model-faithful):

| Setting | Why |
|---|---|
| `STRATA_DENSE_REQUANT=""` | The dense half stays the file's Q8_0. The wrapper's default would requantize it to q6_K, a speed choice. |
| `STRATA_UMA_ZEROCOPY=1`, `--uma-cache-gib 0` | The kernels read arena slots in place: tokens are bit-identical, and the expert tier is 42 ms/token faster. A cache would be a second RAM copy. |
| `--pf-b 0` + `STRATA_PF_NO_DUMMY=1` | No RAM->RAM prefetch staging on UMA. |
| `--uma on --uma-reserve-gib 24` | The arena is MemAvailable after the dense half, minus 24, which gives ~85-87 GiB. |
| `--no-chunk-mmq` | The 8192-token MMQ prompt path faults (`mul_mat_q<Q3_K,128>`). MMVQ is also faster here. |
| `--route-bias 0 --route-bias-file 0` | Exact routing. `--arena-adapt` stays on: the conversation's experts move into the arena. |
| `--kv-fp4`, `--ctx 262144`, `--prompt-batch 8192`, `--pcie 1 --tier-batch-pcie 1` | Every expert runs on the GPU. |
| `STRATA_DBG_OQUANT=1`, `STRATA_IDX_BF16=1`, `STRATA_DBG_PREC32=1`, `STRATA_DENSE_F16=output.weight` | The four fidelity switches, closer to model.py: FP8 o before wo_a, BF16 index scores, F32 accumulation in F16-weight prefill GEMMs, an F16 head. |
| `STRATA_FILE_FAST=1`, `STRATA_ARENA_LFU=0.995`, `STRATA_ARENA_EXT_GIB=10` | The file-tier read path (2026-10-25): file misses are read early by a persistent pool straight into their arena slot and run on the GPU; arena eviction by decayed use count; +10 GiB of arena slots during decode, released before a prompt batch. Bit-identical to each other and to verify_setup's reference path. Warm bench: misses 18.0 -> 13.0 per token, decode +4-7%. ROADMAP "File-tier read path". |
| Wrapper (`xolotl/tools/v41/serve/ds41-engine-strix.sh`) | `STRATA_PRESTAGE_GIB=0`, `GGML_CUDA_DISABLE_GRAPHS=1`, `STRATA_NO_LARGEPAGES=1`, DSpark off |

Any of these can be overridden from the environment (every `STRATA_*` variable is passed through) or with
`-- <engine args>`.

Check the startup log:
- The only `dense requant` line is the F16 head (`1 matrices -> (+ STRATA_DENSE_F16 -> f16)`). There's no q6_K.
- `arena: zero-copy` appears.
- The arena is ~86.7 GiB (~6,680 experts).

**pi agent provider:** the `xolotl-ds41` entry in `~/projects/models.json` (copy that entry rather than
retyping it). It sets:
- Connection: `api: "openai-completions"`, `baseUrl` = the server's `/v1` (`http://strixhalo:8150/v1` on the LAN;
  the file has the Tailscale IP), `apiKey: "none"`.
- Model: id `strixolotl-v4.1-flash`, `reasoning: true`.
- Limits: `contextWindow: 212000` and `maxTokens: 49152` (fit the `--no-long` 262144 ctx; the default 1048576 ctx allows
  a larger `contextWindow`).
- Sampling: temperature 1.0, top_p 0.95.
- Thinking: `thinkingFormat: "chat-template"` with `reasoning_effort` / `enable_thinking` kwargs. The thinking levels
  off/minimal/low/medium/high/xhigh/max map to `none/25/50/62/75/88/100`.

### 6.1 Long context (`--long`, the default since 2026-10-31)

```bash
scripts/start_xolotl_dsv41              # --long is the default: --ctx 1048576, the model's limit
scripts/start_xolotl_dsv41 --no-long    # --ctx 262144
```

- Same quality mode, same memory plan: the larger context costs no memory up front (free memory stayed at ~10 GiB
  through a 523K-token conversation). Pass-key recall was 8/8 at 523K (README "Modes").
- Prefill: 77 tok/s at 65K, 73 at 261K, 58.5 at 523K (it drops with depth). A cold 500K-token prompt takes hours;
  later turns read only their new tokens as long as the client sends the conversation unchanged (section 6). An edit
  far back in the history, or a server restart, means reading the whole prompt again.
- Decode does not slow down with depth (plain: 8.24 tok/s at 80-190K). DSpark at long depth is not measured yet.
- Clients: raise the context window to match, e.g. pi `contextWindow: 999000` with `maxTokens: 49152` (prompt + max
  tokens must fit 1,048,576).

## 7. Numbers to check against

| What | Expected |
|---|---|
| GSQ-217 file | 243,075,678,208 bytes; sha256 in the release's `SHA256SUMS` (path A) |
| Teacher head gate (`calib-1017-v2/stats/head_L40.json`), ALL/eval | ppl **6.365**, top-1 0.676 |
| GSQ-217 head gate (`quant-1017-217/stats/head_L40.json`), ALL/eval | ppl **6.374**, top-1 0.675 (+0.15%) |
| Same-text engine A/B, 17 held-out 8K rows (ROADMAP "Quality A/B") | ppl gap to the teacher +1.61%, top-1 = teacher 0.874 |
| `v41_gguf.py --verify` (path B) | `verify: ... tensors checked, 0 problems` (GSQ-200: 1295 tensors) |
| Engine load | ~30 s (the arena is seeded from NVMe; Samsung 990 Pro) |
| Server arena | ~85-87 GiB plain (~6,000-6,680 experts), ~80 GiB with `--dspark` (reserve 30) |
| **Decode, DSpark** (default) | **~11 tok/s**: 11.0-11.6 tok/s on 700-token answers; 10.5 tok/s over an 8-turn agent session (11.3 on a 1,525-token answer). Samsung 990 Pro. |
| **Decode, plain** (`--no-dspark`) | **~9 tok/s**: 8.8-9.2 tok/s on the same answers; 8.3 tok/s over 100 live agent requests at 36-190K context. ~a third of each token waits on NVMe reads ([SSD_SPEED.md](SSD_SPEED.md)). |
| Prefill of new tokens | TTFT of an agent turn (default: fused key gather + predicted reads): +0.8K ~19-24 s, +2.5K ~44 s, +6.5K ~81-84 s; cold 5.6K ~73 s, cold 15K ~147 s. Optional `--fast-prefill` (MMQ, not bit-identical, ppl within noise): +2.5K ~27-30 s, +6.5K ~39 s, cold 15K ~67 s (docs/PREFILL.md) |
| `scripts/verify_setup` | everything PASS; `gen` = 32/32 tokens identical to `scripts/verify_setup.ref` (~1 min engine run; cold arena, no arena-adapt) |

GSQ-200 for comparison (calib-1017, its own held-out set): head gate 7.256 / 0.654 vs the teacher's 7.041 / 0.659
(+3.1%), file 224,824,197,120 bytes. Head gates on different held-out sets are not comparable with each other.

## 8. Verify the setup

```bash
cd $REPO
scripts/verify_setup            # static checks + a deterministic 32-token generation vs scripts/verify_setup.ref (~2 min)
scripts/verify_setup --full     # + path A: sha256 of the GGUF vs the manifest; path B: v41_gguf.py --verify (~2-4 min more)
scripts/verify_setup --no-gen   # static checks only (safe while the server runs)
scripts/verify_setup --record   # path B, after a re-quant or an intended engine change: two runs must match, then the ref is rewritten
```

Run it on the host, not in a toolbox. It tells the two layouts apart by the paths file (`XOLOTL_DSV41_REPO=hf:...`
= path A).

- **Checks:**
  - host: gfx1151 node, /dev/kfd, GTT >= 100 GiB, MemAvailable, memlock
  - toolboxes: xolotl-strix with ROCm; xolotl-quant with torch on the GPU (path B only)
  - the engine binary is newer than the patches; the llama.cpp tree == pin + xolotl-ggml + strix-NN
  - model files: path A the Engram index + shards; path B 48/48 `.ok` markers. Both: the GSQ-217 files,
    `chattr +m`, the paths file
  - a generation: greedy, `--arena-gib 80`, exact routing, quality-mode env
- **Refusals:** the generation won't start while an engine, the server or another GPU user is running, or while
  MemAvailable < 110 GiB.
- **Runs and logs:** each run is under a memory watchdog and is killed on timeout. Logs go to
  `~/.cache/xolotl/verify/<timestamp>/`.
- **Reference:** `scripts/verify_setup.ref` is tied to GSQ-217 and rank-217.bin (`gguf_id`). A downloaded GSQ-217 is
  the same file, so it must match. GPTQ isn't reproducible, so a new quant (path B) needs `--record`.

## 9. Pitfalls

- **Memory reserve.** With the Q8_0 dense half, an 8192-token prompt batch uses ~14.5 GiB beyond the arena. Runs
  stall when MemAvailable nears ~6 GiB:
  - Measured at a fixed arena: 95/93 GiB OOM, 90 GiB stalled, 86 worked (min free 8.5 GiB), 82 worked (12.4 GiB).
  - Hence reserve 24. Raise it to 26 if a long prompt ever stalls. `--dspark` uses 30 (its drafter is pinned too).
  - Never go below ~16: strixite saw hard hangs near 0.5 GiB free.
  - The arena is sized from MemAvailable **at start**. Apps opened later (a browser) come out of the remaining
    headroom.
- **A killed engine frees its pinned ~80-90 GiB arena over several seconds.** Wait for MemAvailable >= 110 GiB
  before starting the next run (`verify_setup` and `scripts/dev/gen.sh` do this). Two engines at once will OOM the box.
- **Another model server** (halogen etc.) shows up as low MemAvailable (< ~115 GiB idle). Stop it before starting
  anything big.
- **Span (`--prefill-chunk`, default 512 = min with the prompt batch) above 512 needs much more memory:** 4096 makes
  an 18 GB graph and OOMs. Keep S = 512.
- **Single-text ppl is noise.** DeepSeek-V4.1's late-context predictions are hypersensitive (top-512 over FP4 indexer
  q/k). Any perturbation moves one 4K text's ppl by several percent. Compare top-1 agreement with the teacher over
  many sequences, teacher KL, or task outcomes, never one text.
- **The drive sets decode speed.** ~13 expert misses per token at 7.4-8.4 ms each on the measured drive, which
  spends 14-34% of decode in ~18 s slow phases under sustained reads. [SSD_SPEED.md](SSD_SPEED.md) has the estimates and a `fio` test.
- **NFS vs local disk.** The share is 100 Mbit on HDDs:
  - Builds, the llama.cpp clone, `__pycache__` (`PYTHONPYCACHEPREFIX`), the work dirs and models all go on the local
    NVMe.
  - `common.sh` refuses NFS/exFAT/SMB for models.
  - The GGUF must be in a `chattr +m` dir on btrfs.
- **Don't delete the Engram tables.** The engine reads them at run time: `$M/V41-1017-Engram` (path A) or the HF
  checkpoint dir (path B). In path B, the two shards plus a minimal index are enough to keep if the rest of the
  checkpoint is deleted (`scripts/download_model --engram-dir` or `xolotl/tools/v41/extract_engram.py`).
- **ggml edits:** only as `strix-NN-*.patch`, because `build_xolotl` resets the llama.cpp checkout (section 3).
- **Short new-token turns** (<= 96 tokens, `STRATA_SERVE_STEP_MAX`) run as exact passes (~7-8 tok/s), and the first
  agent turn is cold. Both are expected.
