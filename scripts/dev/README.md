# scripts/dev - engine test harness (2026-10-19 onwards)

Run on the Strix Halo only (the host check in QUICKSTART.md). Logs and dumps go to `~/.cache/xolotl/runs/` (local disk).

| Script | What |
|---|---|
| `gen.sh NAME [ds41_generate args]` | One-shot engine run in xolotl-strix. Log `runs/NAME.log`, memory `runs/NAME.mem`. Env: `ENVX="export A=1;"` (engine env), `TMO` timeout s (kills the engine itself), `TAILN`, `MW_MIN` watchdog MiB (kills below), `MW_START` waits for MemAvailable >= N GiB before starting (a killed engine frees its ~90 GiB pinned arena over seconds). `PATHS` = a `.paths` file (default `~/models/xolotl/V41-1017.paths`). |
| `memwatch.sh LOG` | MemAvailable + GTT used + NVMe temperature every second; kills ds41_generate below `MW_MIN`. |
| `fidelity.sh` | The fidelity battery (agent4k, quality mode, switches one by one + all). |
| `sweep.sh` | File-penalty sweep (route bias). |
| `cmpref.py run...` | Per-token distance of `runs/nll-<run>.txt` dumps to `runs/nll-gsqtorch.txt` (NOTE: that file is a sibling GPTQ run, not the GGUF's model - use the ablation's `nll_E` as the reference, see ROADMAP "Quality A/B"). |
| `vsT.py run...` | Same vs the teacher (`runs/nll-teacher.txt`) plus pairwise. |
| `ab_v1v2.sh`, `ab_prep.py`, `ab_report.py` | Quality mode v1 (GSQ-200, no fidelity switches) vs v2 (GSQ-217 + switches): `--ppl` on the 17 held-out rows of both chains vs the teacher, paired per row with a bootstrap CI, plus a speed A/B (6144-token prompt + 1000 forced tokens; v1, v2 and GSQ-200 with the v2 switches). Output `runs/ab/`. Server stopped first. |
| `ab_speed.sh`, `ab_speed_prep.py`, `ab_speed_chain.py` | Speed-mode sweep phase 1: warm speed bench (held-out ~15K agent prompt + 3000 forced tokens of quality mode's own output, window >= step 1024) for the speed knobs, then their quality on the 17 rows, then restarts the server. Output `runs/ab/warm/`, report in `ab_report.py`. |
| `teacher_nll8k.py ROW OUT [ROW OUT..]` | `teacher_nll.py` over the whole 8K row (V41_WORK = the chain). |
| `teacher_nll.py ROW OUT` | Per-position NLL/argmax/margin from a chain's h/L40 (V41_WORK = the chain's work dir). |
| `mkprompt.py`, `mklong.py` | Render chat prompts with the 1017's encoding.py into int32 id files. |
| `worst.py`, `cmpnll.py`, `lmdiff.py` | Inspect per-token NLL dumps / per-layer state dumps (`STRATA_LM_DUMP`). |
| `ablate/ablate.py`, `ablate/compare.py` | Torch ablation chains T/E/A/B/AB/C/N (outputs `~/.cache/xolotl/ablate/`). |
| `prof/*.py`, `bench.sh` | rocprofv3 kernel-trace analysis of the dense half. |
| `dspark_ab.sh NAME N [args]` | One-shot plain vs DSpark run on the warm prompt (free greedy, N tokens), quality mode + the served read-path defaults. Env: `XENV` (e.g. `STRATA_DSPARK=<sidecar> STRATA_DSPARK_DEPTH=4 STRATA_DSPARK_CONF=0.6`), `ARENA` (86; 80 with DSpark), `PROMPT` (ids file). Log `runs/dspark/NAME.log`. |
| `dspark_cmp.py REF RUN..` | Token-stream match vs REF (first divergence), tok/s, file reads/token, tokens/pass, verify ms, arena extension. |
| `dspark_conf.py NAME..` | Stall-free DSpark numbers from `STRATA_DSPARK_CONFLOG` (`runs/dspark/NAME.conf`): KFD-stalled passes (> 1 s) and the first 8 dropped. |
| `prefill/` | Prefill / TTFT (docs/PREFILL.md): `turns.py NAME SPEC [args]` (serve turns over the line protocol, `PF_ENV`, `PF_NO_DSPARK`), `report.py NAME..` (per-turn table incl. pred-prefetch use), `ab_mmq.sh CFG [ROWS]` (17-row `--ppl` A/B of prompt-path variants: mmq, mqv, mq3, fkv, v2r) + `ab_cmp.py REF CFG..` (vs REF and the teacher), `routes.py`, `ktsum.py`, `oneshot_prof.sh`, `spec_*.json`. |
| `serve_ab.py NAME` | 3 fixed requests (prose / code / code at 0.7) against a running server; TTFT + decode from the monitor. |
| `prefetch/` | MTP-prefetch study scripts (`docs/MTP_PREFETCH.md`). KFD eviction sampler: `uma/evmon.sh OUT` (`/sys/class/kfd/kfd/proc/<pid>/stats_*/evicted_ms`). |
| `nvme/warm.sh NAME N [args]` | The warm speed bench (ab_speed.sh's `srun`, quality mode, arena 86 GiB) for N decode steps; extra engine env `XENV="A=1 B=2"`. Log `runs/nvme/NAME.log`. |
| `nvme/win.py [--from N] [--to N] LOG..` | Windowed tok/s, misses/token, stage ms and the `file-tier profile` line (DS4_TIER_PROF) of warm-bench logs (default window 1024..end). |
| `nvme/vadapt.sh NAME N "ENV" [args]` | verify_setup's engine run (its prompt/env/arena 80) for N greedy tokens with extra env/args (e.g. `--arena-adapt`) + a decode trace; checks the first 32 tokens against `verify_setup.ref`. |
| `nvme/tracecmp.py A B` | Bitwise losslessness check of two `STRATA_DEC_TRACE` traces (routing ids + f32 weights per layer, greedy argmax + logit per step). |
| `nvme/sim.py TRACE [--from N] [--policies ..] [--extra N] [--spec k:n,..]` | Offline arena-policy simulation (lru = the engine, opt = Belady bound, lfur:D, ...) and read-ahead model on a decode trace (`STRATA_DEC_TRACE`, + `STRATA_FILE_PRED=1` for predictions). |
| `nvme/nvbench.cpp`, `nvme/ab.sh`, `nvme/dstat.sh` | Read-only NVMe microbench of expert-blob reads (O_DIRECT pread pools / spawned threads / raw io_uring, chunk sizes, gaps, contiguous; drops the GGUF's page cache first), interleaved A/B driver, per-second nvme0n1 read MB/s + latency + temperature. Build: `g++ -O2 -std=c++17 -pthread -o nvbench nvbench.cpp -ldl` in a dir under `~/.cache/xolotl/`. |

Prompt files used throughout: `~/.cache/xolotl/runs/first-170/{chat,long,agent4k,agent8k,code4k,code8k}.i32`,
forced decode tokens `runs/forced.ids`. Engine env switches for measuring: `STRATA_PPL_DUMP=<file>` (per-token NLL),
`STRATA_LM_DUMP=<file>` + `STRATA_LM_DUMP_POS=p,..` (per-layer state), `STRATA_DEC_PROF=1`, `STRATA_AR_PROF=1`,
`DS4_TIER_PROF=1` (now also a `file-tier profile` line: read wall per blob, blobs/call), `STRATA_FORCE_IDS=<file>`,
`STRATA_DEC_TRACE=<file>` (decode routing + greedy outputs, binary; see `nvme/sim.py`).
