#!/bin/bash
# oneshot_prof.sh IDS OUTDIR - one-shot prompt (-n 1) under rocprofv3 --kernel-trace (docs/PREFILL.md dense profile):
# the served quality-mode env + read path, plain (no DSpark), under the GPU lock.  The serve loop leaves by _Exit, which
# skips rocprofv3's finalization, so the profile is a one-shot run.  Log OUTDIR.log; summary: ktsum.py OUTDIR
IDS=$1; OUT=$2; . ~/models/xolotl/V41-1017.paths; X=~/projects/Xolotl_StrixHalo/xolotl
E="STRATA_DENSE_REQUANT= STRATA_UMA_ZEROCOPY=1 STRATA_PF_NO_DUMMY=1 STRATA_DBG_OQUANT=1 STRATA_IDX_BF16=1 STRATA_DBG_PREC32=1 STRATA_DENSE_F16=output.weight STRATA_FILE_FAST=1 STRATA_ARENA_LFU=0.995 STRATA_ARENA_EXT_GIB=10 STRATA_HIPBLASLT_TUNING=$X/tools/hip/gfx1151-hipblaslt-100401.txt STRATA_BUILD_DIR=$HOME/.cache/xolotl/runs/prefill/pbin PF_PROF_OUT=$OUT STRATA_DENSE_PROF=1 MIMO_CHUNK_PROF=1"
flock ~/.cache/xolotl/gpu.lock toolbox run -c xolotl-strix bash -c ". /opt/xolotl/rocm.env; export $E; $X/tools/v41/serve/ds41-engine-strix.sh -m '$XOLOTL_DSV41_GGUF' --rank '$XOLOTL_DSV41_RANK' --engram '$XOLOTL_DSV41_ENGRAM' --backend cuda --experts gpu --uma on --pcie 1 --tier-batch-pcie 1 --ctx 8192 --temp 0 --ids-file '$IDS' -n 1 --arena-adapt --uma-reserve-gib 24 --prompt-batch 8192 --uma-cache-gib 0 --pf-b 0 --no-chunk-mmq --kv-fp4 --route-bias 0 --route-bias-file 0" > $OUT.log 2>&1
echo "rc=$?"
