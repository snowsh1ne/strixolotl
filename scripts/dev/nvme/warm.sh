#!/bin/bash
# warm.sh NAME N [extra ds41_generate args] - the warm speed bench of ab_speed.sh (srun, quality mode = V2) for N
# decode steps: runs/ab/warm/prompt.i32 + STRATA_FORCE_IDS=force.ids, --arena-adapt --arena-gib 86, STRATA_DEC_PROF=1.
# Extra engine env: XENV="A=1 B=2".  Log runs/nvme/NAME.log (+ .mem).  Server must be stopped (gen.sh waits for memory).
S=$(cd "$(dirname "$0")/.." && pwd); R=~/.cache/xolotl/runs; W=$R/ab/warm; M=~/models/xolotl; . $M/V41-1017.paths
N=$1; NT=${2:-3000}; shift 2
mkdir -p $R/nvme
V2="STRATA_DENSE_REQUANT= STRATA_PF_NO_DUMMY=1 STRATA_UMA_ZEROCOPY=1 STRATA_DBG_OQUANT=1 STRATA_IDX_BF16=1 STRATA_DBG_PREC32=1 STRATA_DENSE_F16=output.weight"
BASE="--prompt-batch 8192 --uma-cache-gib 0 --pf-b 0 --no-chunk-mmq --backend cuda --experts gpu --engram $XOLOTL_DSV41_ENGRAM --kv-fp4 --route-bias 0 --route-bias-file 0"
PATHS=$M/V41-1017.paths MW_MIN=5120 TMO=${TMO:-2700} TAILN=1 ENVX="export $V2 STRATA_DEC_PROF=1 STRATA_FORCE_IDS=$W/force.ids ${XENV:-};" \
  $S/gen.sh nvme/$N --ids-file $W/prompt.i32 -n $NT --ctx 20480 --arena-adapt --arena-gib 86 --uma-reserve-gib 24 $BASE "$@"
