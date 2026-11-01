#!/bin/bash
# dspark_ab.sh NAME N [extra ds41_generate args] - plain vs DSpark A/B on the warm bench prompt (runs/ab/warm/prompt.i32,
# 14.8K agent prompt), FREE greedy decode of N tokens (DSpark refuses STRATA_FORCE_IDS; the first turn ends at ~1294).
# Same config as nvme/warm.sh (quality mode V2, --arena-adapt --arena-gib 86) plus the served read-path defaults
# (STRATA_FILE_FAST=1 STRATA_ARENA_LFU=0.995 STRATA_ARENA_EXT_GIB=10).  DSpark arm: XENV="STRATA_DSPARK=<sidecar>
# STRATA_DSPARK_DEPTH=4 ...". PROMPT=<ids file> overrides the prompt, STRATA_BUILD_DIR the binary.  Log runs/dspark/NAME.log (+ .mem), the token stream = the log's bare-number lines.
S=$(cd "$(dirname "$0")" && pwd); R=~/.cache/xolotl/runs; W=$R/ab/warm; M=~/models/xolotl; . $M/V41-1017.paths
N=$1; NT=${2:-1400}; shift 2
mkdir -p $R/dspark
V2="STRATA_DENSE_REQUANT= STRATA_PF_NO_DUMMY=1 STRATA_UMA_ZEROCOPY=1 STRATA_DBG_OQUANT=1 STRATA_IDX_BF16=1 STRATA_DBG_PREC32=1 STRATA_DENSE_F16=output.weight"
RP="STRATA_FILE_FAST=1 STRATA_ARENA_LFU=0.995 STRATA_ARENA_EXT_GIB=10"
BASE="--prompt-batch 8192 --uma-cache-gib 0 --pf-b 0 --no-chunk-mmq --backend cuda --experts gpu --engram $XOLOTL_DSV41_ENGRAM --kv-fp4 --route-bias 0 --route-bias-file 0"
PATHS=$M/V41-1017.paths MW_MIN=${MW_MIN:-4096} TMO=${TMO:-2700} TAILN=${TAILN:-6} ENVX="export $V2 $RP STRATA_DEC_PROF=1 ${XENV:-};" \
  $S/gen.sh dspark/$N --ids-file ${PROMPT:-$W/prompt.i32} -n $NT --ctx 20480 --arena-adapt --arena-gib ${ARENA:-86} --uma-reserve-gib 24 $BASE "$@"
