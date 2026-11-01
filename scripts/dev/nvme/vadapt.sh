#!/bin/bash
# vadapt.sh NAME N "ENV" [extra args] - scripts/verify_setup's engine run (its prompt, quality-mode env, arena 80 GiB,
# greedy) for N tokens with extra engine env and args (e.g. --arena-adapt) and STRATA_DEC_TRACE; prints the output
# tokens' agreement with scripts/verify_setup.ref (the first 32).  Log/trace in ~/.cache/xolotl/runs/nvme/NAME.*.
# Server must be stopped.  Compare two runs' traces with tracecmp.py.
REPO=~/projects/Xolotl_StrixHalo; R=~/.cache/xolotl/runs/nvme; mkdir -p $R
. ~/models/xolotl/V41-1017.paths
N=$1; NT=$2; X=$3; shift 3
REF=$REPO/scripts/verify_setup.ref
ids=$(sed -n 's/^prompt: //p' $REF | tr ' ' ',')
want=$(sed -n 's/^output: //p' $REF)
for i in $(seq 1 120); do [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -ge 110 ] && ! pgrep -x ds41_generate >/dev/null && break; sleep 2; done
cd $REPO
timeout 1500 toolbox run -c xolotl-strix bash -c ". /opt/xolotl/rocm.env; env STRATA_DENSE_REQUANT= STRATA_UMA_ZEROCOPY=1 STRATA_PF_NO_DUMMY=1 \
  STRATA_DBG_OQUANT=1 STRATA_IDX_BF16=1 STRATA_DBG_PREC32=1 STRATA_DENSE_F16=output.weight \
  STRATA_HIPBLASLT_TUNING=$REPO/xolotl/tools/hip/gfx1151-hipblaslt-100401.txt STRATA_DEC_TRACE=$R/$N.trace $X \
  xolotl/tools/v41/serve/ds41-engine-strix.sh -m $XOLOTL_DSV41_GGUF --rank $XOLOTL_DSV41_RANK --engram $XOLOTL_DSV41_ENGRAM \
  --backend cuda --experts gpu --uma on --pcie 1 --tier-batch-pcie 1 \
  --arena-gib 80 --uma-reserve-gib 24 --uma-cache-gib 0 --pf-b 0 --no-chunk-mmq --kv-fp4 \
  --route-bias 0 --route-bias-file 0 --prompt-batch 8192 --ctx 8192 --temp 0 -n $NT --ids $ids $*" > $R/$N.out 2> $R/$N.log < /dev/null
pkill -KILL -x ds41_generate 2>/dev/null
got=$(grep -E '^[0-9]+$' $R/$N.out | head -32 | tr '\n' ' ' | sed 's/ $//')
[ "$got" = "$want" ] && echo "$N: first 32 tokens = verify_setup.ref (PASS)" || echo "$N: first 32 tokens DIFFER from verify_setup.ref"
grep -h "^decode:\|arena ext" $R/$N.log | cut -c1-160
