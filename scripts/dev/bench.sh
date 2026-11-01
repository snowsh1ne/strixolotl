#!/bin/bash
# bench.sh NAME "extra env exports" [extra engine args]  - the exact decode benchmark via gen.sh
N=$1; X=$2; shift 2
while pgrep -x ds41_generate >/dev/null; do echo "engine running, abort"; exit 1; done
M=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo); [ "$M" -lt 112 ] && { echo "MemAvailable $M GiB, abort"; exit 1; }
. ~/models/xolotl/V41-1017.paths
ENVX="export STRATA_DEC_PROF=1 STRATA_PF_NO_DUMMY=1 STRATA_FORCE_IDS=$HOME/.cache/xolotl/runs/forced.ids; $X" TMO=${TMO:-600} TAILN=60 \
  $(dirname "$(readlink -f "$0")")/gen.sh $N --ids-file ~/.cache/xolotl/runs/first-170/long.i32 -n ${NTOK:-400} --uma-reserve-gib 14 --uma-cache-gib 1 --no-chunk-mmq --backend cuda --experts gpu --engram "$XOLOTL_DSV41_ENGRAM" --kv-fp4 --arena-adapt --route-bias 0 --route-bias-file 0 --pf-b 0 "$@" | grep -E "rc=|WATCH|ar-prof|decode step|tok/s" | tail -4
