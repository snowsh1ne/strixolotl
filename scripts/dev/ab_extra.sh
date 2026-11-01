#!/bin/bash
# ab_extra.sh - after ab_v1v2.sh, on GSQ-217 with the v2 switches (runs/ab, same rows, teacher refs and speed prompts):
#   E0 v2rep  v2 again on two rows (determinism: dumps must be identical)
#   E1 p03 / E2 p06  speed-mode file penalty --route-bias-file 0.3 / 0.6 with --bias-prompt 1 (the decode bias applied
#                    to the prompt: the engine's quality proxy), every held-out row
#   E3 q6     STRATA_DENSE_REQUANT=q6_K (dense half Q8_0 -> q6_K), 9 rows
#   + the speed bench (6144-token prompt + 1000 forced tokens) for p03, p06, q6 on agentic_v2 and code_v2
# Resumable like ab_v1v2.sh.  Report: python3 scripts/dev/ab_report.py (extras section).
S=$(dirname "$(readlink -f "$0")"); R=~/.cache/xolotl/runs; A=$R/ab
M=~/models/xolotl; . $M/V41-1017.paths
log() { echo "$(date '+%F %T') $*" | tee -a $A/ab.log; }
V2="STRATA_DENSE_REQUANT= STRATA_PF_NO_DUMMY=1 STRATA_UMA_ZEROCOPY=1 STRATA_DBG_OQUANT=1 STRATA_IDX_BF16=1 STRATA_DBG_PREC32=1 STRATA_DENSE_F16=output.weight"
declare -A ENV_OF=([v2rep]="$V2" [p03]="$V2" [p06]="$V2" [q6]="$V2 STRATA_DENSE_REQUANT=q6_K")
declare -A ARG_OF=([v2rep]="" [p03]="--route-bias-file 0.3 --bias-prompt 1" [p06]="--route-bias-file 0.6 --bias-prompt 1" [q6]="")
BASE="--ctx 16384 --prompt-batch 8192 --uma-reserve-gib 24 --uma-cache-gib 0 --pf-b 0 --no-chunk-mmq --backend cuda --experts gpu --engram $XOLOTL_DSV41_ENGRAM --kv-fp4 --route-bias 0"
pgrep -x ds41_generate >/dev/null && { log "extra: an engine is running - stop the server first"; exit 1; }
log "extra start"
ALL=$(python3 -c "import json; print(' '.join(r['file'] for r in json.load(open('$A/rows.json'))))")
Q9="v1_r06 v1_r12 v1_r18 v2_r14 v2_r16 v2_r21 v2_r28 v2_r35 v2_r44"
qrun() { local cfg=$1 f=$2 out=$A/nll-$1-$2.txt
  [ -s $out ] && [ $(wc -l < $out) -ge 8191 ] && return
  # the last --route-bias-file wins: BASE has none, ARG_OF adds the penalty (exact routing = 0 by default)
  PATHS=$M/V41-1017.paths MW_MIN=5120 TMO=1500 TAILN=1 ENVX="export ${ENV_OF[$cfg]} STRATA_PPL_DUMP=$out;" \
    $S/gen.sh ab/q-$cfg-$f --ids-file $A/rows/$f.i32 -n 1 --ppl --arena-gib 82 $BASE --route-bias-file 0 ${ARG_OF[$cfg]} > /dev/null
  log "quality $cfg $f: $(grep -h '^PPL' $A/q-$cfg-$f.log | tail -1) ($(wc -l < $out 2>/dev/null || echo 0) positions)"; }
for f in v2_r14 v1_r06; do qrun v2rep $f; done
for f in $ALL; do qrun p03 $f; done
for f in $ALL; do qrun p06 $f; done
for f in $Q9; do qrun q6 $f; done
for spec in agentic_v2:p03 agentic_v2:p06 agentic_v2:q6 code_v2:p03 code_v2:p06 code_v2:q6; do
  IFS=: read -r p cfg <<< "$spec"; n=s-$cfg-$p
  grep -q 'decode step ms/token' $A/$n.log 2>/dev/null && continue
  PATHS=$M/V41-1017.paths MW_MIN=5120 TMO=1500 TAILN=1 ENVX="export ${ENV_OF[$cfg]} STRATA_DEC_PROF=1 STRATA_FORCE_IDS=$A/speed/$p.force;" \
    $S/gen.sh ab/$n --ids-file $A/speed/$p.i32 -n 1000 --arena-adapt --arena-gib 86 $BASE --route-bias-file 0 ${ARG_OF[$cfg]} > /dev/null
  log "speed $n: $(grep -h '^decode:' $A/$n.log | tail -1 | cut -c1-110)"
done
log "extra done"
