#!/bin/bash
# ab_v1v2.sh - quality mode v1 (GSQ-200, quality-mode flags of 2026-10-21 morning: no fidelity switches) vs v2
# (GSQ-217 + the four fidelity switches), on the held-out rows of both calibration chains, plus a speed A/B.
#   quality: every row (ab_prep.py) through --ppl at a fixed --arena-gib 82, per-token dumps runs/ab/nll-<cfg>-<row>.txt,
#            teacher references runs/ab/nll-T-<row>.txt (teacher_nll8k.py from each chain's h/L40)
#   speed:   6144-token prompt + 1000 forced real tokens (STRATA_FORCE_IDS) at --arena-gib 86 with --arena-adapt and
#            STRATA_DEC_PROF; configs v1, v2 and m (GSQ-200 with the v2 switches: separates file size from the switches)
# The server must be stopped first (one GPU user).  Resumable: finished runs (dump / log with a decode line) are skipped.
# Log: runs/ab/ab.log.  Report: python3 scripts/dev/ab_report.py
S=$(dirname "$(readlink -f "$0")"); R=~/.cache/xolotl/runs; A=$R/ab; mkdir -p $A
REPO=~/projects/Xolotl_StrixHalo; M=~/models/xolotl; . $M/V41-1017.paths
log() { echo "$(date '+%F %T') $*" | tee -a $A/ab.log; }
Q="STRATA_DENSE_REQUANT= STRATA_PF_NO_DUMMY=1 STRATA_UMA_ZEROCOPY=1"
FID="STRATA_DBG_OQUANT=1 STRATA_IDX_BF16=1 STRATA_DBG_PREC32=1 STRATA_DENSE_F16=output.weight"
declare -A PATHS_OF=([v1]=$M/V41-1017-200.paths [v2]=$M/V41-1017.paths [m]=$M/V41-1017-200.paths)
declare -A ENV_OF=([v1]="$Q" [v2]="$Q $FID" [m]="$Q $FID")
BASE="--ctx 16384 --prompt-batch 8192 --uma-reserve-gib 24 --uma-cache-gib 0 --pf-b 0 --no-chunk-mmq --backend cuda --experts gpu --engram $XOLOTL_DSV41_ENGRAM --kv-fp4 --route-bias 0 --route-bias-file 0"
pgrep -x ds41_generate >/dev/null && { log "an engine is running - stop the server first"; exit 1; }
log "start"

# 1. inputs + teacher references (xolotl-quant, torch on the GPU)
[ -f $A/rows.json ] || toolbox run -c xolotl-quant bash -c "cd ~ && python3 $S/ab_prep.py $A" >> $A/ab.log 2>&1
for c in v1 v2; do
  W=$M/v41-work/calib-1017; [ $c = v2 ] && W=$M/v41-work/calib-1017-v2
  jobs=$(python3 -c "import json; print(' '.join(f\"{r['row']} $A/nll-T-{r['file']}.txt\" for r in json.load(open('$A/rows.json')) if r['chain']=='$c' and not __import__('os').path.exists('$A/nll-T-'+r['file']+'.txt')))")
  [ -n "$jobs" ] && toolbox run -c xolotl-quant bash -c "cd ~ && . $REPO/scripts/v41_1017.env && export V41_WORK=$W && python3 $S/teacher_nll8k.py $jobs" >> $A/ab.log 2>&1
done
log "teacher refs: $(ls $A/nll-T-*.txt 2>/dev/null | wc -l)"

# 2. quality runs (v1 and v2 alternate per row)
for f in $(python3 -c "import json; print(' '.join(r['file'] for r in json.load(open('$A/rows.json'))))"); do
  for cfg in v1 v2; do
    out=$A/nll-$cfg-$f.txt; [ -s $out ] && [ $(wc -l < $out) -ge 8191 ] && continue
    PATHS=${PATHS_OF[$cfg]} MW_MIN=5120 TMO=1500 TAILN=1 ENVX="export ${ENV_OF[$cfg]} STRATA_PPL_DUMP=$out;" \
      $S/gen.sh ab/q-$cfg-$f --ids-file $A/rows/$f.i32 -n 1 --ppl --arena-gib 82 $BASE > /dev/null
    log "quality $cfg $f: $(grep -h '^PPL' $A/q-$cfg-$f.log | tail -1) ($(wc -l < $out 2>/dev/null || echo 0) positions)"
  done
done

# 3. speed runs
for spec in agentic_v2:v1 agentic_v2:v2 agentic_v2:m code_v2:v1 code_v2:v2 code_v2:m agentic_v1:v1 agentic_v1:v2 agentic_v1:m \
            agentic_v2:v2:b agentic_v2:v1:b; do
  IFS=: read -r p cfg rep <<< "$spec"; n=s-$cfg-$p${rep:+-$rep}
  grep -q 'decode step ms/token' $A/$n.log 2>/dev/null && continue
  PATHS=${PATHS_OF[$cfg]} MW_MIN=5120 TMO=1500 TAILN=1 ENVX="export ${ENV_OF[$cfg]} STRATA_DEC_PROF=1 STRATA_FORCE_IDS=$A/speed/$p.force;" \
    $S/gen.sh ab/$n --ids-file $A/speed/$p.i32 -n 1000 --arena-adapt --arena-gib 86 $BASE > /dev/null
  log "speed $n: $(grep -h '^decode:' $A/$n.log | tail -1 | cut -c1-110)"
done
log "done"
