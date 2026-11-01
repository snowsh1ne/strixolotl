#!/bin/bash
# ab_speed.sh - the speed-mode sweep (speed mode, phase 1; paused: ROADMAP "Roadmap"), after ab_v1v2.sh / ab_extra.sh, on GSQ-217:
#   A. warm speed bench: runs/ab/warm/prompt.i32 (ab_speed_prep.py: ~15K tokens of a held-out agent session, ending
#      in <｜Assistant｜><think>) + 3000 forced tokens (warm/force.ids: quality mode's own greedy continuation, end of
#      turn dropped so no run stops early), --arena-adapt, STRATA_DEC_PROF.  The arena warms in decode only (the
#      one-shot prompt batch does not adapt it), so ab_report.py measures tokens >= 1024 from the windowed prof lines.
#      Fixed --arena-gib 86, except q6r18 / q6r16 (q6_K at --uma-reserve-gib 18 / 16, auto arena: the speed mode's
#      bigger arena + the prompt batch's peak memory; q6r16 only if q6r18's min MemAvailable left >= 8 GiB).
#   B. quality on the 17 held-out rows (--ppl, --arena-gib 82, as ab_extra.sh) in priority order.
#   C. the server again in quality mode.
# Configs (on GSQ-217 + the v2 fidelity switches): v2 = quality mode; q6 / q5 = STRATA_DENSE_REQUANT=q6_K / q5_K;
# q6pNNN = q6 + --route-bias-file 0.NNN --bias-prompt 1; q6f = q6 + STRATA_DENSE_REQUANT_F16=q8_0.
# Resumable (finished runs skipped).  Log runs/ab/ab.log.  Report: python3 scripts/dev/ab_report.py
S=$(dirname "$(readlink -f "$0")"); R=~/.cache/xolotl/runs; A=$R/ab; W=$A/warm
REPO=~/projects/Xolotl_StrixHalo; M=~/models/xolotl; . $M/V41-1017.paths
log() { echo "$(date '+%F %T') $*" | tee -a $A/ab.log; }
V2="STRATA_DENSE_REQUANT= STRATA_PF_NO_DUMMY=1 STRATA_UMA_ZEROCOPY=1 STRATA_DBG_OQUANT=1 STRATA_IDX_BF16=1 STRATA_DBG_PREC32=1 STRATA_DENSE_F16=output.weight"
Q6="$V2 STRATA_DENSE_REQUANT=q6_K"
declare -A ENV_OF=([v2]="$V2" [v2b]="$V2" [q6]="$Q6" [q5]="$V2 STRATA_DENSE_REQUANT=q5_K" [q6f]="$Q6 STRATA_DENSE_REQUANT_F16=q8_0"
                   [q6p005]="$Q6" [q6p010]="$Q6" [q6p015]="$Q6" [q6p020]="$Q6" [q6r18]="$Q6" [q6r16]="$Q6")
declare -A ARG_OF=([q6p005]="--route-bias-file 0.05 --bias-prompt 1" [q6p010]="--route-bias-file 0.1 --bias-prompt 1"
                   [q6p015]="--route-bias-file 0.15 --bias-prompt 1" [q6p020]="--route-bias-file 0.2 --bias-prompt 1")
BASE="--prompt-batch 8192 --uma-cache-gib 0 --pf-b 0 --no-chunk-mmq --backend cuda --experts gpu --engram $XOLOTL_DSV41_ENGRAM --kv-fp4 --route-bias 0 --route-bias-file 0"
pgrep -x ds41_generate >/dev/null && { log "speed: an engine is running - stop the server first"; exit 1; }
[ -s $W/prompt.i32 ] && [ -s $W/force.ids ] || { log "speed: no $W/prompt.i32 / force.ids (ab_speed_prep.py + the generation run)"; exit 1; }
log "speed start"

# A. warm speed bench
minfree() { awk '$2 ~ /^[0-9]+$/ {print $2}' $W/$1.mem 2>/dev/null | sort -n | head -1; }
srun() { local cfg=$1 n=s-$1 arena="--arena-gib 86 --uma-reserve-gib 24"
  grep -q '^decode:' $W/$n.log 2>/dev/null && return
  case $cfg in q6r18) arena="--uma-reserve-gib 18" ;; q6r16) arena="--uma-reserve-gib 16" ;; esac
  PATHS=$M/V41-1017.paths MW_MIN=5120 TMO=2700 TAILN=1 ENVX="export ${ENV_OF[$cfg]} STRATA_DEC_PROF=1 STRATA_FORCE_IDS=$W/force.ids;" \
    $S/gen.sh ab/warm/$n --ids-file $W/prompt.i32 -n 3000 --ctx 20480 --arena-adapt $arena $BASE ${ARG_OF[$cfg]:-} > /dev/null
  log "speed $n: $(grep -h '^decode:' $W/$n.log | tail -1 | cut -c1-110); min MemAvailable $(minfree $n) MiB$(grep -h WATCHDOG $W/$n.mem | head -1 | sed 's/^/; /')"; }
for cfg in v2 q6 q6p010 q6p015 q6p005 q6p020 q5 q6f q6r18; do srun $cfg; done
if [ -n "$(minfree s-q6r18)" ] && [ "$(minfree s-q6r18)" -ge 8192 ] && ! grep -q WATCHDOG $W/s-q6r18.mem; then srun q6r16
else log "speed: q6r16 skipped (q6r18 min MemAvailable $(minfree s-q6r18) MiB)"; fi
srun v2b

# B. quality, 17 held-out rows each (q6 had 9)
ALL=$(python3 -c "import json; print(' '.join(r['file'] for r in json.load(open('$A/rows.json'))))")
qrun() { local cfg=$1 f=$2 out=$A/nll-$1-$2.txt
  [ -s $out ] && [ $(wc -l < $out) -ge 8191 ] && return
  PATHS=$M/V41-1017.paths MW_MIN=5120 TMO=1500 TAILN=1 ENVX="export ${ENV_OF[$cfg]} STRATA_PPL_DUMP=$out;" \
    $S/gen.sh ab/q-$cfg-$f --ids-file $A/rows/$f.i32 -n 1 --ppl --ctx 16384 --arena-gib 82 --uma-reserve-gib 24 $BASE ${ARG_OF[$cfg]:-} > /dev/null
  log "quality $cfg $f: $(grep -h '^PPL' $A/q-$cfg-$f.log | tail -1) ($(wc -l < $out 2>/dev/null || echo 0) positions)"; }
for cfg in q6 q6p010 q6p015 q6p005 q5 q6f q6p020; do for f in $ALL; do qrun $cfg $f; done; done
log "speed done"
python3 $S/ab_report.py > $A/report-speed.txt 2>&1

# C. the server again (quality mode)
[ "${NO_SERVER:-0}" = 1 ] && exit 0
for i in $(seq 1 60); do [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -ge 110 ] && ! pgrep -x ds41_generate >/dev/null && break; sleep 5; done
cd $REPO && setsid nohup toolbox run -c xolotl-strix bash scripts/start_xolotl_dsv41 --model 1017 --host 0.0.0.0 --no-auth \
  > ~/.local/state/xolotl/logs/server-console.log 2>&1 < /dev/null & disown
log "speed: server started (quality mode)"
