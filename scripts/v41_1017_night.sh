#!/bin/bash
# scripts/v41_1017_night.sh - unattended continuation of the 1017 quantization after calibration (tools/v41/all_day.sh's
# steps 1b-4 for this box): waits for calibration layers 0-39, then
#   1. teacher head gate   v41_calib.py head -> calib/stats/head_L40.json   (gate: finite, ppl < 30, top-1 > 0.35)
#   2. tier map            v41_alloc.py --per-proj --budget-gib 170 -> calib/alloc_170.json (+ curve 142-217 in the log)
#   3. GGUF layout         v41_gguf.py --experts-in-place --no-engram into a btrfs no-compress dir (chattr +m)
#   4. GSQ quantize pass   quant_layers.sh, experts written in place, + its own head gate (quantized chain)
# Stops at the first failure or red gate.  Engram shard 2, sidecar, rank.bin and validation stay manual.
# Run inside xolotl-quant:   toolbox run -c xolotl-quant bash scripts/v41_1017_night.sh
# Log: $V41_WORK/night.log.  Stop: touch $V41_WORK/night.STOP (and $QWORK/STOP to stop the quant pass between layers).
set -u
cd "$(dirname "$(readlink -f "$0")")/.."; R=$(pwd)
. scripts/v41_1017.env
unset V41_NO_ENGRAM
W=$V41_WORK
BUDGET=${BUDGET_GIB:-200}
MAP=$W/alloc_$BUDGET.json
GDIR=$HOME/models/xolotl/V41-1017-GSQ; GGUF=$GDIR/V41-1017-GSQ-$BUDGET.gguf
QWORK=$HOME/models/xolotl/v41-work/quant-1017-$BUDGET
T=xolotl/tools/v41
export NEED_GIB=60 MAX_GIB=80
LOG=$W/night.log
log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
die() { log "FAILED: $*"; exit 1; }
chk() { [ -f "$W/night.STOP" ] && { log "stop requested - exiting"; exit 0; }; }
rm -f "$W/night.STOP"
log "night start (pid $$): budget $BUDGET GiB, map $MAP, gguf $GGUF, qwork $QWORK"

# 0. wait for calibration ---------------------------------------------------------------------------------------------
while pgrep -f '^bash xolotl/tools/v41/calib_layers.sh' >/dev/null; do chk; sleep 60; done
for L in $(seq 0 39); do [ -f "$W/stats/L$(printf %02d "$L").json" ] || die "calibration incomplete: stats/L$(printf %02d "$L").json missing ($(tail -1 "$W/calib.log"))"; done
log "calibration complete (40/40)"; chk

# 1. teacher head gate ------------------------------------------------------------------------------------------------
if [ ! -f "$W/stats/head_L40.json" ]; then
  until flock -n "$DS4_LOCK" true && [ "$(awk '/^MemAvailable:/{print int($2/1048576)}' /proc/meminfo)" -ge "$NEED_GIB" ]; do chk; sleep 30; done
  MEMGUARD_ALLOW_COMFY=0 bash $T/../ds4/memguard.sh "$MAX_GIB" "$NEED_GIB" -- nice -n 5 python3 $T/v41_calib.py head > "$W/logs/head.log" 2>&1 \
    || die "teacher head gate (logs/head.log: $(tail -1 "$W/logs/head.log"))"
fi
gate=$(python3 - "$W/stats/head_L40.json" <<'EOF'
import json, math, sys
d = json.load(open(sys.argv[1]))["by_domain"]
e = d.get("ALL/eval") or d["ALL/fit"]
ok = math.isfinite(e["ppl"]) and e["ppl"] < 30 and e["top1"] > 0.35
print(("GREEN" if ok else "RED") + f" ALL/eval ppl {e['ppl']:.3f} top1 {e['top1']:.3f}; fit ppl {d['ALL/fit']['ppl']:.3f}")
EOF
) || die "head gate unreadable"
log "teacher head gate: $gate"
case $gate in GREEN*) ;; *) die "teacher head gate red - not quantizing";; esac
chk

# 2. tier map ---------------------------------------------------------------------------------------------------------
if [ ! -f "$MAP" ]; then
  : > "$W/logs/alloc_curve.log"
  for B in 142.38 155 170 185 200 217; do
    echo "== $B GiB" >> "$W/logs/alloc_curve.log"
    python3 $T/v41_alloc.py --work "$W" --layers 0-39 --per-proj --budget-gib "$B" 2>&1 | sed -n '1,2p' >> "$W/logs/alloc_curve.log"
  done
  python3 $T/v41_alloc.py --work "$W" --layers 0-39 --per-proj --budget-gib "$BUDGET" --out "$MAP" > "$W/logs/alloc_$BUDGET.log" 2>&1 \
    || die "v41_alloc.py (logs/alloc_$BUDGET.log)"
fi
log "tier map: $(sed -n '1,2p' "$W/logs/alloc_$BUDGET.log" | tr '\n' ' ')"
log "proxy curve: $(grep -E '^==|proxy' "$W/logs/alloc_curve.log" | sed -E 's/proxy error.*all-Q2_0 ([0-9.]+) +mixed ([0-9.]+).*/q2 \1 mixed \2/' | tr '\n' ' ')"
chk

# 3. GGUF layout ------------------------------------------------------------------------------------------------------
[ -d "$GDIR" ] || die "$GDIR missing: create it on the host with  mkdir $GDIR && chattr +m $GDIR  (the toolbox has no chattr)"
if command -v lsattr >/dev/null; then lsattr -d "$GDIR" | grep -q '^[^ ]*m' || die "$GDIR is not chattr +m (btrfs would compress the GGUF)"; fi
if [ ! -f "$GGUF.experts.json" ]; then
  free=$(df -BG --output=avail "$GDIR" | tail -1 | tr -dc 0-9)
  [ "$free" -ge 230 ] || die "only ${free} GiB free for the GGUF"
  (cd ~ && nice -n 10 python3 "$R/$T/v41_gguf.py" --out "$GGUF" --rtn-map "$MAP" --experts-in-place --no-engram --threads 8) \
    > "$W/logs/gguf_layout.log" 2>&1 || die "v41_gguf.py layout (logs/gguf_layout.log)"
fi
log "layout: $(ls -la "$GGUF" 2>&1)"; chk

# 4. GSQ quantize pass ------------------------------------------------------------------------------------------------
if [ ! -f "$QWORK/stats/head_L40.json" ]; then
  for att in 1 2 3; do
    bash $T/quant_layers.sh "$QWORK" 0 39 -- --map "$MAP" --gguf "$GGUF" >> "$LOG" 2>&1 && [ -f "$QWORK/stats/L39.json" ] && break
    chk; [ -f "$QWORK/STOP" ] && { log "quant pass stopped by request"; exit 0; }
    [ "$att" = 3 ] && die "quant pass, 3 attempts: $(tail -1 "$QWORK/quant.log")"
    log "quant pass exited early (attempt $att: $(tail -1 "$QWORK/quant.log")) - restarting"; sleep 5
  done
  [ -f "$QWORK/stats/head_L40.json" ] || die "quant pass ended without head_L40.json"
fi
log "DONE: $GGUF; head gates teacher $W/stats/head_L40.json vs GSQ $QWORK/stats/head_L40.json"
