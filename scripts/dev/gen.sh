#!/bin/bash
# gen.sh NAME [extra ds41_generate args...]  - one-shot run in xolotl-strix, log in ~/.cache/xolotl/runs/NAME.log
# memwatch kills ds41_generate below 3 GiB MemAvailable; on timeout the engine itself is killed (not just the toolbox call)
N=$1; shift; R=~/.cache/xolotl/runs; . ${PATHS:-$HOME/models/xolotl/V41-1017.paths}
S=$(dirname "$(readlink -f "$0")")
cd ~/projects/Xolotl_StrixHalo
# wait until the previous engine's memory is back (a killed run frees ~90 GiB of pinned arena over seconds)
for i in $(seq 1 120); do [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -ge "${MW_START:-110}" ] && ! pgrep -x ds41_generate >/dev/null && break; sleep 2; done
rm -f "$R/$N.mem"; MW_MIN=${MW_MIN:-3072} "$S/memwatch.sh" "$R/$N.mem" & MW=$!
timeout ${TMO:-1500} toolbox run -c xolotl-strix bash -c "${ENVX:-} . /opt/xolotl/rocm.env; export STRATA_HIPBLASLT_TUNING=\$PWD/xolotl/tools/hip/gfx1151-hipblaslt-100401.txt; xolotl/tools/v41/serve/ds41-engine-strix.sh -m '$XOLOTL_DSV41_GGUF' --rank '$XOLOTL_DSV41_RANK' --uma on --pcie 1 --tier-batch-pcie 1 --ctx 8192 --temp 0 $*" > $R/$N.log 2>&1
rc=$?
pkill -KILL -x ds41_generate 2>/dev/null; kill $MW 2>/dev/null
echo "rc=$rc"; grep -h WATCHDOG "$R/$N.mem"; tail -${TAILN:-12} $R/$N.log
