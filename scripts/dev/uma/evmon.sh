#!/bin/bash
# evmon.sh OUT - every 0.25 s: t evicted_ms MemAvail_MiB VmSwap_kB gtt_MiB | vmstat: pswpin pswpout pgsteal_anon pgscan_anon pgmigrate_success compact_stall compact_daemon_migrate_scanned allocstall_movable pgscan_direct
G=/sys/class/drm/card0/device/mem_info_gtt_used
while :; do
  P=$(pgrep -x ds41_generate | head -1); e=-1; sw=-1
  if [ -n "$P" ]; then e=$(cat /sys/class/kfd/kfd/proc/$P/stats_*/evicted_ms 2>/dev/null | head -1); sw=$(awk '/VmSwap/{print $2}' /proc/$P/status 2>/dev/null); fi
  v=$(awk '{a[$1]=$2} END{printf "%s %s %s %s %s %s %s %s %s", a["pswpin"],a["pswpout"],a["pgsteal_anon"],a["pgscan_anon"],a["pgmigrate_success"],a["compact_stall"],a["compact_daemon_migrate_scanned"],a["allocstall_movable"],a["pgscan_direct"]}' /proc/vmstat)
  ma=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
  echo "$(date +%s.%N | cut -c1-14) ${e:--1} $ma ${sw:--1} $(( $(cat $G) >> 20 )) $v"
  sleep 0.25
done >> "$1"
