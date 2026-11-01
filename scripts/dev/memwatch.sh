#!/bin/bash
# memwatch.sh LOG - MemAvailable + GTT used (MiB) + NVMe temperature (C) every second; kills ds41_generate below
# MW_MIN MiB available (default 3 GiB)
NV=$(grep -l '^nvme' /sys/class/hwmon/hwmon*/name 2>/dev/null | head -1); NV=${NV%/name}
while :; do
  ma=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
  gtt=$(( $(cat /sys/class/drm/card*/device/mem_info_gtt_used | head -1) >> 20 ))
  echo "$(date +%T) $ma $gtt $(( $(cat $NV/temp1_input 2>/dev/null || echo 0) / 1000 ))" >> "$1"
  if [ "$ma" -lt "${MW_MIN:-3072}" ]; then echo "$(date +%T) WATCHDOG: MemAvailable ${ma} MiB - killing ds41_generate" >> "$1"; pkill -KILL -x ds41_generate; fi
  sleep 1
done
