#!/bin/bash
# dstat.sh SECONDS - per-second nvme0n1 stats: read MB/s, reads/s, avg read latency (ms), write MB/s, in-flight, temp
N=${1:-30}
prev=($(awk '$3=="nvme0n1"' /proc/diskstats))
for i in $(seq 1 $N); do
  sleep 1
  cur=($(awk '$3=="nvme0n1"' /proc/diskstats))
  r=$(( ${cur[3]} - ${prev[3]} )); rs=$(( ${cur[5]} - ${prev[5]} )); rt=$(( ${cur[6]} - ${prev[6]} ))
  w=$(( ${cur[7]} - ${prev[7]} )); ws=$(( ${cur[9]} - ${prev[9]} )); busy=$(( ${cur[12]} - ${prev[12]} ))
  lat=$(awk -v t=$rt -v n=$r 'BEGIN{ if(n>0) printf "%.2f", t/n; else print "-"}')
  echo "$(date +%T) rMB/s $(( rs / 2048 )) r/s $r rlat_ms $lat wMB/s $(( ws / 2048 )) w/s $w busy% $(( busy / 10 )) inflight ${cur[11]} temp $(( $(cat /sys/class/nvme/nvme0/hwmon0/temp1_input) / 1000 ))"
  prev=("${cur[@]}")
done
