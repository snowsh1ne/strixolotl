#!/bin/bash
# ab.sh REPS "cfg1" "cfg2" ... : interleaved short nvbench runs, prints the median of the means per config
R=$1; shift   # run from the directory holding the built ./nvbench
declare -A acc
for i in $(seq 1 $R); do
  k=0
  for c in "$@"; do
    m=$(./nvbench $c -r $((i*100+k)) | sed 's/.*mean \([0-9.]*\) ms.*/\1/'); acc[$k]="${acc[$k]} $m"; k=$((k+1))
  done
done
k=0
for c in "$@"; do
  echo "$c | means:${acc[$k]} | median $(echo ${acc[$k]} | tr ' ' '\n' | sort -n | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}')"
  k=$((k+1))
done
