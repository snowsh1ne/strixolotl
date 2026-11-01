#!/bin/bash
# file-penalty sweep on GSQ-200: ppl (--bias-prompt) on 8K code/agent + 200-token decode on the chat prompt
S=$(dirname "$0"); R=~/.cache/xolotl/runs; export PATHS=~/models/xolotl/V41-1017-200.paths; . $PATHS
for fp in 0 0.1 0.3 0.6; do
  for d in code agent; do
    $S/gen.sh sw-$fp-$d --ids-file $R/first-170/${d}8k.i32 -n 1 --ppl --ctx 16384 --no-chunk-mmq --bias-prompt 1 --backend cuda --experts gpu --engram "$XOLOTL_DSV41_ENGRAM" --kv-fp4 --route-bias 0 --route-bias-file $fp > /dev/null
    echo "fp $fp $d: $(grep -h '^PPL' $R/sw-$fp-$d.log)"
  done
  $S/gen.sh sw-$fp-dec --ids-file $R/first-170/chat.i32 -n 200 --backend cuda --experts gpu --engram "$XOLOTL_DSV41_ENGRAM" --kv-fp4 --arena-adapt --route-bias 0 --route-bias-file $fp > /dev/null
  echo "fp $fp decode: $(grep -h '^decode:' $R/sw-$fp-dec.log | cut -c1-110)"
done
