#!/bin/bash
# ab_mmq.sh CFG [ROWS..] - the 17-row same-text quality A/B of ab_v1v2.sh (v2 = the served quality config: GSQ-217 + the
# fidelity switches, --ppl, --arena-gib 82, --prompt-batch 8192 = every 8K row in ONE batch) for prompt-path variants.
#   CFG mmq: --chunk-mmq (MMQ expert kernels for the prompt; docs/PREFILL.md #1)   CFG v2r: MMVQ again (the current
#   binary vs the 2026-10-22 v2 dumps: is the reference still bit-identical?)   CFG fkv: MMVQ with STRATA_FUSED_KV (#2)
#   CFG mq3: MMQ only for the pure Q3_K expert classes (STRATA_MMQ_Q3K_ONLY=1)
# Dumps runs/ab/nll-CFG-ROW.txt; compare with: python3 -B scripts/dev/prefill/ab_cmp.py v2 CFG.  Server stopped first.
S=$(dirname "$(readlink -f "$0")")/..; A=~/.cache/xolotl/runs/ab
. ~/models/xolotl/V41-1017.paths
CFG=$1; shift
Q="STRATA_DENSE_REQUANT= STRATA_PF_NO_DUMMY=1 STRATA_UMA_ZEROCOPY=1"
FID="STRATA_DBG_OQUANT=1 STRATA_IDX_BF16=1 STRATA_DBG_PREC32=1 STRATA_DENSE_F16=output.weight"
BASE="--ctx 16384 --prompt-batch 8192 --uma-reserve-gib 24 --uma-cache-gib 0 --pf-b 0 --no-chunk-mmq --backend cuda --experts gpu --engram $XOLOTL_DSV41_ENGRAM --kv-fp4 --route-bias 0 --route-bias-file 0"
case $CFG in
  mmq) X="--chunk-mmq"; E=""; ARENA=78 ;;   # MMQ buffers + the 4.2 GB --ppl logits do not fit next to 82 GiB (page thrash)
  mq3) X="--chunk-mmq"; E="STRATA_MMQ_Q3K_ONLY=1"; ARENA=78 ;;   # MMQ for the pure Q3_K classes only (diagnostic)
  mqv) X="--chunk-mmq"; E="STRATA_MMQ_ACT_MMVQ=1"; ARENA=78 ;;   # MMQ with the MMVQ path's activation values (diagnostic)
  v2r) X=""; E="STRATA_FUSED_KV=0" ;;
  fkv) X=""; E="STRATA_FUSED_KV=1" ;;
  *) X="${AB_ARGS:-}"; E="${AB_ENV:-}" ;;
esac
rows=${*:-$(python3 -c "import json; print(' '.join(r['file'] for r in json.load(open('$A/rows.json'))))")}
for f in $rows; do
  out=$A/nll-$CFG-$f.txt; [ -s $out ] && [ $(wc -l < $out) -ge 8191 ] && continue
  MW_MIN=5120 TMO=1500 TAILN=1 ENVX="export $Q $FID $E STRATA_PPL_DUMP=$out;" \
    flock ~/.cache/xolotl/gpu.lock $S/gen.sh ab/q-$CFG-$f --ids-file $A/rows/$f.i32 -n 1 --ppl --arena-gib ${ARENA:-82} $BASE $X > /dev/null
  echo "$(date '+%F %T') $CFG $f: $(grep -h '^PPL' $A/q-$CFG-$f.log | tail -1) ($(wc -l < $out 2>/dev/null || echo 0) positions) $(grep -hc 'fault\|Memory access' $A/q-$CFG-$f.log) faults"
done
