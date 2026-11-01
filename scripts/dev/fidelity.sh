#!/bin/bash
# fidelity battery: agent4k, quality mode (Q8_0 dense), S = 512, fixed arena; dumps nll-fid-<name>.txt
S=$(dirname "$0"); R=~/.cache/xolotl/runs; . ~/models/xolotl/V41-1017.paths
C="--ids-file $R/first-170/agent4k.i32 -n 1 --ppl --ctx 8192 --prompt-batch 8192 --arena-gib 82 --uma-reserve-gib 24 --uma-cache-gib 0 --pf-b 0 --no-chunk-mmq --backend cuda --experts gpu --engram $XOLOTL_DSV41_ENGRAM --kv-fp4 --route-bias 0 --route-bias-file 0"
Q="STRATA_DENSE_REQUANT= STRATA_PF_NO_DUMMY=1 STRATA_UMA_ZEROCOPY=1"
run() { local name=$1 extra_env=$2 extra_args=$3
  MW_MIN=5120 TMO=900 ENVX="export $Q $extra_env STRATA_PPL_DUMP=$R/nll-fid-$name.txt;" TAILN=2 $S/gen.sh fid-$name $C $extra_args > /dev/null
  echo "$name: $(grep -h '^PPL' $R/fid-$name.log || tail -1 $R/fid-$name.log)"; }
run s384a "" "--prefill-chunk 384"
run s384b "" "--prefill-chunk 384"
run base ""
run oq "STRATA_DBG_OQUANT=1"
run bf "STRATA_IDX_BF16=1"
run p32 "STRATA_DBG_PREC32=1"
run head "STRATA_DENSE_F16=output.weight"
run all "STRATA_DBG_OQUANT=1 STRATA_IDX_BF16=1 STRATA_DBG_PREC32=1 STRATA_DENSE_F16=output.weight"
cmp -s $R/nll-fid-s384a.txt $R/nll-fid-s384b.txt && echo "S=384 twice: IDENTICAL" || echo "S=384 twice: differ"
python3 $S/cmpref.py fid-base fid-oq fid-bf fid-p32 fid-head fid-all fid-s384a pc512
