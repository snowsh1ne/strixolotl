#!/usr/bin/env bash
# ds41-engine-strix.sh - the DeepSeek-V4.1 engine as Strata's server starts it (`<exe> --serve <args>`) on Strix Halo
# (strix-port; the Strix counterpart of ds41-engine.sh).  The 4090 box's wrapper adds memguard's protections - a
# ComfyUI check, a shared GPU lock and an 84 GiB systemd-run scope - which are that machine's; in a toolbox there is
# no user systemd to ask for a scope, and on the APU the engine's own MemAvailable floor (--uma, --mem-floor-gib)
# is the guard.  The defaults below are the upstream wrapper's, except where the unified memory changes the answer.
set -u
cd "$(dirname "$0")/../../.."
export GGML_NO_BACKTRACE=1
# a GPU fault otherwise writes a core dump of GPU memory into the working directory
export HSA_DISABLE_COREDUMP_ON_EXCEPTION=1
# 4 KiB pages pin the arena fastest (upstream: 1.2 GB/s vs 0.16 GB/s with THP)
export STRATA_NO_LARGEPAGES="${STRATA_NO_LARGEPAGES:-1}"
# no D3 prefetch on UMA: with --uma-cache-gib 0 there is no prefetch slot, so predict_multi ran for nothing on every
# verify layer (docs/UMA_DEEP_DIVE.md R3: -3.7 ms per DSpark pass, same tokens)
export STRATA_D3_PF="${STRATA_D3_PF:-0}"
# exact short prompt turns, as upstream (ds41-engine.sh)
export STRATA_SERVE_STEP_MAX="${STRATA_SERVE_STEP_MAX:-96}"
export STRATA_SERVE_PASS_BIAS="${STRATA_SERVE_PASS_BIAS:-0}"
# DSpark speculative decode: OFF for the first bring-up (scripts/start_xolotl_dsv41 --dspark sets the sidecar).
# Upstream measured +30% decode with it; it adds ~2.9 GB of device weights and the DSpark kernels on HIP are untested.
if [ -n "${STRATA_SERVE_DSPARK:-}" ]; then
    export STRATA_DSPARK_KEEP="${STRATA_DSPARK_KEEP:-$PWD/tools/v41/serve/dspark_expert_counts.txt:64}"
    export STRATA_DSPARK_DEPTH="${STRATA_DSPARK_DEPTH:-4}"
    export STRATA_DSPARK_CONF="${STRATA_DSPARK_CONF:-0.6}"
    export STRATA_SERVE_LOOKUP="${STRATA_SERVE_LOOKUP:-14}"
    export STRATA_SERVE_DSPARK_EVICT="${STRATA_SERVE_DSPARK_EVICT:-4096}"
else
    unset STRATA_SERVE_DSPARK
fi
# the dense half re-quantized Q8_0 -> Q6_K at load (upstream default: decode +11-35%, KL +1%)
export STRATA_DENSE_REQUANT="${STRATA_DENSE_REQUANT-q6_K}"
[ -z "$STRATA_DENSE_REQUANT" ] && unset STRATA_DENSE_REQUANT
# UMA: the chunk prestage region is a DRAM -> DRAM copy of arena experts here, and it takes memory the arena could
# hold: off (upstream 1.5 GiB on the 24 GB card)
export STRATA_PRESTAGE_GIB="${STRATA_PRESTAGE_GIB:-0}"
# ggml's HIP graphs off: vincalkr/Strata-DSV4.1 measured +6.6% decode on HIP without them (HANDOFF.md S6 research)
export GGML_CUDA_DISABLE_GRAPHS="${GGML_CUDA_DISABLE_GRAPHS:-1}"
# strix-port: ggml disables graphs when the variable is SET (any value): GGML_CUDA_DISABLE_GRAPHS=0 = graphs on
[ "$GGML_CUDA_DISABLE_GRAPHS" = 0 ] && unset GGML_CUDA_DISABLE_GRAPHS
[ -n "${STRATA_PROMPT_BATCH:-}" ] && set -- "$@" --prompt-batch "$STRATA_PROMPT_BATCH"
BUILD_DIR="${STRATA_BUILD_DIR:-${XOLOTL_BUILD_DIR:-$HOME/.cache/xolotl/build-strix}}"
[ -x "$BUILD_DIR/ds41_generate" ] || { echo "ds41-engine-strix: no $BUILD_DIR/ds41_generate (scripts/build_xolotl)" >&2; exit 3; }
exec "$BUILD_DIR/ds41_generate" "$@"
