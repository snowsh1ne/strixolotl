# scripts/lib/common.sh - shared paths and checks for the Xolotl Strix Halo scripts (sourced, not run).
#
# Every path can be overridden from the environment:
#   XOLOTL_MODELS     where models live            (default ~/models/xolotl; ext4/btrfs/xfs, never exFAT or NFS)
#   XOLOTL_CACHE      llama.cpp checkout + build   (default ~/.cache/xolotl; local disk: the repo may be on NFS)
#   XOLOTL_BUILD_DIR  the CMake build folder       (default $XOLOTL_CACHE/build-strix)
#   XOLOTL_STATE      generated configs and logs   (default ~/.local/state/xolotl)

set -euo pipefail

XOLOTL_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
XOLOTL_SRC="$XOLOTL_REPO/xolotl"
XOLOTL_MODELS="${XOLOTL_MODELS:-$HOME/models/xolotl}"
XOLOTL_CACHE="${XOLOTL_CACHE:-$HOME/.cache/xolotl}"
XOLOTL_BUILD_DIR="${XOLOTL_BUILD_DIR:-$XOLOTL_CACHE/build-strix}"
XOLOTL_LLAMA_DIR="${XOLOTL_LLAMA_DIR:-$XOLOTL_CACHE/llama.cpp}"
XOLOTL_STATE="${XOLOTL_STATE:-${XDG_STATE_HOME:-$HOME/.local/state}/xolotl}"
XOLOTL_GFX="${XOLOTL_GFX:-gfx1151}"
# the llama.cpp commit Xolotl pins (xolotl/tools/xolotl/fetch-llama.sh, CMakeLists.txt FetchContent)
XOLOTL_LLAMA_PIN=3cf03257f219afbe7334045ff7c6a06ac68c627d

say()  { printf '\033[1;36m[xolotl]\033[0m %s\n' "$*" >&2; }
warn() { printf '\033[1;33m[xolotl] WARNING:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[xolotl] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

# The toolbox's ROCm (toolbox/Containerfile).  Outside the toolbox a ROCm in /opt/rocm is used if it has clang.
xolotl_rocm_env() {
    if [ -r /opt/xolotl/rocm.env ]; then
        # shellcheck disable=SC1091
        . /opt/xolotl/rocm.env
    else
        warn "not in the xolotl-strix toolbox (/opt/xolotl/rocm.env missing): trying a system ROCm in /opt/rocm"
        XOLOTL_ROCM_ROOT=/opt/rocm
        XOLOTL_HIP_CLANG="$(ls /opt/rocm/lib/llvm/bin/clang++ /opt/rocm/llvm/bin/clang++ 2>/dev/null | head -1 || true)"
        XOLOTL_ROCM_BITCODE="$(ls -d /opt/rocm/lib/llvm/amdgcn/bitcode /opt/rocm/amdgcn/bitcode 2>/dev/null | head -1 || true)"
        XOLOTL_ROCM_PREFIX="/opt/rocm;/opt/rocm/lib/llvm"
        XOLOTL_ROCM_LIBDIRS="/opt/rocm/lib"
        XOLOTL_ROCM_BINDIRS="/opt/rocm/bin"
        XOLOTL_HIPBLASLT_VERSION=""
        [ -x "$XOLOTL_HIP_CLANG" ] || die "no ROCm found: run this inside the toolbox (scripts/toolbox_create, then toolbox enter xolotl-strix)"
    fi
    export ROCM_PATH="$XOLOTL_ROCM_ROOT" HIP_PATH="$XOLOTL_ROCM_ROOT" HIP_PLATFORM=amd
    export PATH="${XOLOTL_ROCM_BINDIRS}:$PATH"
    # the toolbox registers the lib dirs with ldconfig; LD_LIBRARY_PATH as well for a ROCm outside it
    export LD_LIBRARY_PATH="${XOLOTL_ROCM_LIBDIRS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    # a GPU fault otherwise dumps GPU memory into the working directory (strixite deploy/strix-server.service)
    export HSA_DISABLE_COREDUMP_ON_EXCEPTION=1
}

xolotl_python() {
    if [ -x /opt/xolotl/venv/bin/python3 ]; then echo /opt/xolotl/venv/bin/python3; else command -v python3; fi
}

# The GPU must be visible (toolbox / compose pass /dev/kfd and /dev/dri) and be the Strix Halo iGPU.
xolotl_check_gpu() {
    [ -e /dev/kfd ] || die "/dev/kfd is missing: the container needs --device /dev/kfd --device /dev/dri"
    [ -r /dev/kfd ] && [ -w /dev/kfd ] || die "no access to /dev/kfd: add your user to the render and video groups (then log in again)"
    local found=""
    for n in /sys/class/kfd/kfd/topology/nodes/*/properties; do
        if grep -q "gfx_target_version 110501" "$n" 2>/dev/null; then found=1; fi
    done
    [ -n "$found" ] || warn "no gfx1151 (gfx_target_version 110501) in the KFD topology: this is not a Strix Halo GPU?"
    if [ -n "${HSA_OVERRIDE_GFX_VERSION:-}" ]; then
        warn "HSA_OVERRIDE_GFX_VERSION=$HSA_OVERRIDE_GFX_VERSION is set: a native gfx1151 build needs none (unsetting it)"
        unset HSA_OVERRIDE_GFX_VERSION
    fi
}

# GiB the GPU can allocate (GTT) and MemAvailable, for the start-up summary and the memory plan.
xolotl_gtt_gib() {
    local f
    for f in /sys/class/drm/card*/device/mem_info_gtt_total; do
        [ -r "$f" ] && { echo $(( $(cat "$f") / 1073741824 )); return; }
    done
    echo 0
}
xolotl_memavail_gib() { awk '/^MemAvailable:/ {printf "%d\n", $2 / 1048576}' /proc/meminfo; }

# Refuse exFAT / FAT / NFS for model files: upstream measured exFAT expert reads 5x slower (0.96 vs 5.1 GB/s,
# upstream's hand-off notes), and O_DIRECT reads over NFS are worse still.
xolotl_check_fs() {
    local dir="$1" fs
    mkdir -p "$dir"
    fs="$(stat -f -c %T "$dir" 2>/dev/null || echo unknown)"
    case "$fs" in
        exfat|vfat|msdos|fuseblk|nfs|nfs4|smb2|cifs)
            die "$dir is on $fs: put the models on a local ext4/btrfs/xfs NVMe (XOLOTL_MODELS=/path)";;
    esac
}

xolotl_free_gb() { df --output=avail -B1000000000 "$1" | tail -1 | tr -d ' '; }
