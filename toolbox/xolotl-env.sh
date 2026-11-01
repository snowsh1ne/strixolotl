# /etc/profile.d/xolotl-env.sh - the toolbox's ROCm environment for interactive shells (toolbox enter).
# The scripts under scripts/ source /opt/xolotl/rocm.env themselves and do not depend on this file.
if [ -r /opt/xolotl/rocm.env ]; then
    . /opt/xolotl/rocm.env
    export ROCM_PATH="$XOLOTL_ROCM_ROOT" HIP_PATH="$XOLOTL_ROCM_ROOT" HIP_PLATFORM=amd XOLOTL_TOOLBOX=1
    export HSA_DISABLE_COREDUMP_ON_EXCEPTION=1
    case ":$PATH:" in
        *":/opt/xolotl/venv/bin:"*) ;;
        *) export PATH="/opt/xolotl/venv/bin:${XOLOTL_ROCM_BINDIRS}:$PATH" ;;
    esac
fi
