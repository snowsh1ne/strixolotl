#!/usr/bin/env python3
"""rocm-discover.py - lay TheRock's ROCm wheels (installed into this venv) out as one ROCm for the Xolotl build.

strix-port: the image now installs AMD's ROCm 7.14 RPMs (one tree under /opt/rocm/core); `--root /opt/rocm/core`
skips the wheel search and checks/writes the same paths for that tree.  The wheel path below is kept as a fallback.

TheRock's Python wheels put ROCm in several site-packages folders (_rocm_sdk_core, _rocm_sdk_devel,
_rocm_sdk_libraries*, _rocm_sdk_device_<gfx>*).  The engine's CMake build and the dynamic loader each want plain
paths, so this script, run once at image build time:

  --link /opt/rocm        symlinks /opt/rocm to the SDK root (`rocm-sdk path --root`)
  --ldconf FILE           writes every lib folder of every _rocm_sdk_* package into an ld.so.conf.d file
  --env FILE              writes the paths the build needs (shell syntax, sourced by scripts/lib/common.sh):
                          XOLOTL_ROCM_ROOT, XOLOTL_HIP_CLANG, XOLOTL_ROCM_BITCODE, XOLOTL_ROCM_PREFIX (CMake ';' list),
                          XOLOTL_ROCM_LIBDIRS (':' list), XOLOTL_HIPBLASLT_VERSION (e.g. 100400)

It fails loudly when the compiler, the device bitcode or the gfx device libraries are missing, so a broken ROCm
install stops the image build instead of the engine build 20 minutes later.  Python stdlib only.
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path


def die(msg: str) -> None:
    print(f"rocm-discover: ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def sdk_root() -> Path:
    exe = Path(sys.executable).parent / "rocm-sdk"
    if exe.exists():
        r = subprocess.run([str(exe), "path", "--root"], capture_output=True, text=True)
        if r.returncode == 0 and r.stdout.strip():
            return Path(r.stdout.strip()).resolve()
        print(f"rocm-discover: `rocm-sdk path --root` failed ({r.stderr.strip()}); searching site-packages",
              file=sys.stderr)
    for name in ("_rocm_sdk_devel", "_rocm_sdk_core"):
        p = Path(sysconfig.get_paths()["purelib"]) / name
        if p.is_dir():
            return p.resolve()
    die("no ROCm SDK root: is `rocm[libraries,devel,device-<gfx>]` installed in this venv?")


def packages() -> list[Path]:
    seen, out = set(), []
    for key in ("purelib", "platlib"):
        sp = Path(sysconfig.get_paths()[key])
        for p in sorted(sp.glob("_rocm_sdk_*")):
            if p.is_dir() and p.resolve() not in seen:
                seen.add(p.resolve())
                out.append(p.resolve())
    return out


def first(paths) -> Path | None:
    return next((p for p in paths if p.exists()), None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gfx", default="gfx1151")
    ap.add_argument("--rocm-version", default="")
    ap.add_argument("--env", required=True)
    ap.add_argument("--ldconf")
    ap.add_argument("--link")
    ap.add_argument("--root", help="a plain ROCm tree (the RPM install): no wheel search")
    a = ap.parse_args()

    if a.root:
        root = Path(a.root).resolve()
        pkgs = [root]
    else:
        root = sdk_root()
        pkgs = packages()
    if not pkgs:
        die("no _rocm_sdk_* packages in site-packages")

    clang = first([root / "llvm" / "bin" / "clang++", root / "lib" / "llvm" / "bin" / "clang++"])
    if clang is None:
        die(f"no clang++ under {root}/llvm/bin or {root}/lib/llvm/bin (the devel wheel is missing?)")
    bitcode = first([root / "lib" / "llvm" / "amdgcn" / "bitcode", root / "llvm" / "amdgcn" / "bitcode",
                     root / "amdgcn" / "bitcode"])
    if bitcode is None:
        die(f"no amdgcn device bitcode under {root}")

    libdirs = []
    for base in [root, *pkgs]:
        for sub in ("lib", "lib64", "lib/llvm/lib", "llvm/lib", "lib/rocm_sysdeps/lib"):
            d = base / sub
            if d.is_dir() and any(d.glob("*.so*")) and str(d) not in libdirs:
                libdirs.append(str(d))
    prefixes = []
    for base in [root, *pkgs]:
        if (base / "lib" / "cmake").is_dir() or (base / "lib64" / "cmake").is_dir():
            prefixes.append(str(base))
    for extra in (root / "lib" / "rocm_sysdeps", root / "lib" / "llvm", root / "llvm"):
        if extra.is_dir():
            prefixes.append(str(extra))
    prefixes = list(dict.fromkeys(prefixes))

    hipblaslt = ""
    for base in [root, *pkgs]:
        h = base / "include" / "hipblaslt" / "hipblaslt-version.h"
        if h.exists():
            t = h.read_text(encoding="utf-8", errors="replace")
            try:
                v = [int(re.search(rf"#define\s+HIPBLASLT_VERSION_{k}\s+(\d+)", t).group(1))
                     for k in ("MAJOR", "MINOR", "PATCH")]
                hipblaslt = str(v[0] * 100000 + v[1] * 100 + v[2])
            except AttributeError:
                pass
            break

    # gfx device code: rocBLAS / hipBLASLt kernel libraries named after the target.  Without them every BLAS call
    # fails at run time with "no kernel" - stop here instead.
    gfx_hits = 0
    for p in pkgs:
        for f in p.rglob(f"*{a.gfx}*"):
            gfx_hits += 1
            if gfx_hits > 3:
                break
        if gfx_hits > 3:
            break
    if gfx_hits == 0:
        die(f"no {a.gfx} device libraries in {', '.join(map(str, pkgs))} (install rocm[device-{a.gfx}])")

    if a.link:
        link = Path(a.link)
        if link.is_symlink() or link.exists():
            if link.is_symlink() or link.is_file():
                link.unlink()
            else:
                shutil.rmtree(link)
        link.symlink_to(root)

    if a.ldconf:
        Path(a.ldconf).write_text("\n".join(libdirs) + "\n", encoding="utf-8")

    def q(s: str) -> str:
        return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'

    env = {
        "XOLOTL_ROCM_VERSION": a.rocm_version,
        "XOLOTL_ROCM_ROOT": str(root),
        "XOLOTL_HIP_CLANG": str(clang),
        "XOLOTL_ROCM_BITCODE": str(bitcode),
        "XOLOTL_ROCM_PREFIX": ";".join(prefixes),
        "XOLOTL_ROCM_LIBDIRS": ":".join(libdirs),
        "XOLOTL_ROCM_BINDIRS": ":".join(str(d) for d in (root / "bin", clang.parent) if d.is_dir()),
        "XOLOTL_HIPBLASLT_VERSION": hipblaslt,
        "XOLOTL_GFX": a.gfx,
    }
    Path(a.env).write_text("# written by rocm-discover.py at image build time\n"
                           + "".join(f"{k}={q(v)}\n" for k, v in env.items()), encoding="utf-8")
    print(f"rocm-discover: root {root}, {len(pkgs)} packages, {len(libdirs)} lib dirs, hipBLASLt {hipblaslt or '?'}")


if __name__ == "__main__":
    main()
