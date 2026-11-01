#!/usr/bin/env python3
"""turns.py NAME SPEC.json [extra engine args...] - prefill / TTFT measurement of serve turns (docs/PREFILL.md).

Starts ds41_generate --serve exactly as `scripts/start_xolotl_dsv41 --model 1017 --no-auth --long --dspark` does (same
engine args and env, no HTTP front end), under `flock ~/.cache/xolotl/gpu.lock`, inside the xolotl-strix toolbox, and
drives it over the engine's line protocol (GEN n temperature=0 ids).  Every request is timed client side (send -> first
token = TTFT, send -> DONE).  A 1 s sampler logs the 990's read / write MB/s (/sys/block/nvme1n1/stat), the Kingston's r+w MB/s (nvme0n1), the engine's KFD
evicted_ms and whether scripts/hf_upload is running.

SPEC: a JSON list of requests {"src": ids.i32, "from": a, "to": b, "extend": true|false, "n": max_new, "tag": str}.
  extend=true: prompt = the previous request's prompt + its generated tokens + src[a:b] (an agent turn: the tool output
  appended to the conversation); extend=false: prompt = src[a:b] (a cold / unrelated prompt).
Env: PF_ENV="K=V K=V" extra engine env (MIMO_CHUNK_PROF=1, STRATA_DENSE_PROF=1, ...), PF_NO_DSPARK=1, PF_CTX (1048576).
Output: ~/.cache/xolotl/runs/prefill/NAME.{log,json,samp}.  Read it with report.py.
"""
import json
import os
import subprocess
import sys
import threading
import time

import numpy as np

REPO = os.path.expanduser("~/projects/Xolotl_StrixHalo")
RUNS = os.path.expanduser("~/.cache/xolotl/runs/prefill")
LOCK = os.path.expanduser("~/.cache/xolotl/gpu.lock")


def paths():
    p = {}
    for ln in open(os.path.expanduser("~/models/xolotl/V41-1017.paths")):
        ln = ln.split("#", 1)[0].strip()
        if "=" in ln:
            k, v = ln.split("=", 1)
            p[k] = os.path.expanduser(os.path.expandvars(v.strip().strip('"')))
    return p


def main():
    name, spec = sys.argv[1], json.load(open(sys.argv[2]))
    extra = sys.argv[3:]
    os.makedirs(RUNS, exist_ok=True)
    P = paths()
    dspark = os.environ.get("PF_NO_DSPARK") != "1"
    src = REPO + "/xolotl"
    env = {
        "STRATA_BUILD_DIR": os.path.expanduser("~/.cache/xolotl/build-strix"),
        "STRATA_DENSE_REQUANT": "", "STRATA_UMA_ZEROCOPY": "1", "STRATA_PF_NO_DUMMY": "1",
        "STRATA_DBG_OQUANT": "1", "STRATA_IDX_BF16": "1", "STRATA_DBG_PREC32": "1", "STRATA_DENSE_F16": "output.weight",
        "STRATA_FILE_FAST": "1", "STRATA_ARENA_LFU": "0.985" if dspark else "0.995", "STRATA_ARENA_EXT_GIB": "10",
        "STRATA_HIPBLASLT_TUNING": src + "/tools/hip/gfx1151-hipblaslt-100401.txt",
    }
    if dspark:
        env.update({"STRATA_SERVE_DSPARK": P["XOLOTL_DSV41_DSPARK"], "STRATA_DSPARK_DEPTH": "4", "STRATA_DSPARK_CONF": "0.6",
                    "STRATA_DSPARK_KEEP": src + "/tools/v41/serve/dspark_expert_counts_1017.txt:64"})
    for kv in os.environ.get("PF_ENV", "").split():
        k, v = kv.split("=", 1)
        env[k] = v
    args = ["--serve", "-m", P["XOLOTL_DSV41_GGUF"], "--backend", "cuda", "--experts", "gpu", "--engram",
            P["XOLOTL_DSV41_ENGRAM"], "--rank", P["XOLOTL_DSV41_RANK"], "--ctx", os.environ.get("PF_CTX", "1048576"),
            "--stop", "1", "--arena-adapt", "--kv-fp4", "--route-bias", "0", "--route-bias-file", "0",
            "--prompt-batch", "8192", "--uma", "on", "--uma-reserve-gib", "30" if dspark else "24", "--uma-cache-gib", "0",
            "--pf-b", "0", "--no-chunk-mmq", "--pcie", "1", "--tier-batch-pcie", "1"] + extra
    exports = " ".join(f"export {k}='{v}';" for k, v in env.items())
    inner = f". /opt/xolotl/rocm.env; {exports} exec {src}/tools/v41/serve/ds41-engine-strix.sh " + " ".join(f"'{a}'" for a in args)
    cmd = ["flock", LOCK, "toolbox", "run", "-c", "xolotl-strix", "bash", "-c", inner]
    # a previous engine frees its ~90 GiB pinned arena over seconds after it exits: the UMA arena is sized from
    # MemAvailable at load, so wait for it (as gen.sh does) or the run gets a smaller arena
    for _ in range(120):
        ma = int(next(l for l in open("/proc/meminfo") if l.startswith("MemAvailable")).split()[1]) >> 20
        busy = subprocess.run(["pgrep", "-x", "ds41_generate"], capture_output=True).returncode == 0
        if ma >= int(os.environ.get("PF_MW_START", "105")) and not busy:
            break
        time.sleep(2)
    log = open(f"{RUNS}/{name}.log", "w")
    log.write(f"# turns.py {name}: env {json.dumps(env)}\n# args {' '.join(args)}\n")
    log.flush()
    t_start = time.time()
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True, bufsize=1, cwd=REPO)

    samp = open(f"{RUNS}/{name}.samp", "w")
    stop = threading.Event()
    eng_pid = [None]

    def sampler():   # t, nvme1 read MB/s, write MB/s, evicted_ms, hf_upload running, nvme0 r+w MB/s
        last = None
        while not stop.is_set():
            try:
                f = open("/sys/block/nvme1n1/stat").read().split()
                g = open("/sys/block/nvme0n1/stat").read().split()
                rd, wr, rd0 = int(f[2]) * 512, int(f[6]) * 512, (int(g[2]) + int(g[6])) * 512
            except OSError:
                rd = wr = rd0 = 0
            ev = -1
            if eng_pid[0]:
                try:
                    import glob
                    for g in glob.glob(f"/sys/class/kfd/kfd/proc/{eng_pid[0]}/stats_*/evicted_ms"):
                        ev = int(open(g).read().split()[0])
                        break
                except (OSError, ValueError):
                    pass
            hf = subprocess.run(["pgrep", "-f", "scripts/hf_upload"], capture_output=True).returncode == 0
            now = time.time()
            if last:
                dt = now - last[0]
                samp.write(f"{now - t_start:.1f} {(rd - last[1]) / dt / 1e6:.0f} {(wr - last[2]) / dt / 1e6:.0f} {ev} {int(hf)} "
                           f"{(rd0 - last[3]) / dt / 1e6:.0f}\n")
                samp.flush()
            last = (now, rd, wr, rd0)
            stop.wait(1.0)
    th = threading.Thread(target=sampler, daemon=True)
    th.start()

    while True:
        ln = p.stdout.readline()
        if not ln:
            stop.set()
            sys.exit("engine died before READY (see log)")
        if ln.startswith("READY"):
            break
    r = subprocess.run(["pgrep", "-x", "ds41_generate"], capture_output=True, text=True).stdout.split()
    eng_pid[0] = r[0] if len(r) == 1 else None
    print(f"ready in {time.time() - t_start:.0f} s (engine pid {eng_pid[0]})", flush=True)
    res, prev = [], None
    try:
        for i, rq in enumerate(spec):
            ext = np.fromfile(os.path.expanduser(rq["src"]), np.int32)[rq.get("from", 0): rq.get("to", None)].tolist()
            ids = (prev["prompt"] + prev["tokens"] + ext) if rq.get("extend") and prev else ext
            t0 = time.time()
            p.stdin.write(f"GEN {rq.get('n', 8)} temperature=0 " + ",".join(map(str, ids)) + "\n")
            p.stdin.flush()
            toks, info, ttft = [], {}, None
            while True:
                ln = p.stdout.readline()
                if not ln:
                    raise RuntimeError("engine died mid-request")
                ln = ln.strip()
                if ln.startswith("T "):
                    if ttft is None:
                        ttft = time.time() - t0
                    toks.append(int(ln[2:]))
                elif ln.startswith("RESUME"):
                    info["resume"] = int(ln.split()[1])
                elif ln.startswith("DONE") or ln.startswith("ERR"):
                    info["end"] = ln
                    break
            prev = {"prompt": ids, "tokens": toks}
            row = {"i": i, "tag": rq.get("tag", ""), "prompt": len(ids), "new": len(ext), "tokens": toks,
                   "ttft": round(ttft or -1, 2), "wall": round(time.time() - t0, 2), "t_send": round(t0 - t_start, 1), **info}
            res.append(row)
            print(f"req {i} {row['tag']}: prompt {len(ids)} (+{len(ext)}) ttft {row['ttft']} s wall {row['wall']} s "
                  f"{info.get('end', '')[:60]}", flush=True)
            json.dump(res, open(f"{RUNS}/{name}.json", "w"), indent=1)
    finally:
        try:
            p.stdin.write("QUIT\n")
            p.stdin.flush()
            p.wait(timeout=300)
        except Exception:
            if eng_pid[0]:
                subprocess.run(["kill", eng_pid[0]])
            p.kill()
        stop.set()
        th.join(timeout=3)
    print(f"done in {time.time() - t_start:.0f} s", flush=True)


if __name__ == "__main__":
    main()
