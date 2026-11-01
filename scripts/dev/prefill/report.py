#!/usr/bin/env python3
"""report.py NAME [NAME..] - per-turn prefill table from turns.py runs (~/.cache/xolotl/runs/prefill/NAME.*).

Per request: new tokens, context before, client TTFT, the engine's prompt phases (dense / experts / logits / DSpark
tail), file reads, token-expert pairs, MIMO_CHUNK_PROF (GPU grouped-kernel ms, read waits, reader GB/s) and
STRATA_DENSE_PROF (gpu compute / build / alloc) when the run had them, plus the 990's mean read MB/s during the request
the KFD evicted_ms that accrued while it ran (evict, s: queue evictions stall the GPU) and whether hf_upload ran.  Derived: us per token-expert pair (experts - read waits), ms per dense column-layer.
python3 -B (no __pycache__ on the share).
"""
import json
import os
import re
import sys

RUNS = os.path.expanduser("~/.cache/xolotl/runs/prefill")


def blocks(log):
    """Split the engine log at each 'serve: reuse' line (one block per request)."""
    out, cur = [], None
    for ln in open(log, errors="replace"):
        if ln.startswith("serve: reuse"):
            cur = []
            out.append(cur)
        if cur is not None:
            cur.append(ln.rstrip("\n"))
    return out


def num(rx, s, i=1, f=float):
    m = re.search(rx, s)
    return f(m.group(i)) if m else None


def parse(block):
    t = "\n".join(block)
    r = {}
    r["phases_s"] = num(r"prompt phases: \d+ new = \d+ \w+ \(([\d.]+) s", t)
    r["dense"] = num(r"prompt phases: .*?dense ([\d.]+)", t)
    r["experts"] = num(r"prompt phases: .*?experts ([\d.]+)", t)
    r["logits"] = num(r"prompt phases: .*?logits ([\d.]+)", t)
    r["tail_open"] = num(r"cache open ([\d.]+) s", t)
    r["prompt_s"] = num(r"serve: prompt \d+ tokens \(\d+ reused, [\w ]+\) in ([\d.]+) s", t)
    r["pairs"] = num(r"/ (\d+) pcie / (\d+) file\)", t, 1, int)
    r["file"] = num(r"/ (\d+) pcie / (\d+) file\)", t, 2, int)
    r["batch"] = num(r"prompt batch (\d+)", t, 1, int)
    r["drv_wait_ms"] = num(r"after waiting (\d+) ms for the driver", t, 1, int)
    m = re.search(r"chunk prof: (\d+) calls, (\d+) launches: wall (\d+) ms = waits on reads (\d+) \+ wave/copy waits (\d+) "
                  r"\+ final sync (\d+) \+ other host (-?\d+); reader wall (\d+) ms; GPU grouped kernels (\d+) ms "
                  r"\(resident (\d+)\); file tier ([\d.]+) GB read in (\d+) ms of reader time = ([\d.]+) GB/s", t)
    if m:
        g = [float(x) for x in m.groups()]
        r.update(cp_calls=g[0], cp_wall=g[2], cp_readwait=g[3], cp_copywait=g[4], cp_sync=g[5], cp_host=g[6],
                 cp_kern=g[8], cp_gb=g[10], cp_rd_ms=g[11], cp_gbs=g[12])
    m = re.search(r"dense prof turn: (\d+) attn spans: build (\d+) ms, alloc (\d+), gpu compute (\d+), engram wait (\d+), "
                  r"readback (\d+), upload (\d+), finish build (\d+)", t)
    if m:
        g = [float(x) for x in m.groups()]
        r.update(dp_spans=g[0], dp_build=g[1], dp_alloc=g[2], dp_gpu=g[3], dp_engram=g[4], dp_readback=g[5], dp_upload=g[6])
    r["ext_release"] = "arena ext: released" in t
    r["dspark_freed"] = "DSpark weights freed" in t
    r["enc"] = num(r"phase A (\d+) cols", t, 1, int)
    m = re.search(r"pred prefetch: (\d+) layers, (\d+) blobs issued, (\d+) read before their layer ran, (\d+) used", t)
    if m:   # STRATA_PF_PRED_GIB (docs/PREFILL.md #3)
        r.update(pp_issued=int(m.group(2)), pp_done=int(m.group(3)), pp_used=int(m.group(4)))
    r["want"] = num(r"for the driver \(want ([\d.]+)\)", t)
    return r


def main():
    for name in sys.argv[1:]:
        res = json.load(open(f"{RUNS}/{name}.json"))
        bl = blocks(f"{RUNS}/{name}.log")
        samp = []
        if os.path.exists(f"{RUNS}/{name}.samp"):
            for ln in open(f"{RUNS}/{name}.samp"):
                p = ln.split()
                if len(p) >= 5:
                    samp.append([float(x) for x in p])
        print(f"== {name}")
        hdr = ("tag", "ctx", "new", "ttft", "prompt", "dense", "exp", "kern", "rdwait", "cpwait", "file", "pairs",
               "us/pr", "us/pr-k", "GB/s", "ms/cl", "dgpu", "dbuild", "dalloc", "rdMB/s", "evict", "hf")
        print(" ".join(f"{h:>8}" for h in hdr))
        for q, b in zip(res, bl):
            r = parse(b)
            ctx = q["prompt"] - q["new"]
            pairs = r.get("pairs") or 0
            colayers = pairs / 6 if pairs else 0
            exp = r.get("experts") or 0
            us = 1e6 * exp / pairs if pairs else 0
            usk = 1e3 * r["cp_kern"] / pairs if pairs and "cp_kern" in r else 0
            mscl = 1e3 * (r.get("dense") or 0) / colayers if colayers else 0
            t0, t1 = q["t_send"], q["t_send"] + q["wall"]
            w = [s for s in samp if t0 <= s[0] <= t0 + q["ttft"]]
            rd = sum(s[1] for s in w) / len(w) if w else 0
            hf = max((s[4] for s in w), default=0)
            evs = [s[3] for s in w if s[3] >= 0]
            ev = (evs[-1] - evs[0]) / 1e3 if len(evs) > 1 else 0.0
            row = (q["tag"], ctx, q["new"], q["ttft"], r.get("prompt_s"), r.get("dense"), exp, (r.get("cp_kern") or 0) / 1e3,
                   (r.get("cp_readwait") or 0) / 1e3, (r.get("cp_copywait") or 0) / 1e3, r.get("file"), pairs, us, usk,
                   r.get("cp_gbs"), mscl, (r.get("dp_gpu") or 0) / 1e3, (r.get("dp_build") or 0) / 1e3,
                   (r.get("dp_alloc") or 0) / 1e3, rd, ev, int(hf))
            print(" ".join(f"{x:>8.2f}" if isinstance(x, float) else f"{str(x):>8}" for x in row))
            extra = []
            if r.get("batch"):
                extra.append(f"batch capped {r['batch']}")
            if r.get("drv_wait_ms"):
                extra.append(f"driver wait {r['drv_wait_ms']} ms")
            if r.get("dspark_freed"):
                extra.append(f"DSpark freed (reopen {r.get('tail_open')} s)")
            if r.get("pp_issued") is not None:
                extra.append(f"pred prefetch {r['pp_used']} used / {r['pp_done']} read / {r['pp_issued']} issued")
            if r.get("enc"):
                extra.append(f"enc-skip phase A {r['enc']} cols")
            if extra:
                print("         " + "; ".join(extra))


if __name__ == "__main__":
    main()
