#!/usr/bin/env python3
"""compare.py - per-position NLL distances (mean |dNLL| by position range) between the ablation chains (nll.npz),
the stored torch chains and the engine runs (~/.cache/xolotl/runs/nll-*.txt)."""
import json, os, sys
import numpy as np

CK = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/.cache/xolotl/ablate")
RUNS = os.path.expanduser("~/.cache/xolotl/runs")
tokens = np.load(os.path.expanduser("~/models/xolotl/v41-work/calib-1017/tokens.npy"), mmap_mode="r")[12, :4096]
z = np.load(os.path.join(CK, "nll.npz"))
nll = {k[4:]: z[k] for k in z.files if k.startswith("nll_")}
top1 = {k[5:]: z[k] for k in z.files if k.startswith("top1_")}
kl = {k[3:]: z[k] for k in z.files if k.startswith("kl_")}
for f in ("gsqtorch", "teacher", "fid-base", "fid-all"):
    a = np.loadtxt(os.path.join(RUNS, f"nll-{f}.txt"))
    nll["file:" + f] = a[:, 1]
    top1["file:" + f] = a[:, 2].astype(np.int64)   # column 3 = the run's argmax
R = [(0, 1024), (1024, 2048), (2048, 4095)]


def dist(a, b):
    n = min(len(nll[a]), len(nll[b]))
    d = np.abs(nll[a][:n] - nll[b][:n])
    return [float(d[lo:min(hi, n)].mean()) for lo, hi in R]


def show(pairs):
    for a, b in pairs:
        r = dist(a, b)
        extra = ""
        if a in top1 and b in top1:
            ag = (top1[a] == top1[b])
            extra = " | argmax agree " + " ".join(f"{ag[lo:hi].mean():.3f}" for lo, hi in R)
        if b == "E" and a in kl:
            extra += " | KL(E||v) " + " ".join(f"{kl[a][lo:hi].mean():.4f}" for lo, hi in R)
        print(f"{a:>18s} vs {b:<18s} mean|dNLL| " + " ".join(f"{x:.3f}" for x in r) + extra)


print("ppl:", {k: round(float(np.exp(v[:4095].mean())), 3) for k, v in nll.items()})
print("--- sanity: stored chains vs their files, my T vs the stored teacher")
show([("gsqtorch_stored", "file:gsqtorch"), ("teacher_stored", "file:teacher"), ("T", "teacher_stored")])
print("--- ablations vs exact GSQ chain E (GGUF experts, exact math)")
show([(v, "E") for v in ("N", "C", "A", "B", "AB", "T")])
print("--- E vs the torch GSQ chain the 0.28 refers to (a different GSQ run: other expert bytes)")
show([("E", "gsqtorch_stored"), ("N", "gsqtorch_stored"), ("AB", "gsqtorch_stored"), ("T", "gsqtorch_stored")])
print("--- engine vs chains")
for eng in ("file:fid-base", "file:fid-all"):
    show([(eng, x) for x in ("gsqtorch_stored", "E", "N", "A", "B", "AB", "C")])
