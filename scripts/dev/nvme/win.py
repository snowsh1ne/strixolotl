#!/usr/bin/env python3
"""win.py [--from N] [--to N] LOG... - windowed warm-bench numbers from STRATA_DEC_PROF lines (running averages):
tok/s, file misses/token, attn_router / tier / other ms per token, tier ms per miss (tier minus a fixed 20 ms of
arena experts, divided by misses: an estimate) and the 'file-tier profile' line of DS4_TIER_PROF, if present.
Window = steps [from, to] (default 1024..last line)."""
import re, sys, os
prof = re.compile(r"decode step ms/token: begin ([\d.]+) predict ([\d.]+) attn_router ([\d.]+) tier ([\d.]+) finish ([\d.]+) "
                  r"spec ([\d.]+) \((\d+) steps\) file (\d+)")
a0, b0, logs, i = 1024, 10**9, [], 1
while i < len(sys.argv):
    if sys.argv[i] == "--from": a0 = int(sys.argv[i + 1]); i += 2
    elif sys.argv[i] == "--to": b0 = int(sys.argv[i + 1]); i += 2
    else: logs.append(sys.argv[i]); i += 1
for f in logs:
    t = open(f, errors="ignore").read()
    P = [tuple(float(x) for x in m[:6]) + (int(m[6]), int(m[7])) for m in prof.findall(t)]
    if len(P) < 2: print(f"{os.path.basename(f):28s} no prof lines"); continue
    ia = next((k for k, p in enumerate(P) if p[6] >= a0), 0)
    ib = max(k for k, p in enumerate(P) if p[6] <= b0)
    na, nb = P[ia][6], P[ib][6]
    if nb <= na: print(f"{os.path.basename(f):28s} window empty ({na}..{nb})"); continue
    st = [(P[ib][k] * nb - P[ia][k] * na) / (nb - na) for k in range(6)]
    miss = (P[ib][7] - P[ia][7]) / (nb - na)
    tot = sum(st)
    fp = re.search(r"file-tier profile.*", t)
    print(f"{os.path.basename(f):28s} steps {na}..{nb}: {1000 / tot:5.2f} tok/s  {tot:6.1f} ms/token  misses/token {miss:5.2f}  "
          f"attn_router {st[2]:5.1f} tier {st[3]:6.1f} other {st[0] + st[1] + st[4] + st[5]:4.1f}  "
          f"(tier-20)/miss {(st[3] - 20) / miss if miss else 0:5.2f} ms")
    if fp: print("    " + fp.group(0)[:400])
