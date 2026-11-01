#!/usr/bin/env python3
"""ab_cmp.py REF CFG [CFG..] - the 17-row same-text A/B (runs/ab/nll-<cfg>-<row>.txt from ab_v1v2.sh / ab_mmq.sh)
against the teacher (nll-T-*): per row ppl gap to T, top-1 agreement with T, and CFG vs REF (ppl, top-1 agreement
with REF, bit-identical rows); pooled token-weighted, plus a paired bootstrap CI over rows.  python3 -B."""
import json, os, sys
import numpy as np
A = os.path.expanduser("~/.cache/xolotl/runs/ab/")
rows = json.load(open(A + "rows.json"))
ref, cfgs = sys.argv[1], sys.argv[2:]
ld = lambda k, f: np.loadtxt(f"{A}nll-{k}-{f}.txt") if os.path.exists(f"{A}nll-{k}-{f}.txt") and os.path.getsize(f"{A}nll-{k}-{f}.txt") else None
for c in cfgs:
    print(f"== {c} vs {ref} (and both vs the teacher T)")
    print(f"{'row':8s} {'domain':13s} {'T ppl':>8s} {ref + ' gap':>9s} {c + ' gap':>9s} {c + '-' + ref:>9s} {'top1=T ' + ref + '/' + c:>17s} {'argmax=' + ref:>11s} {'same':>5s}")
    P = {"T": [], "r": [], "c": [], "agr": [], "agc": [], "rc": []}
    d = []
    for r in rows:
        T, a, b = ld("T", r["file"]), ld(ref, r["file"]), ld(c, r["file"])
        if T is None or a is None or b is None or not len(T) == len(a) == len(b):
            continue
        tn, ta = T[:, 1], T[:, 2].astype(int)
        g = lambda x: np.exp(x[:, 1].mean() - tn.mean()) - 1
        ag = lambda x: (x[:, 2].astype(int) == ta).mean()
        rc = (a[:, 2].astype(int) == b[:, 2].astype(int))
        same = np.array_equal(a, b)
        dd = b[:, 1].mean() - a[:, 1].mean()
        d.append(dd)
        for k, v in (("T", tn), ("r", a[:, 1]), ("c", b[:, 1]), ("agr", a[:, 2].astype(int) == ta), ("agc", b[:, 2].astype(int) == ta), ("rc", rc)):
            P[k].append(v)
        print(f"{r['file']:8s} {r['domain']:13s} {np.exp(tn.mean()):8.3f} {g(a):+9.2%} {g(b):+9.2%} {np.exp(dd) - 1:+9.2%} "
              f"{ag(a):8.3f}/{ag(b):.3f} {rc.mean():11.4f} {'yes' if same else 'no':>5s}")
    if not d:
        continue
    C = {k: np.concatenate(v) for k, v in P.items()}
    gap = lambda k: np.exp(C[k].mean() - C["T"].mean()) - 1
    d = np.array(d)
    rng = np.random.default_rng(0)
    bs = np.array([d[rng.integers(len(d), size=len(d))].mean() for _ in range(20000)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print(f"pooled over {len(d)} rows: gap to T {ref} {gap('r'):+.2%} / {c} {gap('c'):+.2%}; {c} vs {ref} ppl {np.exp(C['c'].mean() - C['r'].mean()) - 1:+.2%}; "
          f"top-1=T {C['agr'].mean():.4f} / {C['agc'].mean():.4f}; argmax agreement {c} vs {ref} {C['rc'].mean():.4f}")
    print(f"paired per row ({c} - {ref}, < 0 = {c} better): better on {(d < 0).sum()}/{len(d)}; mean {np.exp(d.mean()) - 1:+.2%}, "
          f"95% CI [{np.exp(lo) - 1:+.2%}, {np.exp(hi) - 1:+.2%}]")
