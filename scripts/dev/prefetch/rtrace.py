#!/usr/bin/env python3
"""rtrace.py ROUTE_TRACE [--from-pass N] - the DSpark verify passes' file-tier misses from a STRATA_ROUTE_TRACE
(ds41_generate.cpp; format in xolotl/tools/v41/sim/route_sim.py): misses per pass and per layer, how bursty a layer's
misses are, which share only the rejected draft rows caused, and a least-squares fit of verify ms on the pass's misses
(the marginal cost of a file miss in a verify pass).  python3 -B.
"""
import sys
import numpy as np

FILEG = 3


def load(path):
    b = open(path, 'rb').read()
    a = np.frombuffer(b, dtype=np.int32)
    L, X = int(a[1]), int(a[2])
    i = 3 + L * X * 3
    passes, calls = [], []
    while i < len(a):
        tag = a[i]
        if tag == ord('C'):
            layer, nt, nu = int(a[i + 1]), int(a[i + 2]), int(a[i + 3])
            calls.append((layer, nt, a[i + 4:i + 4 + 4 * nu].reshape(nu, 4).copy()))
            i += 4 + 4 * nu
        elif tag == ord('P'):
            n = int(a[i + 1])
            passes.append((np.frombuffer(b[(i + 2) * 4:(i + 2 + n) * 4], dtype=np.float32).copy(), calls))
            calls = []
            i += 2 + n
        else:
            raise SystemExit(f'bad tag {tag} at word {i}')
    return L, X, passes


def main():
    path = sys.argv[1]
    p0 = int(sys.argv[sys.argv.index('--from-pass') + 1]) if '--from-pass' in sys.argv else 0
    L, X, passes = load(path)
    passes = passes[p0:]
    per_layer = np.zeros(L); per_layer_n = np.zeros((L, 12))
    rows = acc_rows = 0; fm = []; ms = []; rej_only = tot = 0; distinct = 0; ncall = 0
    by_row = np.zeros(8)
    for v, calls in passes:
        m, k, acc, refeed, vms = int(v[0]), int(v[1]), int(v[2]), int(v[3]), float(v[4])
        last_acc = refeed + acc   # rows 0..refeed+acc are kept (re-fed token, t, accepted drafts)
        rows += m; acc_rows += acc + 1
        f = 0
        for (layer, nt, u) in calls:
            ncall += 1; distinct += len(u)
            fg = u[u[:, 2] == FILEG]
            per_layer[layer] += len(fg); per_layer_n[layer, min(len(fg), 11)] += 1
            f += len(fg)
            tot += len(fg); rej_only += int((fg[:, 1] > last_acc).sum())
            for r in fg[:, 1]: by_row[min(int(r), 7)] += 1
        fm.append(f); ms.append(vms)
    n = len(passes)
    fm = np.array(fm, float); ms = np.array(ms)
    A = np.vstack([fm, np.ones(n)]).T
    (slope, icpt), *_ = np.linalg.lstsq(A, ms, rcond=None)
    print(f"{path}: {n} passes, {rows / n:.2f} rows/pass, {acc_rows / n:.2f} emitted tokens/pass, "
          f"{distinct / max(ncall, 1):.1f} distinct experts per layer call")
    print(f"file misses: {tot / n:.2f}/pass = {tot / max(acc_rows, 1):.2f}/emitted token; only rejected draft rows: "
          f"{rej_only / max(tot, 1):.1%}")
    print(f"misses by first row: " + " ".join(f"r{i} {by_row[i] / max(tot, 1):.1%}" for i in range(6)))
    print(f"verify ms = {icpt:.1f} + {slope:.2f} x misses (fit over passes; mean {ms.mean():.1f} ms, {fm.mean():.1f} misses)")
    pl = per_layer / n
    print("misses per pass by layer: " + " ".join(f"{x:.2f}" for x in pl))
    frac0 = per_layer_n[:, 0].sum() / per_layer_n.sum()
    hist = per_layer_n.sum(0) / per_layer_n.sum()
    print(f"layer calls with 0/1/2/3/4/5+ misses: " + " ".join(f"{hist[i]:.1%}" for i in range(5)) + f" {hist[5:].sum():.1%}")
    # misses in a layer call weighted by the call's count: how much of the wait sits in bursts
    w = sum(per_layer_n[:, j].sum() * j for j in range(12))
    print("share of misses in calls with >= 3 / >= 5 misses: "
          f"{sum(per_layer_n[:, j].sum() * j for j in range(3, 12)) / w:.1%} / {sum(per_layer_n[:, j].sum() * j for j in range(5, 12)) / w:.1%}")


if __name__ == '__main__':
    main()
