#!/usr/bin/env python3
"""main_proxy.py GGUF DEC_TRACE MAIN_DUMP [--from 128] [--policy lfur:0.995] [--extra 635] - can a LATE-layer hidden state
(what the DSpark drafter reads and writes) predict the trunk's routing at layer L?  Proxy for "the drafter's hidden
through the trunk's layer-L router": the drafter's output is an estimate of the next token's late-layer state, so the
trunk's TRUE late state of the same token (STRATA_DUMP_MAIN: HC-mean of the states leaving layers 36/37/38, per decode
step) through router L is an optimistic bound for it; the previous token's state is what a read issued at the pass
start would actually have.  Prediction = top N of sqrt(softplus(W_L . rms_norm(h) * ffn_norm_L)) + exp_probs_b_L.
Scored like pred.py: recall of the file misses (arena replayed from the trace, ../nvme/sim.py) and reads issued.
Needs a plain-decode run with STRATA_DEC_TRACE + STRATA_DUMP_MAIN (same steps: both are written inside step()).
"""
import sys, os
import numpy as np
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "nvme")); sys.path.insert(0, HERE)
import sim            # noqa: E402
import pred           # noqa: E402
from emb_router import gguf_tensors   # noqa: E402


def main():
    a = sys.argv[1:]
    opt = {"--from": "128", "--policy": "lfur:0.995", "--extra": "635"}
    pos = [x for i, x in enumerate(a) if not x.startswith("--") and (i == 0 or not a[i - 1].startswith("--"))]
    for i, x in enumerate(a):
        if x.startswith("--"): opt[x] = a[i + 1]
    gg, tr, dm = pos[:3]
    sim.EXTRA = int(opt["--extra"]); start = int(opt["--from"])
    nl, ne, arena0, R, P = pred.load(tr)
    T = gguf_tensors(gg); fh = open(gg, "rb")

    def get(name):
        typ, dims, off = T[name]
        assert typ == 0, (name, typ)
        fh.seek(off)
        return np.frombuffer(fh.read(int(np.prod(dims)) * 4), np.float32).reshape(list(reversed(dims)))
    D = int(T["token_embd.weight"][1][0])
    raw = np.fromfile(dm, dtype=np.uint8)
    rec = 8 + 3 * D * 4
    ns = len(raw) // rec
    H = raw[: ns * rec].reshape(ns, rec)[:, 8:].copy().view(np.float32).reshape(ns, 3, D)
    nst = R[-1][0] + 1
    print(f"{ns} dump rows, {nst} trace steps (aligned 1:1 from the start)")
    n = min(ns, nst)
    Hn = H / np.sqrt((H * H).mean(2, keepdims=True) + 1e-6)
    tops = {}   # (variant, L) -> [n, 16]
    for L in range(nl):
        W = get(f"blk.{L}.ffn_gate_inp.weight"); nw = get(f"blk.{L}.ffn_norm.weight").reshape(D)
        b = get(f"blk.{L}.exp_probs_b.bias").reshape(-1)
        for j, name in ((0, "h36"), (2, "h38")):
            z = (Hn[:n, j] * nw) @ W.T
            sc = np.sqrt(np.logaddexp(0, z)) + b
            tops[(name, L)] = np.argsort(-sc, 1)[:, :16]
    variants = [(v, d, N) for v in ("h36", "h38") for d in (0, 1) for N in (4, 8, 16)]
    st = {x: np.zeros((nl, 4)) for x in variants}   # misses, hit, reads, routed-recall hits
    rt = np.zeros(nl)
    A = sim.Arena(nl, ne, arena0, opt["--policy"])
    for (s, l, ids) in R:
        res = set(A.res[l]) if s >= start and s < n else None
        mm = A.call(l, ids)
        if res is None: continue
        missed = set(e for e, q in zip(ids, mm) if q)
        rt[l] += 6
        for x in variants:
            v, d, N = x
            if s - d < 0: continue
            p = [int(e) for e in tops[(v, l)][s - d][:N]]
            pr = [e for e in p if e not in res]
            y = st[x][l]
            y[0] += len(missed); y[1] += len(missed & set(pr)); y[2] += len(pr); y[3] += len(set(p) & set(ids))
    nw_ = max(1, min(n, nst) - start)
    print(f"{'predictor':16s} {'miss recall':>11s} {'reads/tok':>9s} {'precision':>9s} {'routed recall':>13s}   miss recall by layer 0-9/10-19/20-29/30-39")
    for x in variants:
        v, d, N = x
        y = st[x]; t = y.sum(0)
        g = [y[i * 10:(i + 1) * 10, 1].sum() / max(y[i * 10:(i + 1) * 10, 0].sum(), 1) for i in range(4)]
        name = f"{v}{'(same tok)' if d == 0 else '(prev tok)'}.{N}"
        print(f"{name:16s} {t[1] / max(t[0], 1):11.1%} {t[2] / nw_:9.2f} {t[1] / max(t[2], 1):9.1%} {t[3] / rt.sum():13.1%}   "
              + " / ".join(f"{q:5.1%}" for q in g))


if __name__ == "__main__":
    main()
