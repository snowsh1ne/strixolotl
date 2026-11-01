#!/usr/bin/env python3
"""pred.py TRACE FORCE_IDS [--from 1024] [--policy lfur:0.995] [--extra 635] [--passes CONFLOG] [--emb EMB.npz]
            [--json OUT]
Offline recall / precision of file-tier read-ahead predictors on a plain-decode STRATA_DEC_TRACE that was recorded
with STRATA_FILE_PRED=1 (router k layers ahead, top 16, k = 1..4; e.g. runs/nvme/T1.trace + runs/ab/warm/force.ids,
the forced warm bench).  Run with python3 -B (no __pycache__ in the repo).

Misses = routed experts outside the RAM arena, the arena replayed with ../nvme/sim.py's Arena (default lfur:0.995 =
the served STRATA_ARENA_LFU, --extra = free extension slots like STRATA_ARENA_EXT_GIB).  The arena state of layer L
only changes in layer-L calls, so "in the arena when the read would be issued" = "at the call".

Predictors of layer L's routing at step s (a read is issued for every predicted expert outside the arena):
  rK.N    the trunk router of layer L applied to the state entering layer L-K (the 'P' records), top N
  tok.N   token-id history: the experts layer L routed the last times this input token id was decoded, top N by count
  emb.N   the token embedding (rms-normed) through layer L's router (+ exp_probs_b), top N (needs --emb, emb_router.py)
  prev    the previous step's layer-L experts (temporal reuse; the arena already holds most of them)
  a+b     union
--passes CONFLOG (STRATA_DSPARK_CONFLOG of a DSpark run: "k acc ms p.."): also group the steps into pseudo verify passes
of acc+1 accepted rows (the real pass also carries the k-acc rejected drafts: not in a plain trace) and score the
per-pass unions (one call per layer per pass, as gpu_run_n_ex): misses / reads per pass.
"""
import sys, os, struct, collections, json
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "nvme"))
import sim  # noqa: E402  (Arena)


def load(path):
    b = open(path, "rb").read()
    o = 0
    arena0 = None; R = []; P = {}
    step = -1; nl = ne = 0
    while o + 4 <= len(b):
        tag, = struct.unpack_from("<i", b, o)
        need = {ord('A'): 12, ord('R'): 56, ord('P'): 76, ord('O'): 16}.get(tag, 4)
        if o + need > len(b): break
        if tag == ord('A'):
            nl, ne = struct.unpack_from("<ii", b, o + 4); o += 12
            arena0 = bytearray(b[o:o + nl * ne]); o += nl * ne
        elif tag == ord('R'):
            l, = struct.unpack_from("<i", b, o + 4); o += 8
            ids = struct.unpack_from("<6i", b, o); o += 24
            o += 24
            if l == 0: step += 1
            R.append((step, l, ids))
        elif tag == ord('P'):
            l, k = struct.unpack_from("<ii", b, o + 4); o += 12
            ids = struct.unpack_from("<16i", b, o); o += 64
            P[(step + 1 if l == 0 else step, l + k, k)] = ids   # keyed by the TARGET layer
        elif tag == ord('O'):
            o += 16
        else:
            raise SystemExit(f"bad tag {tag} at {o}")
    if R and len(R) % nl: R = R[: len(R) - len(R) % nl]
    return nl, ne, arena0, R, P


def main():
    a = sys.argv[1:]
    opt = {"--from": "1024", "--policy": "lfur:0.995", "--extra": "635", "--passes": "", "--emb": "", "--json": ""}
    pos = [x for i, x in enumerate(a) if not x.startswith("--") and (i == 0 or not a[i - 1].startswith("--"))]
    for i, x in enumerate(a):
        if x.startswith("--"): opt[x] = a[i + 1]
    trace, fids = pos[0], pos[1]
    start = int(opt["--from"])
    sim.EXTRA = int(opt["--extra"])
    nl, ne, arena0, R, P = load(trace)
    toks = [int(x) for x in open(fids) if x.strip().lstrip('-').isdigit()]
    nsteps = R[-1][0] + 1
    routing = {}   # (s, l) -> ids
    for (s, l, ids) in R: routing[(s, l)] = ids
    emb = None
    if opt["--emb"]:
        import numpy as np
        z = np.load(opt["--emb"])
        emb = {int(t): z["top"][i] for i, t in enumerate(z["tok"])}   # tok -> [L, 16] ranked ids
    # ---- the predictors, as functions (s, l) -> ranked list (computed BEFORE step s's layer-l call)
    hist = collections.defaultdict(list)   # tok -> list of steps (filled after each step)

    def tokpred(s, l, n):
        t = toks[s] if s < len(toks) else None
        c = collections.Counter()
        for j, s2 in enumerate(reversed(hist.get(t, [])[-8:])):
            for r, e in enumerate(routing[(s2, l)]): c[e] += 8 - j + (6 - r) * 0.01
        return [e for e, _ in c.most_common(n)]

    preds = {}
    for K in (1, 2, 3, 4):
        for N in (2, 4, 6, 8, 16):
            preds[f"r{K}.{N}"] = (lambda K, N: lambda s, l: list(P.get((s, l, K), ())[:N]))(K, N)
    for N in (6, 12, 24):
        preds[f"tok.{N}"] = (lambda N: lambda s, l: tokpred(s, l, N))(N)
    if emb is not None:
        for N in (4, 8, 16):
            preds[f"emb.{N}"] = (lambda N: lambda s, l: list(emb[toks[s]][l][:N]) if s < len(toks) and toks[s] in emb else [])(N)
    preds["prev"] = lambda s, l: list(routing.get((s - 1, l), ()))
    preds["r1.4+tok.6"] = lambda s, l: list(dict.fromkeys(preds["r1.4"](s, l) + preds["tok.6"](s, l)))
    preds["r1.8+tok.6"] = lambda s, l: list(dict.fromkeys(preds["r1.8"](s, l) + preds["tok.6"](s, l)))
    preds["r2.4+r1.4"] = lambda s, l: list(dict.fromkeys(preds["r2.4"](s, l) + preds["r1.4"](s, l)))
    names = list(preds)
    # per predictor, per layer: [misses scored, hits, reads issued, useful reads]
    st = {n: [[0, 0, 0, 0] for _ in range(nl)] for n in names}
    A = sim.Arena(nl, ne, arena0, opt["--policy"])
    nw = 0; miss_l = [0] * nl
    for (s, l, ids) in R:
        win = s >= start
        if win:
            res = set(A.res[l])
            pv = {n: preds[n](s, l) for n in names}
        mm = A.call(l, ids)
        if win:
            if l == 0: nw += 1
            missed = set(e for e, m in zip(ids, mm) if m)
            miss_l[l] += len(missed)
            routed = set(ids)
            for n in names:
                p = [e for e in dict.fromkeys(pv[n]) if e >= 0 and e not in res]
                x = st[n][l]
                x[0] += len(missed); x[1] += len(missed & set(p)); x[2] += len(p); x[3] += len(set(p) & routed)
        if l == nl - 1 and s < len(toks): hist[toks[s]].append(s)
    print(f"trace {trace}: {nsteps} steps, window >= {start} ({nw} steps), policy {opt['--policy']} extra {sim.EXTRA}: "
          f"{sum(miss_l) / max(nw, 1):.2f} file misses/token")
    print(f"{'predictor':12s} {'recall':>7s} {'reads/tok':>9s} {'precision':>9s}   recall by layer 0-9 / 10-19 / 20-29 / 30-39")
    out = {"misses_per_token": sum(miss_l) / max(nw, 1), "miss_by_layer": [m / max(nw, 1) for m in miss_l], "pred": {}}
    for n in names:
        tot = [sum(x[i] for x in st[n]) for i in range(4)]
        rec = tot[1] / max(tot[0], 1); prc = tot[3] / max(tot[2], 1)
        grp = []
        for g in range(4):
            xs = st[n][g * 10:(g + 1) * 10]
            grp.append(sum(x[1] for x in xs) / max(sum(x[0] for x in xs), 1))
        print(f"{n:12s} {rec:7.1%} {tot[2] / max(nw, 1):9.2f} {prc:9.1%}   " + " / ".join(f"{g:5.1%}" for g in grp))
        out["pred"][n] = {"recall": rec, "reads_per_token": tot[2] / max(nw, 1), "precision": prc,
                          "recall_by_layer": [x[1] / max(x[0], 1) for x in st[n]],
                          "reads_by_layer": [x[2] / max(nw, 1) for x in st[n]]}
    if opt["--passes"]:
        acc = []
        for line in open(opt["--passes"]):
            f = line.split()
            if len(f) >= 2: acc.append(int(f[1]))
        # pseudo passes over the same steps: rows = acc+1 consecutive accepted tokens; one arena call per layer and pass
        A = sim.Arena(nl, ne, arena0, opt["--policy"])
        hist.clear()
        pst = {n: [0, 0, 0, 0] for n in names}
        s = 0; pi = 0; npass = 0; ntok = 0; mp = 0
        rows_hist = []
        while s < nsteps:
            m = acc[pi % len(acc)] + 1; pi += 1
            rows = list(range(s, min(s + m, nsteps)))
            win = rows[0] >= start
            for l in range(nl):
                u = list(dict.fromkeys(e for r in rows for e in routing[(r, l)]))
                if win:
                    res = set(A.res[l])
                    pv = {}
                    for n in names:
                        p = []
                        for r in rows: p += preds[n](r, l)
                        pv[n] = [e for e in dict.fromkeys(p) if e >= 0 and e not in res]
                mm = A.call(l, u)
                if win:
                    missed = set(e for e, q in zip(u, mm) if q)
                    mp += len(missed)
                    for n in names:
                        x = pst[n]
                        x[0] += len(missed); x[1] += len(missed & set(pv[n])); x[2] += len(pv[n]); x[3] += len(set(pv[n]) & set(u))
            for r in rows:
                if r < len(toks): hist[toks[r]].append(r)
            if win: npass += 1; ntok += len(rows)
            s += m
        print(f"pseudo passes (accepted rows only, {ntok / max(npass, 1):.2f} rows/pass): {mp / max(npass, 1):.2f} misses/pass "
              f"= {mp / max(ntok, 1):.2f}/token")
        print(f"{'predictor':12s} {'recall':>7s} {'reads/pass':>10s} {'precision':>9s}")
        out["passes"] = {"rows_per_pass": ntok / max(npass, 1), "misses_per_pass": mp / max(npass, 1), "pred": {}}
        for n in names:
            x = pst[n]
            print(f"{n:12s} {x[1] / max(x[0], 1):7.1%} {x[2] / max(npass, 1):10.2f} {x[3] / max(x[2], 1):9.1%}")
            out["passes"]["pred"][n] = {"recall": x[1] / max(x[0], 1), "reads_per_pass": x[2] / max(npass, 1),
                                        "precision": x[3] / max(x[2], 1)}
    if opt["--json"]:
        json.dump(out, open(opt["--json"], "w"), indent=1)


if __name__ == "__main__":
    main()
