#!/usr/bin/env python3
"""timeline.py TRACE FORCE_IDS CONFLOG [--from 1024] [--read-ms 2.5] [--dense-ms 110] [--tier-ms 81] [--post-ms 6]
               [--draft-ms 15] [--pred-ms 0.25] [--stash 32] [--emb EMB.npz] [--cfg "none,r1.4,..."] [--admit 0|1]
               [--scale-miss 1.0] [--parts 1]
Timeline model of DSpark verify passes with NVMe read-ahead, on the plain-decode trace (pred.py's predictors and
pseudo passes: acc+1 accepted rows per pass from CONFLOG's acceptance sequence).

Per pass: --dense-ms (attention + router, all 40 layers) and --tier-ms (the tier WITHOUT its file wait) are spread
evenly over the layers; a layer's routing is known after its dense share, its file misses are demand reads, the
layer waits for them, then its tier share runs.  One device serves 14.5 MiB blobs back to back at --read-ms each
(2.5 = ~6 GB/s, the 990 Pro), demand before read-ahead, a started read runs to the end (sim.py's Device).  After the
last layer: --post-ms (logits) + --draft-ms (the drafter) with the device idle except for read-ahead.

A config is '+'-joined parts:
  rK.N      at layer L-K's start, the union over the pass rows of the router-ahead top N for layer L (costs --pred-ms of
            critical path per layer and K: one predict_multi graph)
  uK.B      the pass's ranked union of the rows' router-ahead top 16 (STRATA_D3_SPEC's ranking), the first B outside the
            arena (a fixed read budget per layer)
  oK / oS   oracles: the exact routing K layers ahead / of every layer at the pass start (upper bounds)
  tok.N / emb.N   token-only guesses for every layer, issued when the pass's tokens are known (the drafter's end)
At most --stash read-ahead blobs in flight + waiting; a guess for a layer is dropped when that layer has run
(a queued, unstarted read is cancelled).  --admit 1: every read-ahead blob is admitted to the arena (lands in an LFU
victim slot) whether used or not - the arena pollution cost; 0: a separate stash, only used blobs are admitted (as
demand misses are).  --scale-miss: multiply the reported file wait by this (the rejected-draft rows' share, from the
real route trace) - informational only.
"""
import sys, os, collections
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "nvme"))
sys.path.insert(0, HERE)
import sim   # noqa: E402
import pred  # noqa: E402


class PartDevice(sim.Device):
    """sim.Device with blobs read in `parts` pieces: a demand read waits at most for the running PART of a read-ahead
    blob (the engine splits blobs into parts over its reader pool), not for the whole blob"""
    def __init__(s, read_ms, parts):
        super().__init__(read_ms); s.parts = parts; s.left = {}
    def add(s, key, prio, t):
        s.n += 1; s.q.append([prio, s.n, key, t]); s.left[key] = s.parts
    def _one(s, t_lim):
        s.q.sort()
        r = s.q[0]
        st = max(s.free, r[3])
        if t_lim is not None and st > t_lim: return False
        s.free = st + s.rm / s.parts
        s.left[r[2]] -= 1
        if s.left[r[2]] == 0:
            s.q.pop(0); s.done[r[2]] = s.free
        else:
            s.done.setdefault(("started",) + tuple(r[2][1:]), st)
        return True
    def run_until(s, t):
        while s.q and s._one(t): pass
    def wait(s, keys, t):
        keys = [k for k in keys if k not in s.done or s.done[k] > t]
        while any(k not in s.done for k in keys): s._one(None)
        return max([t] + [s.done[k] for k in keys])
    def started(s, key):
        return key in s.done or ("started",) + tuple(key[1:]) in s.done


def main():
    a = sys.argv[1:]
    opt = {"--from": "1024", "--read-ms": "2.5", "--dense-ms": "110", "--tier-ms": "81", "--post-ms": "6",
           "--draft-ms": "15", "--pred-ms": "0.25", "--stash": "32", "--emb": "", "--admit": "0", "--extra": "635",
           "--policy": "lfur:0.995", "--parts": "1",
           "--cfg": "none,r1.2,r1.4,r1.6,r1.8,r2.4,r2.8,r1.4+r2.4,r1.8+r2.8,r1.4+r2.4+r3.4,tok.6,r1.4+tok.6,r1.8+tok.6"}
    pos = [x for i, x in enumerate(a) if not x.startswith("--") and (i == 0 or not a[i - 1].startswith("--"))]
    for i, x in enumerate(a):
        if x.startswith("--"): opt[x] = a[i + 1]
    trace, fids, conflog = pos[:3]
    start = int(opt["--from"]); rm = float(opt["--read-ms"])
    sim.EXTRA = int(opt["--extra"])
    nl, ne, arena0, R, P = pred.load(trace)
    toks = [int(x) for x in open(fids) if x.strip().lstrip('-').isdigit()]
    routing = {(s, l): ids for (s, l, ids) in R}
    nsteps = R[-1][0] + 1
    acc = [int(l.split()[1]) for l in open(conflog) if len(l.split()) >= 2]
    emb = None
    if opt["--emb"]:
        import numpy as np
        z = np.load(opt["--emb"])
        emb = {int(t): z["top"][i] for i, t in enumerate(z["tok"])}
    dl = float(opt["--dense-ms"]) / nl; tl = float(opt["--tier-ms"]) / nl
    post = float(opt["--post-ms"]); draft = float(opt["--draft-ms"]); pms = float(opt["--pred-ms"])
    cap = int(opt["--stash"]); admit = opt["--admit"] == "1"

    def run(cfg):
        parts = [] if cfg == "none" else cfg.split("+")
        rk = [(int(p[1:].split(".")[0]), int(p.split(".")[1])) for p in parts if p.startswith("r")]
        uk = [(int(p[1:].split(".")[0]), int(p.split(".")[1])) for p in parts if p.startswith("u")]   # uK.B: ranked union
        ok = [int(p[1:]) for p in parts if p.startswith("o") and p[1:].isdigit()]   # oK: the exact routing, K layers ahead
        oS = "oS" in parts                                                         # exact routing of every layer at the pass start
        tk = [int(p.split(".")[1]) for p in parts if p.startswith("tok")]
        ek = [int(p.split(".")[1]) for p in parts if p.startswith("emb")]
        A = sim.Arena(nl, ne, arena0, opt["--policy"])
        D = PartDevice(rm, int(opt["--parts"]))
        hist = collections.defaultdict(list)
        t = 0.0; t0 = None; npass = ntok = 0
        miss = cov = iss = wasted = 0; wait = 0.0
        stash = {}   # (pass, layer, e) -> key
        s = 0; pi = 0

        def tokguess(r, l, n):
            c = collections.Counter()
            for j, s2 in enumerate(reversed(hist.get(toks[r], [])[-8:])):
                for q, e in enumerate(routing[(s2, l)]): c[e] += 8 - j + (6 - q) * 0.01
            return [e for e, _ in c.most_common(n)]

        def issue(pid, L, cand, tnow):
            nonlocal iss
            for e in cand:
                if e < 0 or A.has(L, e) or (pid, L, e) in stash: continue
                live = sum(1 for k in stash.values() if k not in D.done or D.done[k] > tnow)
                if live >= cap: return
                key = ("s", pid, L, e); stash[(pid, L, e)] = key; D.add(key, 1, tnow)
                if win: iss += 1
        while s < nsteps:
            m = acc[pi % len(acc)] + 1
            rows = list(range(s, min(s + m, nsteps)))
            win = rows[0] >= start
            if win and t0 is None: t0 = t
            pid = pi; pi += 1
            # token-only guesses: the pass's tokens are known after the drafter (the device was idle meanwhile)
            if oS:
                for L in range(nl): issue(pid, L, list(dict.fromkeys(e for r in rows for e in routing[(r, L)])), t - draft)
            if tk or ek:
                for L in range(nl):
                    cand = []
                    for r in rows:
                        for n in tk: cand += tokguess(r, L, n)
                        for n in ek:
                            if r < len(toks) and toks[r] in emb: cand += [int(x) for x in emb[toks[r]][L][:n]]
                    issue(pid, L, list(dict.fromkeys(cand)), t - draft)
            for l in range(nl):
                for (K, N) in rk:
                    L = l + K
                    if L >= nl: continue
                    cand = []
                    for r in rows: cand += list(P.get((r, L, K), ())[:N])
                    issue(pid, L, list(dict.fromkeys(cand)), t)
                    t += pms
                for (K, B) in uk:   # the pass's ranked union of the rows' router-ahead top 16 (score 16 - rank, summed,
                    L = l + K           # as STRATA_D3_SPEC), the first B outside the arena
                    if L >= nl: continue
                    sc = collections.Counter()
                    for r in rows:
                        for q, e in enumerate(P.get((r, L, K), ())):
                            if e >= 0 and not A.has(L, e): sc[e] += 16 - q
                    issue(pid, L, [e for e, _ in sc.most_common(B)], t)
                    t += pms
                for K in ok:
                    if l + K < nl: issue(pid, l + K, list(dict.fromkeys(e for r in rows for e in routing[(r, l + K)])), t)
                t += dl
                D.run_until(t)
                u = list(dict.fromkeys(e for r in rows for e in routing[(r, l)]))
                mm = A.call(l, u)
                need = []
                for e, q in zip(u, mm):
                    if not q: continue
                    key = stash.get((pid, l, e))
                    if key is not None:
                        D.promote(key); need.append(key)
                        if win: cov += 1
                    else:
                        key = ("d", pid, l, e); D.add(key, 0, t); need.append(key)
                    if win: miss += 1
                t1 = D.wait(need, t)
                if win: wait += t1 - t
                t = t1 + tl
                used = set(e for e, q in zip(u, mm) if q)
                for x in [x for x in stash if x[0] == pid and x[1] == l]:
                    key = stash.pop(x)
                    if x[2] not in used:
                        if win and D.started(key): wasted += 1
                        D.q = [r for r in D.q if r[2] != key]
                        if admit and D.started(key) and not A.has(l, x[2]):
                            A.call(l, [x[2]])   # the blob sits in an arena victim slot now (an extra 'use' too)
            t += post
            # the drafter: tokens for the next pass appear at the end; the device keeps working meanwhile
            t += draft
            D.run_until(t)
            for r in rows:
                if r < len(toks): hist[toks[r]].append(r)
            if win: npass += 1; ntok += len(rows)
            s += m
        ms_pass = (t - t0) / max(npass, 1)
        return ms_pass, miss / max(npass, 1), cov / max(miss, 1), iss / max(npass, 1), wasted / max(npass, 1), \
            wait / max(npass, 1), ntok / max(npass, 1)
    base = None
    print(f"timeline: read {rm} ms/blob, dense {opt['--dense-ms']} + tier {opt['--tier-ms']} + post {post} + draft {draft} "
          f"ms/pass, predict {pms} ms per (layer, K), stash {cap}, admit {int(admit)}, policy {opt['--policy']} "
          f"extra {sim.EXTRA}")
    for cfg in opt["--cfg"].split(","):
        r = run(cfg)
        if base is None: base = r
        print(f"  {cfg:20s} {r[0]:6.1f} ms/pass ({r[0] / base[0] - 1:+6.1%}), file wait {r[5]:5.1f} ms/pass, "
              f"{r[1]:5.1f} misses/pass, covered {r[2]:5.1%}, read-ahead {r[3]:5.1f}/pass ({r[4]:4.1f} wasted), "
              f"device busy {(r[1] + r[4]) * rm / r[0]:5.1%}  [{r[6]:.2f} rows/pass]")


if __name__ == "__main__":
    main()
