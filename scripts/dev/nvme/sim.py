#!/usr/bin/env python3
"""sim.py TRACE [--from STEP] [--read-ms 5.3] [--layer-ms 2.0] [--post-ms 0.5] [--policies lru,opt,lfu,lru-global]
         [--spec "k:n,k:n,..."] - offline simulation on a STRATA_DEC_TRACE decode routing trace (ds41_generate.cpp;
the routing is exact, so neither the cache policy nor read-ahead can change it):

  1. cache policies for the RAM arena, from the traced seed: 'lru' = the engine's arena_adapt (per layer, evict the
     least-recently-used expert not routed now; validates against the engine's file count), 'lru-global' (victim from
     any layer), 'lfu' (decayed use counts), 'opt' (Belady per layer: the lower bound at the same per-layer capacity).
  2. read-ahead with the traced predictions ('P' records, STRATA_FILE_PRED=1): at layer l's start the top-n predicted
     experts of layer l+k that are neither in the arena nor already stashed are read into a stash (one stash entry per
     target call, dropped after it); a routed miss found in the stash waits for that read instead of issuing its own.
     Timing: per layer `layer-ms` of compute before the routing is known (dense + router), `post-ms` after it (arena
     experts + sum), one device serving blobs back to back at `read-ms` each, demand reads first, no preemption.
"""
import struct, sys, collections
EXTRA = 0

def load(path):
    b = open(path, "rb").read()
    o = 0
    arena0 = None; R = []; P = {}
    step = -1; nl = ne = 0
    while o + 4 <= len(b):
        tag, = struct.unpack_from("<i", b, o)
        need = {ord('A'): 12, ord('R'): 56, ord('P'): 76, ord('O'): 16}.get(tag, 4)
        if o + need > len(b): break   # a trace still being written
        if tag == ord('A'):
            nl, ne = struct.unpack_from("<ii", b, o + 4); o += 12
            arena0 = bytearray(b[o:o + nl * ne]); o += nl * ne
        elif tag == ord('R'):
            l, = struct.unpack_from("<i", b, o + 4); o += 8
            ids = struct.unpack_from("<6i", b, o); o += 24
            w = struct.unpack_from("<6f", b, o); o += 24
            if l == 0: step += 1
            R.append((step, l, ids, w))
        elif tag == ord('P'):
            l, k = struct.unpack_from("<ii", b, o + 4); o += 12
            ids = struct.unpack_from("<16i", b, o); o += 64
            P[(step + 1 if l == 0 else step, l, k)] = ids   # written at layer l's start, before its 'R'
        elif tag == ord('O'):
            o += 16
        else:
            raise SystemExit(f"bad tag {tag} at {o}")
    if R and len(R) % nl: R = R[: len(R) - len(R) % nl]   # a cut-off last step
    return nl, ne, arena0, R, P

class Arena:
    def __init__(s, nl, ne, arena0, policy, R=None):
        s.nl, s.ne, s.pol = nl, ne, policy
        s.res = [set(e for e in range(ne) if arena0[l * ne + e]) for l in range(nl)]
        tot = sum(len(r) for r in s.res)
        s.cap = [len(r) + (EXTRA * len(r) + tot // 2) // tot for r in s.res]   # --extra N: free slots, spread like the seed
        s.last = {}; s.fr = {}; s.lastmiss = {}; s.tick = 0; s.lc = [0] * nl
        s.dec = float(policy.partition(':')[2] or 1) if policy.split(':')[0] in ('tlfu', 'lfu', 'lfun', 'lfur') else 1.0
        if policy == "opt":
            s.future = collections.defaultdict(collections.deque)
            for i, (st, l, ids, w) in enumerate(R):
                for e in ids: s.future[(l, e)].append(i)
    def has(s, l, e): return e in s.res[l]
    def fq(s, l, e):   # decayed use count (per call of layer l)
        v = s.fr.get((l, e))
        if not v: return 0.0
        return v[0] * s.dec ** (s.lc[l] - v[1])
    def call(s, l, ids):
        """one gpu_run: returns the miss mask; admits misses per the policy.  Policies: lru (the engine), opt (Belady),
        static (never adapt), adm:W (lru, but a miss is admitted only if it also missed within the layer's last W
        calls), tlfu:D (lru victim, admitted only if its decayed use count beats the victim's; decay D per layer call),
        lfu:D (victim = lowest decayed count, admitted only if higher), lru-global (victim from any layer)"""
        s.tick += 1
        s.lc[l] += 1
        kind, _, arg = s.pol.partition(":")
        if kind == "opt":
            for e in ids: s.future[(l, e)].popleft()
        if kind in ("glfu", "glfuadm"):
            import math, heapq
            if not hasattr(s, "gk"): s.gk = {}; s.gh2 = []; s.ld = -math.log(float(arg)); 
            for e in ids:   # key = log(sum over uses of dec^-t_use) = log of the decayed count + const(T): order-stable
                k0 = s.gk.get((l, e))
                x = s.tick * s.ld
                s.gk[(l, e)] = x if k0 is None or k0 < -1e17 else max(k0, x) + math.log1p(math.exp(-abs(k0 - x)))
        for e in ids:
            s.last[(l, e)] = s.tick
            if kind in ("tlfu", "lfu", "lfun", "lfur"): s.fr[(l, e)] = (s.fq(l, e) + 1.0, s.lc[l])
        mm = [e not in s.res[l] for e in ids]
        taken = set()
        for e, m in zip(ids, mm):
            if not m: continue
            if kind == "static": continue
            if kind == "adm":
                pm = s.lastmiss.get((l, e))
                s.lastmiss[(l, e)] = s.lc[l]
                if pm is None or s.lc[l] - pm > int(arg): continue
            vl, best, bu = l, None, None
            if kind not in ("adm",) and len(s.res[l]) < s.cap[l]:
                s.res[l].add(e); taken.add(e); continue
            if kind in ("glfu", "glfuadm"):   # global decayed-count victim (decay per gpu_run call): static heap keys
                import heapq, math
                if not hasattr(s, "gk"):
                    s.gk = {}; s.gh2 = []
                    s.ld = -math.log(float(arg))
                    for ll in range(s.nl):
                        for v in s.res[ll]:
                            k0 = s.gk.get((ll, v), -1e18); s.gh2.append((k0, ll, v))
                    heapq.heapify(s.gh2)
                skipped = []
                while s.gh2:
                    k0, ll, v = heapq.heappop(s.gh2)
                    if v not in s.res[ll] or s.gk.get((ll, v), -1e18) != k0: continue
                    if ll == l and (s.last.get((ll, v), -1) >= s.tick or v in taken): skipped.append((k0, ll, v)); continue
                    best, vl = v, ll; break
                for x in skipped: heapq.heappush(s.gh2, x)
                if best is not None and kind == "glfuadm" and not s.gk.get((l, e), -1e18) > s.gk.get((vl, best), -1e18):
                    heapq.heappush(s.gh2, (s.gk.get((vl, best), -1e18), vl, best)); continue
            elif kind == "lru-global":   # lazy heap of (last use, layer, expert) over every resident
                import heapq
                if not hasattr(s, "gh"):
                    s.gh = [(s.last.get((ll, v), -1), ll, v) for ll in range(s.nl) for v in s.res[ll]]
                    heapq.heapify(s.gh)
                skipped = []
                while s.gh:
                    u, ll, v = heapq.heappop(s.gh)
                    if v not in s.res[ll] or s.last.get((ll, v), -1) != u: continue   # stale entry
                    if ll == l and (u >= s.tick or v in taken): skipped.append((u, ll, v)); continue
                    best, vl = v, ll; break
                for x in skipped: heapq.heappush(s.gh, x)
            else:
                for v in sorted(s.res[l]):
                    if s.last.get((l, v), -1) >= s.tick or v in taken: continue
                    if kind in ("lru", "adm", "tlfu"): key = s.last.get((l, v), -1)
                    elif kind in ("lfu", "lfun"): key = s.fq(l, v)
                    elif kind == "lfur": key = (s.fq(l, v), s.last.get((l, v), -1))
                    else:
                        q = s.future.get((l, v)); key = -(q[0] if q else 10**12)
                    if bu is None or key < bu: bu, best = key, v
            if best is None: continue
            if kind in ("tlfu", "lfu") and not s.fq(l, e) > s.fq(l, best): continue
            s.res[vl].discard(best); s.res[l].add(e); taken.add(e)
        if kind in ("glfu", "glfuadm") and hasattr(s, "gh2"):
            import heapq
            for e in ids:
                if e in s.res[l]: heapq.heappush(s.gh2, (s.gk[(l, e)], l, e))
        if kind == "lru-global" and hasattr(s, "gh"):
            import heapq
            for e in ids:
                if e in s.res[l]: heapq.heappush(s.gh, (s.tick, l, e))
        return mm

class Device:
    """blobs back to back; demand (prio 0) before spec (prio 1); a started read runs to the end"""
    def __init__(s, read_ms): s.rm, s.free, s.q, s.n, s.done = read_ms, 0.0, [], 0, {}
    def add(s, key, prio, t):
        s.n += 1; s.q.append([prio, s.n, key, t])
    def promote(s, key):
        for r in s.q:
            if r[2] == key: r[0] = 0
    def run_until(s, t):   # start reads while the device frees up before t
        while s.q:
            s.q.sort()
            r = s.q[0]
            st = max(s.free, r[3])
            if st > t: break
            s.q.pop(0); s.free = st + s.rm; s.done[r[2]] = s.free
    def wait(s, keys, t):
        keys = [k for k in keys if k not in s.done or s.done[k] > t]
        while any(k not in s.done for k in keys):
            s.q.sort()
            r = s.q.pop(0)
            st = max(s.free, r[3]); s.free = st + s.rm; s.done[r[2]] = s.free
        return max([t] + [s.done[k] for k in keys])

def timing(nl, ne, arena0, R, P, start, k_n, read_ms, layer_ms, post_ms):
    A = Arena(nl, ne, arena0, "lru")
    D = Device(read_ms)
    t = 0.0; t_start = None; nw = 0
    stash = {}   # (step, l, e) -> read key
    cov = iss = miss = 0
    steps = R[-1][0] + 1
    i = 0
    for (s, l, ids, w) in R:
        if l == 0 and s == start: t_start = t
        # spec reads for later layers, issued at this layer's start
        for (k, n) in k_n:
            if l + k >= nl: continue
            pk = P.get((s, l, k))
            if not pk: continue
            for e in pk[:n]:
                if e < 0 or A.has(l + k, e) or (s, l + k, e) in stash: continue
                key = ("s", s, l + k, e); stash[(s, l + k, e)] = key
                D.add(key, 1, t)
                if s >= start: iss += 1
        t += layer_ms
        D.run_until(t)
        mm = A.call(l, ids)
        need = []
        for e, m in zip(ids, mm):
            if not m: continue
            key = stash.get((s, l, e))
            if key is not None:
                D.promote(key); need.append(key)
                if s >= start: cov += 1
            else:
                key = ("d", s, l, e); D.add(key, 0, t); need.append(key)
            if s >= start: miss += 1
        t = D.wait(need, t)
        t += post_ms
        for e in range(0):
            pass
        # drop this call's stash entries (used or not); their reads still occupy the device if queued: drop queued ones
        dead = [x for x in stash if x[0] == s and x[1] == l]
        for x in dead:
            key = stash.pop(x)
            D.q = [r for r in D.q if r[2] != key]
        if l == nl - 1 and s >= start: nw += 1
    tok = (t - t_start) / max(nw, 1) if t_start is not None else 0
    return tok, miss / max(nw, 1), cov / max(miss, 1), iss / max(nw, 1)

def main():
    a = sys.argv[1:]
    path = a[0]
    opt = {"--from": "1024", "--read-ms": "5.3", "--layer-ms": "2.0", "--post-ms": "0.5", "--policies": "lru,opt,lfu",
           "--spec": "1:1,1:2,1:4,2:1,2:2,2:4,3:2,4:2,1:2+2:2,1:1+2:1+3:1,1:2+2:2+3:2"}
    i = 1
    while i < len(a): opt[a[i]] = a[i + 1]; i += 2
    start = int(opt["--from"])
    global EXTRA
    EXTRA = int(opt.get("--extra", 0))
    nl, ne, arena0, R, P = load(path)
    nsteps = R[-1][0] + 1
    print(f"trace: {nsteps} steps x {nl} layers, arena {sum(arena0)} experts, {len(P)} prediction records; window steps >= {start}")
    for pol in [p for p in opt["--policies"].split(",") if p]:
        A = Arena(nl, ne, arena0, pol, R)
        m = mw = nw = 0
        for (s, l, ids, w) in R:
            mm = A.call(l, ids); m += sum(mm)
            if s >= start: mw += sum(mm); nw += (l == 0)
        print(f"  policy {pol:10s}: misses {m / nsteps:.2f}/token over the run, window {mw / max(nw, 1):.2f}/token")
    rm, lm, pm = float(opt["--read-ms"]), float(opt["--layer-ms"]), float(opt["--post-ms"])
    base = timing(nl, ne, arena0, R, P, start, [], rm, lm, pm)
    print(f"  timing model (read {rm} ms/blob, layer {lm} + {pm} ms): no read-ahead {base[0]:.1f} ms/token "
          f"({1000 / base[0]:.2f} tok/s), {base[1]:.2f} misses/token")
    if not P: return
    for cfg in [c for c in opt["--spec"].split(",") if c]:
        k_n = [tuple(int(x) for x in c.split(":")) for c in cfg.split("+")]
        tok, mpt, cv, iss = timing(nl, ne, arena0, R, P, start, k_n, rm, lm, pm)
        print(f"    spec {cfg:14s}: {tok:6.1f} ms/token ({1000 / tok:5.2f} tok/s, {tok / base[0] - 1:+6.1%})  covers {cv:5.1%} of "
              f"the misses, issues {iss:5.1f} reads/token (device {(iss + mpt * (1 - cv)) * rm / tok:5.1%} busy)")

if __name__ == "__main__":
    main()
