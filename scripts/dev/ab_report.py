# ab_report.py - report for ab_v1v2.sh (runs/ab): quality v1 vs v2 against the teacher on every held-out row, paired
# per row with a bootstrap CI over rows, and the speed A/B (decode tok/s, file-tier misses/token, decode stage timers).
import json, re, os, numpy as np
from collections import defaultdict
A = os.path.expanduser("~/.cache/xolotl/runs/ab/")
rows = json.load(open(A + "rows.json"))
ld = lambda p: np.loadtxt(p) if os.path.exists(p) and os.path.getsize(p) else None

print("== quality: per-token NLL vs the teacher (all positions of each 8K held-out row)")
print(f"{'row':8s} {'domain':13s} {'T ppl':>8s} {'v1 gap':>8s} {'v2 gap':>8s} {'v2-v1':>8s} {'top1=T v1/v2':>13s}")
per, pool = [], defaultdict(lambda: defaultdict(list))
for r in rows:
    T, a, b = (ld(f"{A}nll-{k}-{r['file']}.txt") for k in ("T", "v1", "v2"))
    if T is None or a is None or b is None or not (len(T) == len(a) == len(b)):
        print(f"{r['file']:8s} incomplete"); continue
    tn, ta = T[:, 1], T[:, 2].astype(int)
    g = lambda x: np.exp(x[:, 1].mean() - tn.mean()) - 1
    agree = lambda x: (x[:, 2].astype(int) == ta).mean()
    d = b[:, 1].mean() - a[:, 1].mean()
    per.append(dict(r, d=d, g1=g(a), g2=g(b), mad1=np.abs(a[:, 1] - tn).mean(), mad2=np.abs(b[:, 1] - tn).mean(),
                    ag1=agree(a), ag2=agree(b)))
    for key in ("ALL", r["chain"] + "-set", r["domain"]):
        for k, x in (("T", tn), ("v1", a[:, 1]), ("v2", b[:, 1])): pool[key][k].append(x)
        pool[key]["ag1"].append(a[:, 2].astype(int) == ta); pool[key]["ag2"].append(b[:, 2].astype(int) == ta)
    print(f"{r['file']:8s} {r['domain']:13s} {np.exp(tn.mean()):8.3f} {g(a):+8.2%} {g(b):+8.2%} {np.exp(d) - 1:+8.2%} "
          f"{agree(a):.3f}/{agree(b):.3f}")
if per:
    print("\n== pooled (token-weighted): ppl gap to the teacher, top-1 agreement with the teacher")
    for key, P in pool.items():
        c = {k: np.concatenate(v) for k, v in P.items()}
        gap = lambda k: np.exp(c[k].mean() - c["T"].mean()) - 1
        print(f"  {key:14s} rows {len(P['T']):2d}  v1 {gap('v1'):+7.2%}  v2 {gap('v2'):+7.2%}  "
              f"v2 vs v1 ppl {np.exp(c['v2'].mean() - c['v1'].mean()) - 1:+7.2%}  top1=T {c['ag1'].mean():.4f} -> {c['ag2'].mean():.4f}")
    d = np.array([p["d"] for p in per]); rng = np.random.default_rng(0)
    bs = np.array([d[rng.integers(len(d), size=len(d))].mean() for _ in range(20000)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print(f"\n== paired per row (mean NLL v2 - v1; < 0 = v2 better): v2 better on {(d < 0).sum()}/{len(d)} rows; "
          f"mean {np.exp(d.mean()) - 1:+.2%} ppl, 95% bootstrap CI over rows [{np.exp(lo) - 1:+.2%}, {np.exp(hi) - 1:+.2%}]")
    print(f"   mean |NLL - T| per token: v1 {np.mean([p['mad1'] for p in per]):.4f}  v2 {np.mean([p['mad2'] for p in per]):.4f}")

print("\n== extras vs v2 (ab_extra.sh, ab_speed.sh; GSQ-217 + v2 switches), on the rows each config has")
for cfg in ("v2rep", "p03", "p06", "q6", "q6p005", "q6p010", "q6p015", "q6p020", "q5", "q6f"):
    d, cT, c2, ag, ag2, same, doms = [], [], [], [], [], 0, defaultdict(list)
    for r in rows:
        T, b, x = (ld(f"{A}nll-{k}-{r['file']}.txt") for k in ("T", "v2", cfg))
        if T is None or b is None or x is None or not (len(T) == len(b) == len(x)): continue
        same += np.array_equal(b, x); d.append(x[:, 1].mean() - b[:, 1].mean()); doms[r["domain"]].append(d[-1])
        cT.append(T[:, 1]); c2.append(b[:, 1]); ag.append(x[:, 2].astype(int) == T[:, 2].astype(int))
        ag2.append(b[:, 2].astype(int) == T[:, 2].astype(int)); c2[-1] = (b[:, 1], x[:, 1])
    if not d: continue
    d = np.array(d); tn = np.concatenate(cT); bn = np.concatenate([p[0] for p in c2]); xn = np.concatenate([p[1] for p in c2])
    rng = np.random.default_rng(0); bs = np.array([d[rng.integers(len(d), size=len(d))].mean() for _ in range(20000)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print(f"  {cfg:5s} rows {len(d):2d}: gap to T {np.exp(xn.mean() - tn.mean()) - 1:+7.2%} (v2 {np.exp(bn.mean() - tn.mean()) - 1:+7.2%}); "
          f"vs v2 ppl {np.exp(d.mean()) - 1:+7.2%} CI [{np.exp(lo) - 1:+.2%}, {np.exp(hi) - 1:+.2%}], worse on {(d > 0).sum()}/{len(d)}; "
          f"top1=T {np.concatenate(ag2).mean():.4f} -> {np.concatenate(ag).mean():.4f}" + (f"; identical dumps {same}/{len(d)}" if cfg == "v2rep" else ""))
    if cfg != "v2rep":
        print("        per domain vs v2: " + ", ".join(f"{k} {np.exp(np.mean(v)) - 1:+.1%}" for k, v in sorted(doms.items())))

print("\n== speed (6144-token prompt + 1000 forced tokens, arena 86 GiB, --arena-adapt)")
dec = re.compile(r"decode: (\d+) tokens in ([\d.]+) s = ([\d.]+) tok/s; experts (\d+) hit / (\d+) cpu / (\d+) pcie / (\d+) file")
stg = re.compile(r"decode step ms/token: (.*) \((\d+) steps\)")
res = defaultdict(list)
for f in sorted(os.listdir(A)):
    m = re.match(r"s-(v1|v2|m|p03|p06|q6)-(\w+?)(-b)?\.log$", f)
    if not m: continue
    t = open(A + f, errors="ignore").read(); dm = dec.findall(t); sm = stg.findall(t)
    if not dm: print(f"  {f}: no decode line"); continue
    n, s, tps, _, _, _, fi = dm[-1]
    res[m[1]].append((m[2], float(tps), int(fi) / int(n)))
    print(f"  {m[1]:2s} {m[2]:11s}{m[3] or '  ':2s} {float(tps):5.2f} tok/s  file misses/token {int(fi) / int(n):5.1f}  "
          f"stages {sm[-1][0] if sm else '-'}")
for k in ("v1", "m", "v2", "p03", "p06", "q6"):
    if res[k]:
        print(f"  mean {k:2s}: {np.mean([x[1] for x in res[k]]):.2f} tok/s, {np.mean([x[2] for x in res[k]]):.1f} file misses/token")

print("\n== warm speed (ab_speed.sh: ~15K-token held-out agent prompt + 3000 forced tokens, --arena-adapt; arena 86 GiB")
print("   unless rNN = --uma-reserve-gib NN).  Window = decode steps 1024..end (the arena warms in decode); ms/token by")
print("   stage from the running averages of the STRATA_DEC_PROF lines; tok/s = 1000 / the window's summed stages")
prof = re.compile(r"decode step ms/token: begin ([\d.]+) predict ([\d.]+) attn_router ([\d.]+) tier ([\d.]+) finish ([\d.]+) "
                  r"spec ([\d.]+) \((\d+) steps\) file (\d+)")
WD = A + "warm/"
def window(P, a, b):   # stage ms/token and file misses/token over steps P[a].n .. P[b].n
    na, nb = P[a][6], P[b][6]
    st = [(P[b][i] * nb - P[a][i] * na) / (nb - na) for i in range(6)]
    return st, (P[b][7] - P[a][7]) / (nb - na)
wres = {}
for cfg in ("v2", "v2b", "q6", "q6p005", "q6p010", "q6p015", "q6p020", "q5", "q6f", "q6r18", "q6r16"):
    f = f"{WD}s-{cfg}.log"
    if not os.path.exists(f): continue
    t = open(f, errors="ignore").read()
    P = [tuple(float(x) for x in m[:6]) + (int(m[6]), int(m[7])) for m in prof.findall(t)]
    dm = dec.findall(t); ar = re.search(r"tier: .*?(\d+) in the RAM arena \(([\d.]+) GiB\)", t)
    pm = re.search(r"prompt: (\d+) tokens in ([\d.]+) s", t)
    ml = [l.split() for l in open(f[:-4] + ".mem")] if os.path.exists(f[:-4] + ".mem") else []
    mem = [int(x[1]) for x in ml if x[1].isdigit()]; temp = max([int(x[3]) for x in ml if len(x) > 3 and x[3].isdigit()] or [0])
    if not dm or len(P) < 20: print(f"  {cfg:7s} incomplete ({len(P)} prof lines)"); continue
    a = next(i for i, p in enumerate(P) if p[6] >= 1024)
    st, miss = window(P, a, len(P) - 1)
    wres[cfg] = (1000 / sum(st), miss, st)
    print(f"  {cfg:7s} warm {1000 / sum(st):5.2f} tok/s  file misses/token {miss:5.1f}  ms/token: attn_router {st[2]:5.1f} "
          f"tier {st[3]:5.1f} other {st[0] + st[1] + st[4] + st[5]:4.1f}  | whole decode {float(dm[-1][2]):5.2f} tok/s; arena "
          f"{ar[2] if ar else '?'} GiB ({ar[1] if ar else '?'} experts); prefill {float(pm[1]) / float(pm[2]):4.1f} tok/s; "
          f"min MemAvailable {min(mem) / 1024 if mem else 0:4.1f} GiB; NVMe max {temp} C")
    curve = []
    for i in range(0, len(P) - 4, 4):   # 256-step windows
        s2, m2 = window(P, i, i + 4); curve.append(f"{m2:4.1f}/{s2[3]:3.0f}")
    print(f"          warm-up, per 256 steps (misses/token / tier ms): {' '.join(curve)}")
if "v2" in wres:
    b = wres["v2"]
    print("  vs v2 (warm window): " + ", ".join(f"{k} {v[0] / b[0] - 1:+.0%} tok/s ({v[1] - b[1]:+.1f} misses)" for k, v in wres.items() if k != "v2"))
