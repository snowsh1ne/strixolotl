import csv, sys, bisect, collections, re
d = sys.argv[1]; phase = sys.argv[2] if len(sys.argv) > 2 else "attn_router"
skip = int(sys.argv[3]) if len(sys.argv) > 3 else 16
mk = [r for r in csv.DictReader(open(d + "/run_marker_api_trace.csv"))]
begins = [int(r["Start_Timestamp"]) for r in mk if r["Function"] == "begin"]
t0 = begins[skip]; t1 = begins[-1]   # steps skip .. last-1
ntok = len(begins) - 1 - skip
rng = sorted((int(r["Start_Timestamp"]), int(r["End_Timestamp"])) for r in mk if r["Function"] == phase and t0 <= int(r["Start_Timestamp"]) < t1)
ks = []
for r in csv.DictReader(open(d + "/run_kernel_trace.csv")):
    s = int(r["Start_Timestamp"])
    if t0 <= s < t1: ks.append((s, int(r["End_Timestamp"]), r["Kernel_Name"], r["Grid_Size_X"], r["Grid_Size_Y"], r["Grid_Size_Z"], r["Workgroup_Size_X"]))
ks.sort()
starts = [k[0] for k in ks]
wall = sum(e - s for s, e in rng)
agg = collections.defaultdict(lambda: [0, 0]); aggg = collections.defaultdict(lambda: [0, 0])
ksum = 0; busy = 0; nk = 0; gaps = []
for s, e in rng:
    i = bisect.bisect_left(starts, s); j = bisect.bisect_left(starts, e)
    last = None; first = None
    cur_end = s
    for k in ks[i:j]:
        nk += 1; dur = k[1] - k[0]; ksum += dur
        if first is None: first = k[0]
        busy += max(0, k[1] - max(k[0], cur_end)); cur_end = max(cur_end, k[1])
        name = re.sub(r"\(.*", "", k[2]); name = name[:110]
        agg[name][0] += dur; agg[name][1] += 1
        aggg[(name, k[3], k[4], k[5])][0] += dur; aggg[(name, k[3], k[4], k[5])][1] += 1
    if first: gaps.append((first - s, e - cur_end))
T = ntok * 1e6
print(f"phase {phase}: {ntok} tokens, {len(rng)/ntok:.1f} ranges/token, wall {wall/T:.2f} ms/tok, kernel sum {ksum/T:.2f}, busy(union) {busy/T:.2f}, kernels/token {nk/ntok:.0f}")
if gaps:
    print(f"  per range: launch->first kernel {sum(g[0] for g in gaps)/len(gaps)/1e3:.1f} us, last kernel end->range end {sum(g[1] for g in gaps)/len(gaps)/1e3:.1f} us")
for n, (t, c) in sorted(agg.items(), key=lambda x: -x[1][0])[:25]:
    print(f"  {t/T:7.3f} ms/tok {c/ntok:6.0f}/tok {t/c/1e3:8.1f} us/call  {n}")
if len(sys.argv) > 4:
    print("--- by grid")
    for (n, gx, gy, gz), (t, c) in sorted(aggg.items(), key=lambda x: -x[1][0])[:int(sys.argv[4])]:
        print(f"  {t/T:7.3f} ms/tok {c/ntok:5.0f}/tok {t/c/1e3:8.1f} us  grid {gx}x{gy}x{gz}  {n[:80]}")
