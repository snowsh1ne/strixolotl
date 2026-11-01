import csv, sys, bisect, collections, re
d = sys.argv[1]
mk = [r for r in csv.DictReader(open(d + "/run_marker_api_trace.csv"))]
begins = [int(r["Start_Timestamp"]) for r in mk if r["Function"] == "begin"]
t0, t1 = begins[16], begins[-1]
rng = sorted((int(r["Start_Timestamp"]), int(r["End_Timestamp"])) for r in mk if r["Function"] == "attn_router" and t0 <= int(r["Start_Timestamp"]) < t1)
ks = sorted((int(r["Start_Timestamp"]), int(r["End_Timestamp"]), re.sub(r"\(.*", "", r["Kernel_Name"])[:40], r["Grid_Size_X"]) for r in csv.DictReader(open(d + "/run_kernel_trace.csv")) if t0 <= int(r["Start_Timestamp"]) < t1)
st = [k[0] for k in ks]
pos = collections.defaultdict(lambda: collections.defaultdict(list))
gapsum = collections.defaultdict(list)
for s, e in rng:
    i, j = bisect.bisect_left(st, s), bisect.bisect_left(st, e)
    seen = collections.Counter()
    for k in ks[i:j]:
        key = (k[2], k[3]); seen[key] += 1
        pos[key][seen[key]].append(k[1] - k[0])
for key in [("void rms_norm_f32<1024, true, false>", "1024"), ("void mul_mat_vec_q<", "163840"), ("void mul_mat_vec_q<", "73728"), ("quantize_q8_1", "5120"), ("dsv4_hc_comb_f32", "256")]:
    print(key, {p: round(sum(v)/len(v)/1e3, 1) for p, v in sorted(pos[key].items())})
