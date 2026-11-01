# ab_prep.py OUT - inputs for ab_v1v2.sh: the held-out rows of both calibration chains as int32 id files
# (OUT/rows/<chain>_rNN.i32, 8192 tokens), and speed prompts: the first 6144 tokens of 3 held-out rows
# (OUT/speed/<name>.i32) + the real next 1024 tokens as STRATA_FORCE_IDS files (OUT/speed/<name>.force).
import sys, os, json, numpy as np
OUT = sys.argv[1]; W = os.path.expanduser("~/models/xolotl/v41-work")
CHAINS = {"v1": f"{W}/calib-1017", "v2": f"{W}/calib-1017-v2"}
os.makedirs(f"{OUT}/rows", exist_ok=True); os.makedirs(f"{OUT}/speed", exist_ok=True)
rows = []
for c, d in CHAINS.items():
    T = np.load(f"{d}/tokens.npy"); m = json.load(open(f"{d}/meta.json"))
    for i, (dom, ev) in enumerate(zip(m["domains"], m["is_eval"])):
        if ev:
            T[i].astype("<i4").tofile(f"{OUT}/rows/{c}_r{i:02d}.i32")
            rows.append({"chain": c, "row": i, "domain": dom, "file": f"{c}_r{i:02d}"})
json.dump(rows, open(f"{OUT}/rows.json", "w"), indent=1)
# speed prompts: one agentic_long and one code held-out row of v2, one agentic_long of v1 (different sessions)
pick = [("agentic_v2", "v2", "agentic_long", 0), ("code_v2", "v2", "code", 0), ("agentic_v1", "v1", "agentic_long", 0)]
for name, c, dom, k in pick:
    r = [x for x in rows if x["chain"] == c and x["domain"] == dom][k]
    t = np.fromfile(f"{OUT}/rows/{r['file']}.i32", "<i4")
    t[:6144].tofile(f"{OUT}/speed/{name}.i32")
    open(f"{OUT}/speed/{name}.force", "w").write("\n".join(map(str, t[6144:7168].tolist())) + "\n")
print(len(rows), "held-out rows:", [(r["file"], r["domain"]) for r in rows])
