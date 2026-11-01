#!/usr/bin/env python3
"""ktsum.py TRACE_DIR [N] - GPU time by kernel from a rocprofv3 --kernel-trace CSV (docs/PREFILL.md dense profile).

Splits the kernels into the routed-expert tier (Strata's native_* / mmq / swiglu-entries / weighted-sum kernels) and the
dense half (everything ggml runs), prints the top N kernels of each by summed duration, and the GPU-busy time.
python3 -B (no __pycache__ on the share).
"""
import collections
import csv
import glob
import re
import sys

# (the dense half's Q8_0 GEMMs are ggml's mul_mat_q / quantize_mmq_q8_1 and count as dense; with --chunk-mmq the
# expert products are mul_mat_q too and land in the dense list - profile that mode with care)
EXPERT = re.compile(r"strata::kernels::|native_|swiglu_entries|swiglu_q8_1_entries|weighted_rows|prom_|ptq1_to_q2")


def main():
    d = sys.argv[1]
    top = int(sys.argv[2]) if len(sys.argv) > 2 else 25
    f = glob.glob(f"{d}/**/*kernel_trace.csv", recursive=True)
    if not f:
        sys.exit("no kernel_trace.csv under " + d)
    agg = {"expert": collections.defaultdict(lambda: [0, 0]), "dense": collections.defaultdict(lambda: [0, 0])}
    spans = []
    for fn in f:
        for r in csv.DictReader(open(fn)):
            s, e = int(r["Start_Timestamp"]), int(r["End_Timestamp"])
            name = re.sub(r"\(.*", "", r["Kernel_Name"])
            name = re.sub(r"^void ", "", name)[:100]
            cat = "expert" if EXPERT.search(r["Kernel_Name"]) else "dense"
            agg[cat][name][0] += e - s
            agg[cat][name][1] += 1
            spans.append((s, e))
    spans.sort()
    busy, cur = 0, 0
    for s, e in spans:
        if e > cur:
            busy += e - max(s, cur)
            cur = e
    print(f"kernels {len(spans)}, GPU busy {busy / 1e9:.2f} s, wall {(spans[-1][1] - spans[0][0]) / 1e9:.2f} s")
    for cat in ("dense", "expert"):
        tot = sum(v[0] for v in agg[cat].values())
        n = sum(v[1] for v in agg[cat].values())
        print(f"== {cat}: {tot / 1e9:.2f} s in {n} kernels")
        for k, v in sorted(agg[cat].items(), key=lambda kv: -kv[1][0])[:top]:
            print(f"  {v[0] / 1e9:8.3f} s {100 * v[0] / max(tot, 1):5.1f}% {v[1]:7d}x {v[0] / max(v[1], 1) / 1e3:9.1f} us  {k}")


if __name__ == "__main__":
    main()
