#!/usr/bin/env python3
# dspark_cmp.py REF RUN.. - runs/dspark/<name>.log: the token streams (stdout = one bare id per line) against REF's
# (first divergence, matching prefix), decode tok/s, file-tier reads per token, the DSpark line (tokens/pass).
import os, re, sys
D = os.path.expanduser("~/.cache/xolotl/runs/dspark")

def load(n):
    txt = open(f"{D}/{n}.log", errors="replace").read()
    ids = [int(l) for l in txt.splitlines() if re.fullmatch(r"\d+", l)]
    dec = re.search(r"^decode: (\d+) tokens in ([\d.]+) s = ([\d.]+) tok/s; experts \d+ hit / \d+ cpu / (\d+) pcie / (\d+) file", txt, re.M)
    dsp = re.search(r"^DSpark: .*?([\d.]+) tokens/pass, ([\d.]+)% full.*?verify ([\d.]+) ms/pass, draft ([\d.]+) ms/pass", txt, re.M)
    ext = re.search(r"^arena ext: (\+\d+ slots \([\d.]+ GiB\)|skipped.*)", txt, re.M)
    return ids, dec, dsp, ext

ref = load(sys.argv[1])[0]
print(f"{'run':14} {'tokens':>6} {'tok/s':>6} {'file/tok':>8} {'tok/pass':>8} {'verify ms':>9} {'match':>12}  arena ext")
for n in sys.argv[1:]:
    ids, dec, dsp, ext = load(n)
    m = next((i for i, (a, b) in enumerate(zip(ids, ref)) if a != b), min(len(ids), len(ref)))
    same = "identical" if ids == ref else f"diverge@{m}"
    if not dec:
        print(f"{n:14} {len(ids):6} (no decode line)  {same}")
        continue
    g, s, tps, pcie, fil = int(dec[1]), float(dec[2]), float(dec[3]), int(dec[4]), int(dec[5])
    print(f"{n:14} {g:6} {tps:6.2f} {fil / max(g, 1):8.2f} {dsp[1] if dsp else '-':>8} {dsp[3] if dsp else '-':>9} {same:>12}  "
          f"{ext[1] if ext else '-'}")
