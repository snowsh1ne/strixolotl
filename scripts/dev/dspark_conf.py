#!/usr/bin/env python3
# dspark_conf.py NAME.. - stall-free DSpark numbers from STRATA_DSPARK_CONFLOG (runs/dspark/NAME.conf: "k acc verify_ms
# p0..p4" per pass): passes over 1 s (KFD stalls) and the first 8 passes dropped; est tok/s = tokens/pass / (median
# verify ms + draft ms).  Draft ms from the log's DSpark line.
import os, re, statistics as S, sys
D = os.path.expanduser("~/.cache/xolotl/runs/dspark")
for n in sys.argv[1:]:
    r = [l.split() for l in open(f"{D}/{n}.conf")]
    v = [(int(x[0]), int(x[1]), float(x[2])) for x in r][8:]
    ok = [x for x in v if x[2] < 1000]
    log = open(f"{D}/{n}.log", errors="replace").read()
    dm = re.search(r"draft ([\d.]+) ms/pass", log); dm = float(dm[1]) if dm else 15.0
    fm = re.search(r"^decode: (\d+) tokens .*? (\d+) file", log, re.M)
    tpp = 1 + S.mean(x[1] for x in ok); med = S.median(x[2] for x in ok)
    print(f"{n:12} passes {len(v)} (stalled {len(v) - len(ok)})  tokens/pass {tpp:.2f}  median verify {med:.0f} ms  "
          f"est {1000 * tpp / (med + dm):.2f} tok/s  file/tok {int(fm[2]) / int(fm[1]):.2f}" if fm else "")
