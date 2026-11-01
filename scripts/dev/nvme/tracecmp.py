#!/usr/bin/env python3
"""tracecmp.py A.trace B.trace - losslessness check of two STRATA_DEC_TRACE runs of the same forced bench: the routing
('R': ids + f32 weights per step and layer) and the greedy outputs ('O': argmax token + its f32 logit) must be
bit-identical over the common steps ('A' / 'P' records are ignored: the arena seed and predictions may differ)."""
import struct, sys
def recs(p):
    b = open(p, "rb").read(); o = 0; R = []; O = []
    while o + 4 <= len(b):
        t, = struct.unpack_from("<i", b, o)
        if t == ord('A'):
            nl, ne = struct.unpack_from("<ii", b, o + 4); o += 12 + nl * ne
        elif t == ord('R'):
            if o + 56 > len(b): break
            R.append(b[o:o + 56]); o += 56
        elif t == ord('P'): o += 76
        elif t == ord('O'):
            if o + 16 > len(b): break
            O.append(b[o:o + 16]); o += 16
        else: raise SystemExit(f"{p}: bad tag {t} at {o}")
    return R, O
(Ra, Oa), (Rb, Ob) = recs(sys.argv[1]), recs(sys.argv[2])
n, m = min(len(Ra), len(Rb)), min(len(Oa), len(Ob))
dr = next((i for i in range(n) if Ra[i] != Rb[i]), None)
do = next((i for i in range(m) if Oa[i] != Ob[i]), None)
print(f"routing records compared {n} ({n // 40} steps): {'IDENTICAL' if dr is None else f'FIRST DIFF at record {dr} (step {dr // 40}, layer {dr % 40})'}")
print(f"output records compared {m}: {'IDENTICAL' if do is None else f'FIRST DIFF at step {do}'}")
sys.exit(0 if dr is None and do is None and n > 0 and m > 0 else 1)
