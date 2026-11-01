#!/usr/bin/env python3
"""routes.py ROUTE_LOG - how predictable are a prompt batch's FILE-TIER reads?  (docs/PREFILL.md, layer-ahead reads)

Reads the engine's STRATA_ROUTE_LOG (per batch pass: "layer expert tier count", tier 0 VRAM / 1 arena / 2 file;
"# turn N new D batch" markers).  Per turn: file-tier experts read, and how many of them a predictor would have
named in advance:
  prev   - experts the PREVIOUS turn routed at the same layer (any tier)  -> recall / precision
  hot    - the turn's own file reads sorted by token count: share of reads that serve >= 1/2/4/8 tokens
Per-layer overlap with the previous turn is what a cross-turn prefetch could start before the router runs.
python3 -B (no __pycache__ on the share).
"""
import collections
import sys


def load(path):
    turns, cur = [], None
    for ln in open(path):
        if ln.startswith("#"):
            p = ln.split()
            cur = {"new": int(p[4]), "kind": p[5], "rows": []}
            turns.append(cur)
            continue
        if cur is None:
            continue
        l, e, t, c = map(int, ln.split())
        cur["rows"].append((l, e, t, c))
    return turns


def main():
    turns = load(sys.argv[1])
    prev = None
    print(f"{'turn':>4} {'new':>6} {'file':>6} {'routed':>7} {'prev_rec':>8} {'prev_prec':>9} {'f>=2tok':>8} {'f>=4tok':>8} {'f>=8tok':>8}")
    for i, t in enumerate(turns):
        if t["kind"] != "batch":
            continue
        routed = collections.defaultdict(set)
        files = set()
        fcnt = collections.Counter()
        for l, e, tier, c in t["rows"]:
            routed[l].add(e)
            if tier == 2:
                files.add((l, e))
                fcnt[(l, e)] += c
        nf = len(files)
        if prev is not None and nf:
            pred = {(l, e) for l, s in prev.items() for e in s}
            hit = len(files & pred)
            # prec: share of the previous turn's routed set that this turn read from the file tier (a lower bound on
            # a prefetch's precision: predicted experts already in RAM would not be read at all)
            rec, prec = hit / nf, hit / max(1, len(pred))
        else:
            rec = prec = float("nan")
        ge = lambda k: sum(1 for v in fcnt.values() if v >= k) / max(1, nf)
        nr = sum(len(s) for s in routed.values())
        print(f"{i:>4} {t['new']:>6} {nf:>6} {nr:>7} {rec:>8.2f} {prec:>9.2f} {ge(2):>8.2f} {ge(4):>8.2f} {ge(8):>8.2f}")
        prev = routed


if __name__ == "__main__":
    main()
