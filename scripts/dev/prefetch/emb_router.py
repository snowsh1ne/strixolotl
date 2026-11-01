#!/usr/bin/env python3
"""emb_router.py GGUF IDS OUT.npz - the token-only routing guess for pred.py's emb.N predictor: for every distinct token
id in IDS (one per line), rms_norm(token_embd[tok]) * ffn_norm_L -> layer L's router (sqrt(softplus) + exp_probs_b),
top 16 per layer.  No context at all (no attention, no hyper-connection mix): what a read issued from the draft token
ids alone, before the verify pass starts, could know.  Reads ~0.3 GB of the GGUF (routers + the needed embedding
rows).  python3 -B emb_router.py ...
"""
import sys
import struct
import numpy as np


def gguf_tensors(path):
    """minimal GGUF v3 reader (gguf-py rejects the engine's custom expert types): {name: (type, dims, abs offset)}"""
    f = open(path, 'rb')
    def rd(fmt): return struct.unpack('<' + fmt, f.read(struct.calcsize('<' + fmt)))
    def rstr(): n, = rd('Q'); return f.read(n).decode('utf-8', 'replace')
    sz = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i', 6: 'f', 7: '?', 10: 'Q', 11: 'q', 12: 'd'}
    def rval(t):
        if t == 8: return rstr()
        if t == 9:
            it, n = rd('IQ')
            return [rval(it) for _ in range(n)]
        return rd(sz[t])[0]
    magic, ver, nt, nkv = rd('IIQQ')
    align = 32
    for _ in range(nkv):
        k = rstr(); t, = rd('I'); v = rval(t)
        if k == 'general.alignment': align = v
    info = {}
    for _ in range(nt):
        name = rstr(); nd, = rd('I'); dims = rd('Q' * nd); typ, off = rd('IQ')
        info[name] = (typ, dims, off)
    base = (f.tell() + align - 1) // align * align
    return {k: (t, d, base + o) for k, (t, d, o) in info.items()}


def main():
    gg, ids_path, out = sys.argv[1:4]
    toks = sorted(set(int(x) for x in open(ids_path) if x.strip().lstrip('-').isdigit()))
    T = gguf_tensors(gg)
    fh = open(gg, 'rb')
    dt = {0: np.float32, 1: np.float16}

    def get(name):
        typ, dims, off = T[name]
        assert typ in dt, (name, typ)
        n = int(np.prod(dims))
        fh.seek(off)
        return np.frombuffer(fh.read(n * np.dtype(dt[typ]).itemsize), dt[typ]).astype(np.float32).reshape(list(reversed(dims)))
    typ, dims, off = T["token_embd.weight"]
    assert typ in dt, typ
    D = int(dims[0]); isz = np.dtype(dt[typ]).itemsize
    E = np.zeros((len(toks), D), np.float32)
    for i, t in enumerate(toks):
        fh.seek(off + t * D * isz)
        E[i] = np.frombuffer(fh.read(D * isz), dt[typ])
    nl = 1 + max(int(n.split(".")[1]) for n in T if n.startswith("blk.") and n.endswith("ffn_gate_inp.weight"))
    En = E / np.sqrt((E * E).mean(1, keepdims=True) + 1e-6)
    top = np.zeros((len(toks), nl, 16), np.int32)
    for l in range(nl):
        W = get(f"blk.{l}.ffn_gate_inp.weight")          # (NE, D)
        nw = get(f"blk.{l}.ffn_norm.weight").reshape(D)
        b = get(f"blk.{l}.exp_probs_b.bias").reshape(-1)
        z = (En * nw) @ W.T
        sc = np.sqrt(np.logaddexp(0, z)) + b
        top[:, l, :] = np.argsort(-sc, 1)[:, :16]
    np.savez(out, tok=np.asarray(toks, np.int32), top=top)
    print(f"{len(toks)} tokens x {nl} layers -> {out} (D {D})")


if __name__ == "__main__":
    main()
