import struct, sys, collections
f = open(sys.argv[1], 'rb')
def rd(fmt): return struct.unpack('<' + fmt, f.read(struct.calcsize('<' + fmt)))
assert f.read(4) == b'GGUF'
ver, = rd('I'); nt, nkv = rd('QQ')
def rstr(): n, = rd('Q'); return f.read(n).decode('utf8', 'replace')
SZ = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
def rval(t):
    if t == 8: return rstr()
    if t == 9:
        at, n = rd('IQ')
        return [rval(at) for _ in range(n)] if at in (8, 9) else f.read(SZ[at] * n) and None
    fm = {0:'B',1:'b',2:'H',3:'h',4:'I',5:'i',6:'f',7:'?',10:'Q',11:'q',12:'d'}[t]
    return rd(fm)[0]
kv = {}
for _ in range(nkv):
    k = rstr(); t, = rd('I'); kv[k] = rval(t)
ts = []
for _ in range(nt):
    n = rstr(); nd, = rd('I'); dims = rd('Q' * nd); ty, = rd('I'); off, = rd('Q'); ts.append((n, dims, ty, off))
ts.sort(key=lambda x: x[3])
pat = sys.argv[2] if len(sys.argv) > 2 else 'blk.5.'
tot = collections.Counter()
for i, (n, dims, ty, off) in enumerate(ts):
    nb = (ts[i + 1][3] - off) if i + 1 < len(ts) else 0
    if '_exps' in n: continue
    tot['all'] += nb
    if n.startswith(pat) or not n.startswith('blk.'): print(n, dims, ty, nb)
print(tot, {k: v for k, v in kv.items() if 'count' in k or 'length' in k or 'head' in k})
