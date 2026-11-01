# cmpnll.py REF OTHER... : per-512-block extra NLL vs REF, count of big positions, top positions
import os, sys, numpy as np
R=os.path.expanduser('~/.cache/xolotl/runs/')
def ld(f):
    a=np.loadtxt(f if '/' in f else R+f); return a[:,1],a[:,2].astype(int)
a,am=ld(sys.argv[1])
for f in sys.argv[2:]:
    b,bm=ld(f); d=b-a
    print(f, 'ppl %.4f'%np.exp(b.mean()), 'extra %.1f'%d.sum(), 'n>5: %d'%(d>5).sum(), 'blocks:', [round(d[k*512:(k+1)*512].sum(),1) for k in range(len(d)//512+1)])
    big=np.where(d>5)[0]; print('   big pos:', ' '.join(map(str,big[:40])))
