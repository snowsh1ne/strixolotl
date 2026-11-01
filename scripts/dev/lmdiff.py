# lmdiff.py A.bin B.bin : per layer/pos relative L2 diff of x_state (kind 0) and fn (kind 1), sel overlap (kind 2)
import sys, numpy as np, collections
def ld(f):
    d=collections.OrderedDict(); raw=open(f,'rb').read(); o=0
    while o < len(raw):
        l,p,k,c=np.frombuffer(raw,np.int32,4,o); o+=16
        dt=np.int32 if k==2 else np.float32
        d[(int(l),int(p),int(k))]=np.frombuffer(raw,dt,int(c),o); o+=4*int(c)
    return d
A=ld(sys.argv[1]); B=ld(sys.argv[2])
poss=sorted(set(p for (_,p,_) in A))
for p in poss:
    print('pos',p)
    for l in range(40):
        if (l,p,0) not in A or (l,p,0) not in B: continue
        out=[]
        for k in (0,1):
            x=A[(l,p,k)]; y=B[(l,p,k)]
            out.append('%s %.2e'%('x' if k==0 else 'fn', np.linalg.norm(x-y)/max(1e-30,np.linalg.norm(x))))
        if (l,p,2) in A and (l,p,2) in B:
            sa=A[(l,p,2)]; sb=B[(l,p,2)]; ma=A[(l,p,3)]; mb=B[(l,p,3)]
            va=set(sa[np.isfinite(ma)&(ma==0)].tolist()); vb=set(sb[np.isfinite(mb)&(mb==0)].tolist())
            out.append('sel |A|=%d |B|=%d common=%d'%(len(va),len(vb),len(va&vb)))
        print(' L%02d '%l+'  '.join(out))
