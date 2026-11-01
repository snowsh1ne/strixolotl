# vsT.py run... : ppl, late stats vs teacher and pairwise
import os, sys, numpy as np
R=os.path.expanduser('~/.cache/xolotl/runs/')
T=np.loadtxt(R+'nll-teacher.txt'); tn=T[:,1]; ta=T[:,2].astype(int); conf=T[:,3]>3
rng=[(0,1024),(1024,2048),(2048,4095)]
D={}
for k in sys.argv[1:]:
    try: D[k]=np.loadtxt(R+f'nll-{k}.txt')
    except Exception as e: print(k,'missing'); continue
    b=D[k]; d=b[:,1]-tn
    print('%-11s ppl %.4f | mean|d-T| %s | confident argmax=T %s | n(d>5) %s'%(k,np.exp(b[:,1].mean()),
      [round(np.abs(d[lo:hi]).mean(),3) for lo,hi in rng],
      [round((b[lo:hi,2][conf[lo:hi]]==ta[lo:hi][conf[lo:hi]]).mean(),3) for lo,hi in rng],
      [int((d[lo:hi]>5).sum()) for lo,hi in rng]))
ks=list(D)
for i in range(len(ks)):
    for j in range(i+1,len(ks)):
        a,b=D[ks[i]],D[ks[j]]
        print('  %s vs %s: mean|d| %s'%(ks[i],ks[j],[round(np.abs(a[lo:hi,1]-b[lo:hi,1]).mean(),3) for lo,hi in rng]))
