# cmpref.py run... : per-token distance of each engine dump to the exact torch GSQ-200 chain (nll-gsqtorch.txt)
import os, sys, numpy as np
R=os.path.expanduser('~/.cache/xolotl/runs/')
G=np.loadtxt(R+'nll-gsqtorch.txt'); gn=G[:,1]; ga=G[:,2].astype(int); conf=G[:,3]>3
rng=[(0,1024),(1024,2048),(2048,4095)]
print('%-14s %8s | %-24s | %-22s | %s'%('run','ppl','mean|d - torchGSQ|','argmax = torchGSQ','confident argmax'))
for k in sys.argv[1:]:
    try: b=np.loadtxt(R+f'nll-{k}.txt')
    except Exception: print(k,'missing'); continue
    d=np.abs(b[:,1]-gn)
    print('%-14s %8.4f | %-24s | %-22s | %s'%(k,np.exp(b[:,1].mean()),
      ' '.join('%.3f'%d[lo:hi].mean() for lo,hi in rng),
      ' '.join('%.3f'%(b[lo:hi,2]==ga[lo:hi]).mean() for lo,hi in rng),
      ' '.join('%.3f'%(b[lo:hi,2][conf[lo:hi]]==ga[lo:hi][conf[lo:hi]]).mean() for lo,hi in rng)))
