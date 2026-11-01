import numpy as np, os
from tokenizers import Tokenizer
tk=Tokenizer.from_file(os.environ['V41_SRC']+'/tokenizer.json')
ids=np.fromfile(os.path.expanduser('~/.cache/xolotl/runs/first-170/agent4k.i32'),dtype=np.int32)
a=np.loadtxt(os.path.expanduser('~/.cache/xolotl/runs/nll-pc512.txt')); b=np.loadtxt(os.path.expanduser('~/.cache/xolotl/runs/nll-pc128.txt'))
for p in [1906,3182,3614,2707,1794]:
    ctx=tk.decode(ids[max(0,p-25):p+1].tolist()); tgt=tk.decode([int(ids[p+1])])
    print(f'--- pos {p}: target {tgt!r}  span512 nll {a[p,1]:.2f} (guess {tk.decode([int(a[p,2])])!r})  span128 nll {b[p,1]:.2f} (guess {tk.decode([int(b[p,2])])!r})')
    print('   ...'+repr(ctx[-150:]))
    prev=[int(i) for i in np.where(ids[:p]==ids[p+1])[0]]
    print('   target token seen earlier at', prev[-5:])
