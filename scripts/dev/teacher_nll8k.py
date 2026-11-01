# teacher_nll8k.py ROW OUT [ROW OUT ...] - teacher_nll.py over the whole row (S-1 positions): per-position teacher
# NLL / argmax / top-2 margin from the chain's h/L40 (V41_WORK = the chain's work dir).  Run in xolotl-quant.
import sys, numpy as np, torch, torch.nn.functional as F
import os
sys.path.insert(0, os.path.expanduser('~/projects/Xolotl_StrixHalo/xolotl/tools/v41'))
import v41_calib as C
from safetensors import safe_open
jobs = list(zip(sys.argv[1::2], sys.argv[2::2]))
tokens = np.load(C.wpath("tokens.npy")); N, S = tokens.shape
torch.set_default_dtype(C.BF16)
args = C.model_args(1, S); wm = C.weight_map(); dev = 'cuda'
with torch.device(dev): norm = C.M.RMSNorm(C.D, args.norm_eps)
with safe_open(C.shard_file(wm["norm.weight"]), "pt") as f: norm.weight.data.copy_(f.get_tensor("norm.weight"))
with safe_open(C.shard_file(wm["head.weight"]), "pt") as f: W = f.get_tensor("head.weight").to(dev, torch.float32)
h_in = C.h_memmap(40, "r", N, S); pre = np.load(C.wpath("pre", "L40.npy"))
for row, out in jobs:
    ROW = int(row)
    with torch.inference_mode():
        x = torch.from_numpy(np.asarray(h_in[ROW:ROW + 1])).to(dev).view(C.BF16)
        pm = torch.from_numpy(pre[ROW:ROW + 1]).to(dev)
        y = norm(torch.sum(pm.unsqueeze(-1) * x.float(), dim=2).to(C.BF16))[0]
        tgt = torch.from_numpy(tokens[ROW, 1:].astype(np.int64)).to(dev)
        nll, am, mg = [], [], []
        for a in range(0, S - 1, 1024):   # 1024-position chunks keep the [n, vocab] F32 logits small
            lp = F.log_softmax(y[a:min(a + 1024, S - 1)].float() @ W.T, -1)
            nll.append(-lp.gather(1, tgt[a:a + lp.shape[0], None])[:, 0]); t2 = lp.topk(2, -1)
            am.append(t2.indices[:, 0]); mg.append(t2.values[:, 0] - t2.values[:, 1])
        nll, am, mg = torch.cat(nll), torch.cat(am), torch.cat(mg)
    with open(out, 'w') as f:
        for t in range(S - 1):
            f.write('%d %.6f %d %.4f\n' % (t, float(nll[t]), int(am[t]), float(mg[t])))
    print(f'row {ROW}: teacher ppl {float(torch.exp(nll.mean())):.4f} -> {out}', flush=True)
