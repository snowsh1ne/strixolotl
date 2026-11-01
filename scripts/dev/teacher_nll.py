# per-position teacher NLL / argmax / margin for calib row ROW (first 4096 tokens), from h/L40 (v41_calib head logic)
import sys, numpy as np, torch, torch.nn.functional as F, json
import os
sys.path.insert(0, os.path.expanduser('~/projects/Xolotl_StrixHalo/xolotl/tools/v41'))
import v41_calib as C
from safetensors import safe_open
ROW = int(sys.argv[1]); OUT = sys.argv[2]
tokens = np.load(C.wpath("tokens.npy")); N, S = tokens.shape
torch.set_default_dtype(C.BF16)
args = C.model_args(1, S); wm = C.weight_map(); dev = 'cuda'
with torch.device(dev): norm = C.M.RMSNorm(C.D, args.norm_eps)
with safe_open(C.shard_file(wm["norm.weight"]), "pt") as f: norm.weight.data.copy_(f.get_tensor("norm.weight"))
with safe_open(C.shard_file(wm["head.weight"]), "pt") as f: W = f.get_tensor("head.weight").to(dev, torch.float32)
h_in = C.h_memmap(40, "r", N, S); pre = np.load(C.wpath("pre", "L40.npy"))
with torch.inference_mode():
    x = torch.from_numpy(np.asarray(h_in[ROW:ROW + 1])).to(dev).view(C.BF16)
    pm = torch.from_numpy(pre[ROW:ROW + 1]).to(dev)
    y = norm(torch.sum(pm.unsqueeze(-1) * x.float(), dim=2).to(C.BF16))[0]
    tgt = torch.from_numpy(tokens[ROW, 1:4096].astype(np.int64)).to(dev)
    lg = y[:4095].float() @ W.T
    lp = F.log_softmax(lg, -1)
    nll = -lp.gather(1, tgt[:, None])[:, 0]
    top2 = lp.topk(2, -1)
    with open(OUT, 'w') as f:
        for t in range(4095):
            f.write('%d %.6f %d %.4f\n' % (t, float(nll[t]), int(top2.indices[t, 0]), float(top2.values[t, 0] - top2.values[t, 1])))
print('teacher ppl first 4095', float(torch.exp(nll.mean())))
