#!/usr/bin/env python3
"""ablate.py - where does the engine's distance to the torch GSQ chain come from?  Forward-only, one sequence
(calibration row 12 = agent4k, first S tokens), all 40 layers, several variants carried side by side through the
same layer-streaming code as v41_calib.cmd_layer / v41_quant (DeepSeek's model.py on tools/v41/torch_kernels).

Chain variants (each propagates its own state through all layers):
  T   teacher: model.py's FP4 experts (Expert.forward, FP8 act quant)        -> compare with nll-teacher.txt
  E   exact GSQ: the GGUF's GSQ experts (GPU-decoded), expert_f fp32 (v41_quant's propagation math)
  A   E + q8_1 expert activations (input of gate/up and the SwiGLU output before down, ggml quantize_q8_1)
  B   E + the GGUF's dense re-encoding: Q8_0 of the FP8 attention q_a/q_b/kv/o_a/o_b + shared expert
  AB  A + B
  C   E + the GGUF's F16 storage of hc_fn (attn+ffn) and the compressor wkv/wgate (reference F32)
Local variants (per layer, from E's input, not propagated): A_l, B_l, AB_l, C_l -> per-layer relative error.

Usage: ablate.py run [--S 4096] [--row 12] [--ckpt DIR] [--layers 0-39] ;  ablate.py head [--ckpt DIR]
"""
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as Fnn

TOOLS = os.path.expanduser("~/projects/Xolotl_StrixHalo/xolotl/tools/v41")
sys.path.insert(0, TOOLS)
os.environ.setdefault("V41_GSQ_GGUF", os.path.expanduser("~/models/xolotl/V41-1017-GSQ/V41-1017-GSQ-200.gguf"))
import v41_calib as C  # noqa: E402
import v41_repair as R  # noqa: E402

M, K = C.M, C.K
BF16 = torch.bfloat16
D, HC, NE = C.D, C.HC, C.NE
CALIB = os.path.expanduser("~/models/xolotl/v41-work/calib-1017")
QTORCH = os.path.expanduser("~/models/xolotl/v41-work/quant-1017-200-torch")
# (experts, dense mode, f16 storage); dense mode: None | "q8" (all) | "q8attn" | "q8shexp" | "ident" (the q8 code
# path with the exact FP8 values: a null perturbation = the numerical noise floor of the chain)
CHAIN = {"T": ("fp4", None, False), "E": ("gsq", None, False), "A": ("q8a", None, False),
         "B": ("gsq", "q8", False), "AB": ("q8a", "q8", False), "C": ("gsq", None, True), "N": ("gsq", "ident", False)}
LOCAL = {"A_l": ("q8a", None, False), "B_l": ("gsq", "q8", False), "AB_l": ("q8a", "q8", False),
         "C_l": ("gsq", None, True), "N_l": ("gsq", "ident", False), "Ba_l": ("gsq", "q8attn", False),
         "Bs_l": ("gsq", "q8shexp", False)}
DENSE_FP8 = ("attn.wq_a", "attn.wq_b", "attn.wkv", "attn.wo_b",
             "ffn.shared_experts.w1", "ffn.shared_experts.w2", "ffn.shared_experts.w3")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ------------------------------------------------------------------------------------------ number formats
def roundf(x):
    """C roundf: half away from zero."""
    return torch.sign(x) * torch.floor(x.abs() + 0.5)


def q8_0_roundtrip(W):
    """ggml quantize_row_q8_0_ref + dequant: d = amax/127, q = roundf(x * (1/d)), stored d fp16."""
    R_, Cc = W.shape
    g = W.float().reshape(R_, Cc // 32, 32)
    d = g.abs().amax(-1, keepdim=True) / 127.0
    idv = torch.where(d != 0, 1.0 / d, torch.zeros_like(d))
    q = roundf(g * idv)
    return (q * d.half().float()).reshape(R_, Cc)


def q8_1_act(x):
    """ggml/strata quantize_q8_1 (q8_1_finite.hpp): blocks of 32, d = amax/127, q = roundf(x / d) clamped,
    the dot product uses the fp16 d.  Returns the dequantized activation (fp32)."""
    sh = x.shape
    g = x.float().reshape(*sh[:-1], sh[-1] // 32, 32)
    amax = g.abs().amax(-1, keepdim=True)
    d = amax / 127.0
    q = torch.where(amax > 0, roundf(g / torch.where(d > 0, d, torch.ones_like(d))).clamp(-127, 127),
                    torch.zeros_like(g))
    return (q * d.clamp(max=65504.0).half().float()).reshape(sh)


# ------------------------------------------------------------------------------------------ patched attention
def attn_forward(self, x, start_pos):
    """model.py Attention.forward verbatim, except wo_a may be overridden by fp32 values (Q8_0 re-encoding)."""
    bsz, seqlen, _ = x.size()
    freqs_cis = self.freqs_cis[start_pos: start_pos + seqlen]
    rd = self.rope_head_dim
    qr = self.q_norm(self.wq_a(x))
    q = self.wq_b(qr).unflatten(-1, (self.n_local_heads, self.head_dim))
    M.apply_rotary_emb(q[..., -rd:], freqs_cis)
    kv, topk_idxs = self._window_kv(x, freqs_cis, start_pos)
    if self.compress_ratio:
        compress_kv, compress_idxs = self._compress_kv(x, qr, start_pos, kv.size(1))
        kv = torch.cat([kv, compress_kv], dim=1)
        topk_idxs = torch.cat([topk_idxs, compress_idxs], dim=-1)
    o = M.sparse_attn(q, kv, self.attn_sink, topk_idxs, self.softmax_scale)
    M.apply_rotary_emb(o[..., -rd:], freqs_cis, True)
    o = o.view(bsz, seqlen, self.n_local_groups, -1)
    M.act_quant(o, M.fp8_block_size, M.scale_fmt, M.scale_dtype, True)
    ov = getattr(self, "_wo_a_override", None)
    if ov is None:
        wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
        o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
    else:
        o = torch.einsum("bsgd,grd->bsgr", o.float(), ov.view(self.n_local_groups, self.o_lora_rank, -1)).to(x.dtype)
    return self.wo_b(o.flatten(2))


M.Attention.forward = attn_forward


def fp8act_linear(Wf):
    """FP8 act_quant (ue8m0, 32) on the input, then an fp32 GEMM against the given fp32 weight, out in bf16."""
    def fwd(x):
        a, s = K.act_quant(x, M.fp8_block_size, M.scale_fmt, M.scale_dtype)
        A = K._dequant_act(a, s, M.fp8_block_size)
        return (A @ Wf.t()).to(torch.get_default_dtype())
    return fwd


class Swapper:
    """Per-layer alternative dense weights; apply(dense_q8, f16store) / reset()."""

    def __init__(self, blk):
        self.blk = blk
        mods = dict(blk.named_modules())
        self.mods = mods
        self.q8 = {}
        for n in DENSE_FP8:
            w = mods[n].weight
            assert w.dtype == torch.float8_e4m3fn, (n, w.dtype)
            W = K.dequant_fp8_weight(w, w.scale)
            Wq = q8_0_roundtrip(W)
            self.q8[n] = (Wq, float((Wq - W).norm() / W.norm()), W)
        wa = blk.attn.wo_a.weight.float()
        self.wo_a_q8 = q8_0_roundtrip(wa)
        self.wo_a_f32 = wa
        self.q8["attn.wo_a"] = (self.wo_a_q8, float((self.wo_a_q8 - wa).norm() / wa.norm()), wa)
        self.f16 = {}
        params = dict(blk.named_parameters())
        names = ["hc_attn_fn", "hc_ffn_fn"]
        if blk.attn.compressor is not None:
            names += [n for n in ("attn.compressor.wkv.weight", "attn.compressor.wgate.weight") if n in params]
        for n in names:
            p = params[n]
            orig = p.data.clone()
            alt = p.data.float().half().float().to(p.dtype)
            self.f16[n] = (p, orig, alt, float((alt.float() - orig.float()).norm() / orig.float().norm()))
        self.fwd = {n: fp8act_linear(self.q8[n][0]) for n in DENSE_FP8}
        self.fwd_id = {n: fp8act_linear(self.q8[n][2]) for n in DENSE_FP8}

    def apply(self, dense, f16store):
        self.reset()
        if dense:
            attn = dense in ("q8", "q8attn", "ident")
            shx = dense in ("q8", "q8shexp", "ident")
            fw = self.fwd_id if dense == "ident" else self.fwd
            for n in DENSE_FP8:
                if (n.startswith("attn.") and attn) or (n.startswith("ffn.") and shx):
                    self.mods[n].forward = fw[n]
            if attn:
                self.blk.attn._wo_a_override = self.wo_a_f32 if dense == "ident" else self.wo_a_q8
        if f16store:
            for n, (p, orig, alt, _) in self.f16.items():
                p.data.copy_(alt)

    def reset(self):
        for n in DENSE_FP8:
            self.mods[n].__dict__.pop("forward", None)
        self.blk.attn._wo_a_override = None
        for n, (p, orig, alt, _) in self.f16.items():
            p.data.copy_(orig)

    def info(self):
        return {"q8_rel": {n: v[1] for n, v in self.q8.items()}, "f16_rel": {n: v[3] for n, v in self.f16.items()}}


# ------------------------------------------------------------------------------------------ experts
def expert_exact(x, W1, W3, W2, g, lim):
    return C.expert_f(x, W1, W3, W2, g, lim)


def expert_q8a(x, W1, W3, W2, g, lim):
    """engine: q8_1(x) -> gate/up -> clamp, SwiGLU (unweighted) -> q8_1 -> down -> x routing weight."""
    xq = q8_1_act(x.float())
    gate = xq @ W1.T
    up = xq @ W3.T
    up = up.clamp(-lim, lim)
    gate = gate.clamp(max=lim)
    h = Fnn.silu(gate) * up
    return (q8_1_act(h) @ W2.T) * g.float()


# ------------------------------------------------------------------------------------------ the layer
SHARED = C.SHARED


def get_shared():
    return {k: (None if getattr(M.shared_attn, k) is None else getattr(M.shared_attn, k).clone()) for k in SHARED}


def set_shared(sh):
    for k in SHARED:
        setattr(M.shared_attn, k, None if sh[k] is None else sh[k].clone())


def attn_half(blk, st, hashes_e):
    x, pm = st["h"], st["pre"]
    if blk.engram is not None:
        x = blk.engram(x, hashes_e, None)
    set_shared(st["shared"])
    residual = x
    attn_pre, attn_post, attn_comb = blk.hc_mixes(x, blk.hc_attn_fn, blk.hc_attn_scale, blk.hc_attn_base)
    y = blk.attn(blk.attn_norm(blk.hc_pre(x, pm)), 0)
    x = blk.hc_post(y, residual, attn_post, attn_comb)
    ffn_pre, post, comb = blk.hc_mixes(x, blk.hc_ffn_fn, blk.hc_ffn_scale, blk.hc_ffn_base)
    X = blk.ffn_norm(blk.hc_pre(x, attn_pre)).view(-1, D)
    return {"X": X, "resid": x, "post": post, "comb": comb, "nxt": ffn_pre, "shared": get_shared()}


def run(a):
    dev = "cuda"
    S, row = a.S, a.row
    os.makedirs(a.ckpt, exist_ok=True)
    tokens = np.load(os.path.join(CALIB, "tokens.npy"), mmap_mode="r")[row, :S].astype(np.int64)
    hashes = np.load(os.path.join(CALIB, "hashes.npy"), mmap_mode="r")
    torch.set_default_dtype(BF16)
    args = C.model_args(1, S)
    layout = M.EngramLayout.from_args(args)
    lim = args.swiglu_limit
    L0, L1 = map(int, a.layers.split("-"))
    stfile = os.path.join(a.ckpt, "state.pt")
    if L0 == 0:
        from safetensors import safe_open
        with safe_open(C.shard_file("model-00002-of-00048.safetensors"), "pt") as f:
            emb = f.get_tensor("embed.weight")
        e = emb[torch.from_numpy(tokens)].to(dev)
        h0 = e.view(1, S, 1, D).expand(1, S, HC, D).contiguous()
        pre0 = torch.zeros((1, S, HC), dtype=torch.float32, device=dev)
        pre0[..., 0] = 1.0
        del emb
        states = {v: {"h": h0.clone(), "pre": pre0.clone(), "shared": dict.fromkeys(SHARED)} for v in CHAIN}
    else:
        states = torch.load(stfile, map_location=dev)
        assert states["_next"] == L0, (states["_next"], L0)
        states.pop("_next")
    statsf = open(os.path.join(a.ckpt, "layers.jsonl"), "a")
    for L in range(L0, L1 + 1):
        t0 = time.time()
        torch.set_default_dtype(BF16)
        hashes_e = None
        if L in args.engram_layer_ids:
            ei = args.engram_layer_ids.index(L)
            C.MemmapEngramEmbedding.SRC = (C.shard_file(C.weight_map()[f"layers.{L}.engram.embed.weight"]),
                                           f"layers.{L}.engram.embed.weight", f"layers.{L}.engram.embed.scale")
            M.ParallelEngramEmbedding = C.MemmapEngramEmbedding
            hashes_e = torch.from_numpy(np.ascontiguousarray(hashes[row:row + 1, :S, ei, :])).to(dev)
        with torch.device(dev):
            blk = M.Block(L, args, layout)
        C.load_block(blk, L)
        torch.set_default_device(dev)
        sw = Swapper(blk)
        gsq = R.GsqLayer(L, dev)
        t_load = time.time() - t0
        rec = {"layer": L, "weights": sw.info()}
        with torch.inference_mode():
            # ---- attention halves: chain variants on their own state, local variants on E's
            ah = {}
            for v, (ek, dq8, f16) in list(CHAIN.items()) + list(LOCAL.items()):
                src = states[v] if v in CHAIN else states["E"]
                sw.apply(dq8, f16)
                ah[v] = attn_half(blk, src, hashes_e)
            sw.reset()
            t_attn = time.time() - t0 - t_load
            # ---- routing
            route = {}
            for v in ah:
                w, i = blk.ffn.gate(ah[v]["X"])
                route[v] = (w.float(), i)
            Y = {v: torch.zeros((S, D), dtype=torch.float32, device=dev) for v in ah}
            for ex in range(NE):
                sel = {}
                for v in ah:
                    tok, slot = torch.where(route[v][1] == ex)
                    if len(tok):
                        sel[v] = (tok, slot)
                if not sel:
                    continue
                need_gsq = any(((CHAIN.get(v) or LOCAL.get(v))[0] != "fp4") for v in sel)
                if need_gsq:
                    W1, W3, W2 = gsq.weights(ex)
                for v, (tok, slot) in sel.items():
                    kind = (CHAIN.get(v) or LOCAL.get(v))[0]
                    xe, ge = ah[v]["X"][tok], route[v][0][tok, slot, None]
                    if kind == "fp4":
                        out = blk.ffn.experts[ex](xe, ge)
                    elif kind == "gsq":
                        out = expert_exact(xe, W1, W3, W2, ge, lim).to(BF16)
                    else:
                        out = expert_q8a(xe, W1, W3, W2, ge, lim).to(BF16)
                    Y[v].index_add_(0, tok, out.float())
            # ---- shared expert + hc_post
            outs = {}
            for v in ah:
                ek, dq8, f16 = CHAIN.get(v) or LOCAL.get(v)
                sw.apply(dq8, f16)
                Y[v] += blk.ffn.shared_experts(ah[v]["X"])
                outs[v] = blk.hc_post(Y[v].view(1, S, D).to(BF16), ah[v]["resid"], ah[v]["post"], ah[v]["comb"])
                assert torch.isfinite(outs[v]).all(), (L, v)
            sw.reset()
            # ---- metrics
            ref = outs["E"].float()
            hin = states["E"]["h"].float()
            dn = float((ref - hin).norm())
            rn = float(ref.norm())
            rec["E_out_rms"] = float(ref.square().mean().sqrt())
            rec["E_delta_over_out"] = dn / rn
            loc = {}
            for v in list(LOCAL) + ["T"] + [x for x in CHAIN if x not in ("E", "T")]:
                d = outs[v].float() - ref
                loc[v] = {"rel_out": float(d.norm()) / rn, "rel_delta": float(d.norm()) / dn,
                          "rel_X": float((ah[v]["X"].float() - ah["E"]["X"].float()).norm() / ah["E"]["X"].float().norm())}
                # routing agreement with E
                same = (torch.sort(route[v][1], 1).values == torch.sort(route["E"][1], 1).values).all(1).float().mean()
                loc[v]["route_same"] = float(same)
            rec["vsE"] = loc
            # per-layer relative error of the expert-only part (local variants): MoE output vs E's
            for v in LOCAL:
                rec["vsE"][v]["rel_moe"] = float((Y[v] - Y["E"]).norm() / Y["E"].norm())
            for v in CHAIN:
                states[v] = {"h": outs[v], "pre": ah[v]["nxt"], "shared": ah[v]["shared"]}
        del ah, Y, outs, route
        rec["seconds"] = {"load": round(t_load, 1), "attn": round(t_attn, 1), "total": round(time.time() - t0, 1)}
        statsf.write(json.dumps(rec) + "\n")
        statsf.flush()
        log(f"L{L}: {rec['seconds']}  chain rel_out vs E: " +
            " ".join(f"{v} {rec['vsE'][v]['rel_out']:.2e}" for v in CHAIN if v != "E") + " | local rel_out: " +
            " ".join(f"{v} {rec['vsE'][v]['rel_out']:.2e}" for v in LOCAL) + " | local rel_moe: " +
            " ".join(f"{v} {rec['vsE'][v]['rel_moe']:.2e}" for v in LOCAL))
        del blk, sw, gsq
        torch.set_default_device("cpu")
        torch.cuda.empty_cache()
        if L == L1 or L % 5 == 4:
            states["_next"] = L + 1
            torch.save(states, stfile + ".tmp")
            os.replace(stfile + ".tmp", stfile)
            states.pop("_next")
    states["_next"] = L1 + 1
    torch.save(states, stfile + ".tmp")
    os.replace(stfile + ".tmp", stfile)


def head(a):
    """Per-position NLL for every chain variant (state after layer 39) + the stored chains (teacher, torch GSQ)."""
    from safetensors import safe_open
    dev = "cuda"
    S, row = a.S, a.row
    torch.set_default_dtype(BF16)
    args = C.model_args(1, S)
    wm = C.weight_map()
    with torch.device(dev):
        norm = M.RMSNorm(D, args.norm_eps)
    with safe_open(C.shard_file(wm["norm.weight"]), "pt") as f:
        norm.weight.data.copy_(f.get_tensor("norm.weight"))
    with safe_open(C.shard_file(wm["head.weight"]), "pt") as f:
        W = f.get_tensor("head.weight").to(dev, torch.float32)
    tokens = np.load(os.path.join(CALIB, "tokens.npy"), mmap_mode="r")[row, :S].astype(np.int64)
    tgt = torch.from_numpy(tokens[1:]).to(dev)
    states = torch.load(os.path.join(a.ckpt, "state.pt"), map_location=dev)
    assert states.pop("_next") == 40
    finals = {v: (states[v]["h"], states[v]["pre"]) for v in CHAIN}
    for name, wd in (("teacher_stored", CALIB), ("gsqtorch_stored", QTORCH)):
        Nn, Ss = 32, 8192
        hm = np.memmap(os.path.join(wd, "h", "L40.bin"), dtype=np.int16, mode="r", shape=(Nn, Ss, HC, D))
        pre = np.load(os.path.join(wd, "pre", "L40.npy"), mmap_mode="r")
        finals[name] = (torch.from_numpy(np.ascontiguousarray(hm[row:row + 1, :S])).to(dev).view(BF16),
                        torch.from_numpy(np.ascontiguousarray(pre[row:row + 1, :S])).to(dev))
    names = list(finals)
    ys = {}
    with torch.inference_mode():
        for v, (x, pm) in finals.items():
            ys[v] = norm(torch.sum(pm.unsqueeze(-1) * x.float(), dim=2).to(BF16))[0]
        nll = {v: np.zeros(S - 1) for v in names}
        top1 = {v: np.zeros(S - 1, np.int64) for v in names}
        kl = {v: np.zeros(S - 1) for v in names}   # KL(E || v)
        for c in range(0, S - 1, 512):
            e = min(c + 512, S - 1)
            lp = {v: torch.log_softmax(ys[v][c:e].float() @ W.T, -1) for v in names}
            pE = lp["E"].exp()
            for v in names:
                nll[v][c:e] = (-lp[v].gather(1, tgt[c:e, None])[:, 0]).cpu().numpy()
                top1[v][c:e] = lp[v].argmax(-1).cpu().numpy()
                kl[v][c:e] = (pE * (lp["E"] - lp[v])).sum(-1).cpu().numpy()
    np.savez(os.path.join(a.ckpt, "nll.npz"), **{f"nll_{v}": nll[v] for v in names},
             **{f"top1_{v}": top1[v] for v in names}, **{f"kl_{v}": kl[v] for v in names})
    for v in names:
        log(f"head {v:16s} ppl {np.exp(nll[v].mean()):.4f} mean nll {nll[v].mean():.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "head"])
    ap.add_argument("--S", type=int, default=4096)
    ap.add_argument("--row", type=int, default=12)
    ap.add_argument("--layers", default="0-39")
    ap.add_argument("--ckpt", default=os.path.expanduser("~/.cache/xolotl/ablate"))
    a = ap.parse_args()
    {"run": run, "head": head}[a.cmd](a)


if __name__ == "__main__":
    main()
