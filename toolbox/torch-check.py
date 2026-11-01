"""torch-check.py - does PyTorch work on this GPU the way tools/v41 needs it?  Exit 0 = every check passed.

  python3 -I /opt/xolotl/torch-check.py [--alloc-gib N]

Checks: the HIP device (gfx1151), the fp8/fp4 dtypes DeepSeek's model.py uses (GPU casts == CPU casts), fp32/bf16
matmul throughput, Cholesky + cholesky_inverse on a 5120x5120 Hessian-like matrix (what v41_gptq.hinv_chol does)
against float64 on the CPU, a CUDA-graph capture/replay (v41_gptq_graph), copy bandwidth, and one big allocation
(the calibration needs ~40 GiB at its peak; default = min(40, MemAvailable - 20) GiB).
"""
import argparse, sys, time
import torch

ok = True


def check(name, cond, info=""):
    global ok
    ok &= bool(cond)
    print(f"{'PASS' if cond else 'FAIL'} {name}{'  ' + info if info else ''}", flush=True)


def timed(fn, reps):
    fn(); torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps


def mem_available_gib():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable"):
            return int(line.split()[1]) / 2**20
    return 0.0


p = argparse.ArgumentParser()
p.add_argument("--alloc-gib", type=float, default=None)
a = p.parse_args()

print(f"torch {torch.__version__}  hip {torch.version.hip}  python {sys.version.split()[0]}")
check("HIP build", torch.version.hip is not None)
check("GPU visible", torch.cuda.is_available())
if not torch.cuda.is_available():
    sys.exit(1)
dev = "cuda"
pr = torch.cuda.get_device_properties(0)
arch = getattr(pr, "gcnArchName", "?")
check("device is gfx1151", arch.startswith("gfx1151"), f"{pr.name}, {arch}, {pr.multi_processor_count} CUs, "
      f"{pr.total_memory / 2**30:.1f} GiB")

# fp8 / fp4 dtypes: model.py and v41_calib cast through these on the GPU
for name in ("float8_e4m3fn", "float8_e8m0fnu", "float4_e2m1fn_x2"):
    check(f"dtype torch.{name}", hasattr(torch, name))
x = torch.randn(1 << 20) * 10
g = x.to(dev).to(torch.float8_e4m3fn).float().cpu()
check("fp8 e4m3 cast GPU == CPU", torch.equal(g, x.to(torch.float8_e4m3fn).float()))
e = torch.randint(100, 150, (4096,), dtype=torch.uint8)
s = e.to(dev).view(torch.float8_e8m0fnu).float().cpu()
check("e8m0 scales decode to 2^(e-127)", torch.equal(s, torch.ldexp(torch.ones(4096), e.int() - 127)))

# matmul throughput (GPTQ / Hessians are fp32, the teacher forward is bf16)
n = 8192
for dt in (torch.float32, torch.bfloat16):
    A = torch.randn(n, n, device=dev, dtype=dt); B = torch.randn(n, n, device=dev, dtype=dt)
    t = timed(lambda: A @ B, 5)
    check(f"matmul {str(dt)[6:]} {n}^3", True, f"{2 * n**3 / t / 1e12:.1f} TFLOPS")
    if dt == torch.float32:
        ref = (A[:256].double() @ B.double()).float()
        err = ((A[:256] @ B) - ref).abs().max().item() / ref.abs().max().item()
        check("fp32 matmul accuracy (no TF32-style shortcut)", err < 1e-5, f"max rel err {err:.1e}")
    del A, B

# hinv_chol: L = chol(H); Hi = cholesky_inverse(L); chol(Hi, upper)
C = 5120
X = torch.randn(4 * C, C, device=dev) * torch.rand(C, device=dev) * 2
H = X.T @ X
H += 0.01 * torch.mean(torch.diag(H)) * torch.eye(C, device=dev)
t0 = time.perf_counter()
L = torch.linalg.cholesky(H); Hi = torch.cholesky_inverse(L); U = torch.linalg.cholesky(Hi, upper=True)
torch.cuda.synchronize(); tc = time.perf_counter() - t0
Hd = H.double().cpu()
Ud = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(Hd)), upper=True)
err = ((U.double().cpu() - Ud).abs().max() / Ud.abs().max()).item()
check("cholesky + cholesky_inverse 5120^2 vs float64", err < 1e-3 and torch.isfinite(U).all().item(),
      f"max rel err {err:.1e}, {tc * 1000:.0f} ms")
del X, H, L, Hi, U

# CUDA graph capture + replay (v41_gptq_graph captures GPTQ's 128-column inner loop)
st = torch.cuda.Stream()
xs = torch.randn(4096, 4096, device=dev); ws = torch.randn(4096, 4096, device=dev)
st.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(st):
    for _ in range(2):
        ys = torch.relu(xs @ ws)
torch.cuda.current_stream().wait_stream(st)
gr = torch.cuda.CUDAGraph()
with torch.cuda.graph(gr):
    ys = torch.relu(xs @ ws)
xs.copy_(torch.randn_like(xs)); gr.replay(); torch.cuda.synchronize()
check("CUDA graph capture/replay", torch.equal(ys, torch.relu(xs @ ws)))
del xs, ws, ys, gr

# copy bandwidth (read + write)
z = torch.empty(1 << 30, device=dev, dtype=torch.float32)
y = torch.empty_like(z)
t = timed(lambda: y.copy_(z), 5)
check("device copy 4 GiB", True, f"{2 * z.numel() * 4 / t / 1e9:.0f} GB/s (read+write)")
del z, y
torch.cuda.empty_cache()

# one big allocation, touched
gib = a.alloc_gib if a.alloc_gib is not None else max(0.0, min(40.0, mem_available_gib() - 20))
if gib >= 1:
    try:
        big = torch.empty(int(gib * 2**30) // 4, device=dev, dtype=torch.float32)
        big.fill_(1.0); torch.cuda.synchronize()
        check(f"allocate + touch {gib:.0f} GiB", big[-1].item() == 1.0)
        del big
    except RuntimeError as ex:
        check(f"allocate + touch {gib:.0f} GiB", False, str(ex).splitlines()[0])
else:
    print("SKIP big allocation (MemAvailable too low)")
torch.cuda.empty_cache()

print("ALL PASS" if ok else "SOME CHECKS FAILED")
sys.exit(0 if ok else 1)
