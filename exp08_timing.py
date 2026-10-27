"""Exp 8: cost of SAFI (build + fast apply) vs n, n_u and band limit U.

Fast apply = M+1 type-2 NUFFTs (numpy Greengard-Lee, nufft.py) + dense partial cells.
Reference cost: ONE type-2 NUFFT of the trapezoid sum (what a NUFFT-based method pays).
Single process, CPU otherwise idle.
"""
import sys, json, time
sys.path.insert(0, '.')
import numpy as np
from bench import PiecewiseExp
from safi import SAFI
from nufft import nufft2

M = 6
rng = np.random.default_rng(3)
f = PiecewiseExp.random(rng, [0.3137, 0.6713])


def tmin(fn, rep=3):
    best = np.inf
    for _ in range(rep):
        t = time.perf_counter(); fn(); best = min(best, time.perf_counter() - t)
    return best


out = {'vs_n': [], 'vs_U': []}
nu = 10 ** 5
u = np.random.default_rng(1).uniform(-1e4, 1e4, nu)
for p in [10, 12, 14, 16, 18, 20]:
    n = 2 ** p
    x = np.linspace(0, 1, n + 1); y = f(x)
    tb = tmin(lambda: SAFI(x, y, M, 'adaptive', f=f, q=30), rep=1)
    S = SAFI(x, y, M, 'adaptive', f=f, q=30)
    te = tmin(lambda: S.eval_fast(u))
    tn = tmin(lambda: nufft2(y[:-1], 2 * np.pi * u / n))
    F = f.ft(u[:2000])
    err = float(np.linalg.norm(F - S.eval_fast(u[:2000])) / np.linalg.norm(F))
    out['vs_n'].append(dict(n=n, build_s=tb, apply_s=te, one_nufft_s=tn, ratio=te / tn, rel_err=err))
    print(out['vs_n'][-1], flush=True)

n = 2 ** 14
x = np.linspace(0, 1, n + 1); S = SAFI(x, f(x), M, 'adaptive', f=f, q=40)
for U in [1e2, 1e3, 1e4, 1e5, 1e6, 1e7]:
    uu = np.random.default_rng(2).uniform(-U, U, nu)
    te = tmin(lambda: S.eval_fast(uu))
    F = f.ft(uu[:2000])
    err = float(np.linalg.norm(F - S.eval_fast(uu[:2000])) / np.linalg.norm(F))
    out['vs_U'].append(dict(U=U, apply_s=te, rel_err=err))
    print(out['vs_U'][-1], flush=True)
json.dump(out, open('results/exp08_timing.json', 'w'), indent=1)
