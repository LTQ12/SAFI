"""Exp 3: total function evaluations needed for band-relative error <= eps on all of [1, U].

SAFI-auto (certified, stops on its own estimate) vs. oracle-stopped baselines that are told
the true error: SAFI-point h-refinement, trapezoid sum (NUFFT object), known-break oracle,
and QUADPACK QAWO (adaptive oscillatory quadrature, one call per frequency, per cos/sin).
"""
import sys, json, time
sys.path.insert(0, '.')
import numpy as np
from scipy.integrate import quad
from bench import PiecewiseExp, band_grid, band_rel
from safi import SAFI, safi_auto, riemann_sum

M = 6
UEXP = [6, 8, 10, 12, 14, 16]
EPSS = [1e-4, 1e-8]
NF = 6
rng = np.random.default_rng(31)
funcs = []
while len(funcs) < NF:
    c = np.sort(rng.uniform(0.15, 0.85, 2))
    if c[1] - c[0] > 0.1:
        funcs.append(PiecewiseExp.random(rng, c))


def oracle_n(make, F, lab, eps, nmin=64, nmax=2 ** 16):
    n = nmin
    while n <= nmax:
        if band_rel(F, make(n), lab).max() <= eps:
            return n + 1
        n *= 2
    return np.nan


res = {}
t0 = time.time()
for eps in EPSS:
    for ue in UEXP:
        U = 2.0 ** ue
        u, lab, _ = band_grid(U, per_band=16, rng=np.random.default_rng(ue))
        row = {k: [] for k in ['auto', 'auto_true', 'auto_cert', 'point', 'known', 'trap']}
        for f in funcs:
            F = f.ft(u)
            S, Fh, eh, st = safi_auto(f, 0, 1, u, eps, M=M)
            row['auto'].append(st['nf']); row['auto_true'].append(float(band_rel(F, Fh, lab).max())); row['auto_cert'].append(st['certified'])
            xs = lambda n: np.linspace(0, 1, n + 1)
            row['point'].append(oracle_n(lambda n: SAFI(xs(n), f(xs(n)), M, 'point')(u), F, lab, eps, nmax=2 ** 15))
            row['known'].append(oracle_n(lambda n: SAFI(xs(n), f(xs(n)), M, 'known', breaks=list(f.breaks[1:-1]))(u), F, lab, eps))
            row['trap'].append(oracle_n(lambda n: riemann_sum(xs(n), f(xs(n)), u), F, lab, eps, nmax=2 ** 17))
        res[f'{eps:g}_{ue}'] = {k: v for k, v in row.items()}
        print(f'eps={eps:g} U=2^{ue}', {k: (np.nanmedian(v) if k not in ('auto_true', 'auto_cert') else (np.max(v) if k == 'auto_true' else float(np.mean(v)))) for k, v in row.items()},
              f'{time.time() - t0:.0f}s', flush=True)

# QUADPACK per-frequency adaptive oscillatory quadrature (subset: 2 functions, 3 freq per band)
qp = {}
for eps in EPSS:
    for ue in [6, 10, 14]:
        U = 2.0 ** ue
        u, lab, _ = band_grid(U, per_band=3, rng=np.random.default_rng(100 + ue))
        tot, worst = [], []
        for f in funcs[:2]:
            F = f.ft(u); nev = 0; G = np.empty_like(F)
            for i, uu in enumerate(u):
                w = 2 * np.pi * uu
                tol = 0.5 * eps * abs(F[i])
                g = lambda t: f(np.array([t]))[0]
                rc = quad(g, 0, 1, weight='cos', wvar=w, epsabs=tol, epsrel=0, limit=4000, full_output=1)
                rs = quad(g, 0, 1, weight='sin', wvar=w, epsabs=tol, epsrel=0, limit=4000, full_output=1)
                nev += rc[2]['neval'] + rs[2]['neval']
                G[i] = rc[0] - 1j * rs[0]
            tot.append(nev / u.size); worst.append(float(np.max(np.abs(F - G) / np.abs(F))))
        qp[f'{eps:g}_{ue}'] = dict(evals_per_freq=float(np.median(tot)), worst_pointwise_rel=float(np.max(worst)))
        print('QUADPACK', eps, ue, qp[f'{eps:g}_{ue}'], flush=True)

json.dump(dict(M=M, UEXP=UEXP, EPSS=EPSS, NF=NF, res=res, quadpack=qp), open('results/exp03_complexity.json', 'w'), indent=1)

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
Us = 2.0 ** np.array(UEXP)
nu_per = 16
for a, eps in enumerate(EPSS):
    for k, c, lbl in [('auto', '#126', 'SAFI-auto (certified)'), ('known', '#2a5', 'known breaks (oracle stop)'),
                      ('point', '#d62', 'SAFI point, h-refine (oracle stop)'), ('trap', '#888', 'trapezoid / NUFFT (oracle stop)')]:
        v = [np.nanmedian(res[f'{eps:g}_{ue}'][k]) for ue in UEXP]
        ax[a].loglog(Us, v, '-o', color=c, label=lbl)
    qx = [2.0 ** ue for ue in [6, 10, 14]]
    qv = [qp[f'{eps:g}_{ue}']['evals_per_freq'] * nu_per * np.log2(2.0 ** ue) for ue in [6, 10, 14]]
    ax[a].loglog(qx, qv, '-s', color='#a5c', label='QUADPACK QAWO, all targets')
    ax[a].loglog(Us, Us, 'k:', lw=1, label='~U')
    ax[a].set(xlabel='band limit U', ylabel='function evaluations', title=f'eps = {eps:g} (band relative)', ylim=(50, 3e6))
    ax[a].legend(fontsize=7)
plt.tight_layout(); plt.savefig('figs/exp03_complexity.png', dpi=140)
