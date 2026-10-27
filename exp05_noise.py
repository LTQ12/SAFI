"""Exp 5: noisy point samples, sigma in {1e-10, 1e-7, 1e-4}.

Prediction: bisection still localises the jump (decisions need only |J| >> sigma), so the
location horizon disappears; what remains is the noise floor of the smooth part, which is
flat in u (white noise has a flat transform, ~ sigma sqrt(h)) while |F| ~ |J|/(2 pi u):
relative noise horizon u_noise(eps) ~ eps |J| / (sigma sqrt(h)) -- grows only like sqrt(n).
The estimator adds a noise term obtained by randomised probing of the linear map y -> F[p].
"""
import sys, json
sys.path.insert(0, '.')
import numpy as np
from bench import PiecewiseExp, band_grid, band_rel, horizon
from safi import SAFI

M = 6
u, lab, edges = band_grid(2.0 ** 22, per_band=32, rng=np.random.default_rng(11))
rng = np.random.default_rng(55)
funcs = [PiecewiseExp.random(rng, rng.uniform(0.25, 0.75)) for _ in range(8)]
SIG = [0.0, 1e-10, 1e-7, 1e-4]
NS = [128, 512, 2048]
out = {}
for sig in SIG:
    for n in NS:
        rels, effs, hz, locs = [], [], [], []
        for b, f in enumerate(funcs):
            x = np.linspace(0, 1, n + 1)
            nz = np.random.default_rng(1000 * b + n)
            y = f(x) + sig * nz.normal(size=x.size)
            fq = lambda t, nz=nz: f(t) + sig * nz.normal(size=np.size(t))
            S = SAFI(x, y, M, 'adaptive', f=fq, q=40, sigma=sig)
            if len(S.breaks) != 1:
                rels.append(np.full(edges.size - 1, np.nan)); continue
            F = f.ft(u); G = S(u); e = S.error_estimate(u)
            r = band_rel(F, G, lab); rels.append(r)
            effs.append(np.min([np.linalg.norm(e[lab == k]) / np.linalg.norm((F - G)[lab == k]) for k in np.unique(lab)]))
            hz.append(horizon(r, edges, 1e-4)); locs.append(abs(S.breaks[0].c - f.breaks[1]))
        out[f'{sig:g}_{n}'] = dict(rel_median=np.nanmedian(rels, 0).tolist(), horizon_eps1e4=float(np.median(hz)),
                                   loc_err_median=float(np.median(locs)), min_band_effectivity=float(np.min(effs)),
                                   detected=len(locs))
        print(sig, n, {k: v for k, v in out[f'{sig:g}_{n}'].items() if k != 'rel_median'}, flush=True)
json.dump(dict(edges=edges.tolist(), res=out), open('results/exp05_noise.json', 'w'), indent=1)

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
uc = np.sqrt(edges[:-1] * edges[1:])
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
cols = {0.0: '#126', 1e-10: '#37b', 1e-7: '#d62', 1e-4: '#888'}
for sig in SIG:
    ax[0].loglog(uc, out[f'{sig:g}_512']['rel_median'], '-o', ms=3, color=cols[sig], label=f'sigma={sig:g}')
ax[0].set(xlabel='u', ylabel='band relative error', title='SAFI adaptive (q=40), n=512, noisy samples')
ax[0].legend(fontsize=8)
for sig in SIG[1:]:
    ax[1].loglog(NS, [out[f'{sig:g}_{n}']['horizon_eps1e4'] for n in NS], '-o', color=cols[sig], label=f'sigma={sig:g}')
ax[1].loglog(NS, 3e2 * np.sqrt(NS), 'k:', label='~sqrt(n)')
ax[1].set(xlabel='n', ylabel='horizon u*(1e-4)', title='noise horizon')
ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig('figs/exp05_noise.png', dpi=140)
